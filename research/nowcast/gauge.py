"""Independent CHS gauge snapshots. Not used as town-centre verification truth."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import re

import requests

from .data import UTC, atomic_json, utcnow

URL = "https://saihweb.chsegura.es/apps/iVisor/graficas/graficaVar.php"
STATION = "06A18P01"
LOCATION = {"latitude": 37.716015316905775, "longitude": -.8612551779763589,
            "name": "Desembocadura de la Rambla del Albujón"}
POINT = re.compile(r"\{x:(\d+), y:(null|[\d.]+), y0:(null|[\d.]+), ac:(null|[\d.]+), fb:'([01])', pr:'([^']*)'\}")


def parse_series(html):
    # Parse data as inert text. Never evaluate downloaded JavaScript.
    head, tail = html.split("codigo:'" + STATION + "'", 1)
    if "tipoDatosMHDMA: {text: 'I'}" not in head or not tail.startswith(", um:'mm'"):
        raise ValueError("Gauge units or instantaneous source changed")
    values = tail.split("data: [", 1)[1].split("]", 1)[0]
    matches = list(POINT.finditer(values))
    if not matches or POINT.sub("", values).replace(",", "").strip():
        raise ValueError("Unrecognized gauge series format")
    zone = ZoneInfo("Europe/Madrid")
    records = []
    for item in matches:
        raw_time, _, mm, accumulated, quality, provenance = item.groups()
        # The chart x coordinates encode LOCAL wall-clock as if it were UTC.
        # Internal priFec/ultFec are real UTC epochs and validate that conversion.
        wall = datetime.fromtimestamp(int(raw_time) / 1000, UTC).replace(tzinfo=None)
        early, late = wall.replace(tzinfo=zone, fold=0), wall.replace(tzinfo=zone, fold=1)
        ambiguous = early.utcoffset() != late.utcoffset()
        value = float(mm) if mm != "null" and quality == "1" and not ambiguous else None
        records.append({"raw_chart_ms": int(raw_time), "local_wall_time": wall.isoformat(),
                        "utc": None if ambiguous else early.astimezone(UTC).isoformat(),
                        "increment_mm": value, "quality": quality, "provenance": provenance,
                        "clock_ambiguous": ambiguous})
    good = [row for row in records if row["increment_mm"] is not None]
    for prefix, row in (("pri", good[0]), ("ult", good[-1])):
        declared = int(re.search(prefix + r"Fec:(\d+)", tail)[1])
        if abs(datetime.fromisoformat(row["utc"]).timestamp() - declared) > 1:
            raise ValueError("Gauge chart clock no longer matches internal UTC epochs")
    deltas = [right["raw_chart_ms"] - left["raw_chart_ms"] for left, right in zip(records, records[1:])]
    if any(delta != 300_000 for delta in deltas):
        raise ValueError("Gauge chart no longer has a regular five-minute grid")
    declared_sum = float(re.search(r"acuVal:([\d.]+)", tail)[1])
    if abs(sum(row["increment_mm"] for row in good) - declared_sum) > .051:
        raise ValueError("Gauge increments do not match official accumulated total")
    return {"station": STATION, "location": LOCATION, "source": "CONFEDERACIÓN HIDROGRÁFICA DEL SEGURA",
            "provisional": True, "interval_minutes": 5, "records": records,
            "total_mm": declared_sum, "missing_or_unusable": len(records) - len(good),
            "clock": "Europe/Madrid local chart values validated against internal UTC epochs",
            "limitations": ["Nearby gauge, not rainfall at the town centre",
                            "Interval-end convention and tipping resolution need verification before onset scoring",
                            "Provisional source data subject to review"],
            "approved_for_public_use": False}


def capture(root, start, end):
    if end <= start or end - start > timedelta(days=4):
        raise ValueError("Gauge snapshots are limited to four days")
    zone = ZoneInfo("Europe/Madrid")
    params = {"puntos": STATION, "dfrom": start.astimezone(zone).strftime("%d/%m/%Y %H:%M"),
              "dto": end.astimezone(zone).strftime("%d/%m/%Y %H:%M"), "source": "I",
              "VerNoFiables": "N", "VerLeyendas": "N"}
    with requests.get(URL, params=params, timeout=(10, 45), stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("Unexpected gauge redirect")
        body = bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk)
            if len(body) > 1_000_000:
                raise ValueError("Gauge response too large")
        result = parse_series(body.decode("utf-8", errors="replace"))
        result.update(url=response.url, retrieved_at=utcnow().isoformat(), sha256_source=hashlib.sha256(body).hexdigest())
    dest = Path(root) / "gauges" / (STATION + "-" + utcnow().strftime("%Y%m%dT%H%M%S"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".html").write_bytes(body)
    atomic_json(dest.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    value = capture(args.root, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))
    print({"station": value["station"], "records": len(value["records"]),
           "total_mm": value["total_mm"], "missing_or_unusable": value["missing_or_unusable"]})
