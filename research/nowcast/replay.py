"""Availability-aware replay. Fixed algorithm; last day held out from development."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .data import SITES, UTC, atomic_json, load_frame, parse_stamp, stamp, utcnow
from .model import ALGORITHM, LEADS, availability, categorical, forecast


def metadata_index(root, product):
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (Path(root)/"frames"/product).glob("*.json")}


def choose_sources(dbzh, rate, issued_at, replay):
    allowed = [key for key, meta in dbzh.items() if availability(meta,replay) <= issued_at]
    if not allowed:
        return None
    latest = max(allowed)
    last = parse_stamp(latest)
    keys = [stamp(last-timedelta(minutes=offset)) for offset in (10,5,0)]
    if any(key not in dbzh or availability(dbzh[key],replay)>issued_at for key in keys):
        return None
    allowed_rate = [key for key, meta in rate.items() if availability(meta,replay) <= issued_at]
    if not allowed_rate:
        return None
    return keys, max(allowed_rate)


def summarize(rows, failures, attempts, started, source_count):
    result = {"generated_at":utcnow().isoformat(),"algorithm":ALGORITHM,"approved_for_public_use":False,
              "availability_rule":"archive_last_modified_plus_2_minutes", "reference":"later OPERA rain-rate estimate; not an independent rain gauge",
              "source_files":source_count,"elapsed_seconds":round(time.monotonic()-started,2),
              "failed_issue_sites":dict(failures),"results":[],
              "limitations":["Only three days; events and overlapping horizons are not independent samples",
                             "Radar verification is not independent ground truth",
                             "Onset/end timing below 15 minutes remains unvalidated",
                             "Retrospective publication metadata is a conservative proxy, not a prospective forecast record"]}
    for site in SITES:
        for split in ("development","validation"):
            for lead in LEADS:
                selected=[row for row in rows if row["site"]==site.id and row["split"]==split and row["lead_minutes"]==lead]
                paired=[row for row in selected if all(row.get(key) is not None and np.isfinite(row[key])
                        for key in ("rain_rate","persistence_rate","observed_rate"))]
                emitted=sum(row["forecast_emitted"] for row in selected)
                result["results"].append({"site":site.id,"split":split,"lead_minutes":lead,
                    "cases":len(selected),"attempted_origins":attempts[(site.id,split)],
                    "forecast_origins":emitted,"availability":emitted/len(selected) if selected else None,
                    "paired_cases":len(paired),"nowcast":categorical(paired),
                    "persistence":categorical(paired,"persistence_rate"),
                    "nowcast_all":categorical(selected),"persistence_all":categorical(selected,"persistence_rate")})
    return result


def replay(root, start, end, validation_start):
    root=Path(root)
    dbzh,rate=metadata_index(root,"DBZH"),metadata_index(root,"RATE")
    started=time.monotonic()
    failures=Counter()
    attempts=Counter()
    rows, forecasts=[],[]
    @lru_cache(maxsize=24)
    def read(product,key):
        return load_frame(root,product,key)[0]
    issue=start
    count=0
    while issue<end:
        selection=choose_sources(dbzh,rate,issue,True)
        split="validation" if issue>=validation_start else "development"
        for site in SITES:
            attempts[(site.id,split)]+=1
            record=None
            reason=None
            if selection is None:
                failures["missing_available_sequence"]+=1
                reason="missing_available_sequence"
            else:
                keys,rkey=selection
                try:
                    record=forecast([read("DBZH",key)[site.id] for key in keys], [dbzh[key] for key in keys],
                                    read("RATE",rkey)[site.id],rate[rkey],site.id,issue,leads=LEADS,replay=True)
                except (ValueError,RuntimeError) as exc:
                    reason=str(exc)
                    failures[reason]+=1
            if record:
                record["split"]=split
                forecasts.append(record)
                points=record["points"]
            else:
                # Preserve rejected origins in availability and all-case metrics.
                # The baseline needs only the latest available, fresh point RATE.
                allowed=[key for key,meta in rate.items() if availability(meta,True)<=issue
                         and issue-datetime.fromisoformat(meta["nominal_end"])<=timedelta(minutes=45)]
                persistence=None
                if allowed:
                    key=max(allowed)
                    cy,cx=rate[key]["sites"][site.id]["centre"]
                    value=read("RATE",key)[site.id][cy,cx]
                    persistence=float(value) if np.isfinite(value) else None
                points=[{"lead_minutes":lead,"valid_time":(issue+timedelta(minutes=lead)).isoformat(),
                         "rain_rate":None,"persistence_rate":persistence} for lead in LEADS]
            for point in points:
                # Observation lookup happens only after the forecast function has returned.
                target=datetime.fromisoformat(point["valid_time"])
                tkey=stamp(target)
                truth=None
                if tkey in rate:
                    field=read("RATE",tkey)[site.id]
                    cy,cx=rate[tkey]["sites"][site.id]["centre"]
                    truth=float(field[cy,cx]) if np.isfinite(field[cy,cx]) else None
                rows.append({"site":site.id,"split":split,"issued_at":issue.isoformat(),**point,
                             "forecast_emitted":bool(record),"rejection":reason,"observed_rate":truth})
        count+=1
        if count%24==0:
            print(json.dumps({"issued_at":issue.isoformat(),"issue_times":count,"forecast_site_records":len(forecasts),"failures":sum(failures.values())}),flush=True)
        issue+=timedelta(minutes=15)
    report=summarize(rows,failures,attempts,started,len(dbzh)+len(rate))
    report.update(start=start.isoformat(),end=end.isoformat(),validation_start=validation_start.isoformat(),
                  model_sha256=hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest())
    atomic_json(root/"retrospective-predictions.json",forecasts)
    atomic_json(root/"retrospective-verification.json",rows)
    atomic_json(root/"retrospective-report.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--start",required=True)
    parser.add_argument("--end",required=True)
    parser.add_argument("--validation-start",required=True)
    args=parser.parse_args()
    replay(args.root,datetime.fromisoformat(args.start).replace(tzinfo=UTC),datetime.fromisoformat(args.end).replace(tzinfo=UTC),datetime.fromisoformat(args.validation_start).replace(tzinfo=UTC))
