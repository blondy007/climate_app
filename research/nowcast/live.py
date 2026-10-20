"""Seven-day, resource-bounded research collector. Never serves public forecasts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .data import SITES, atomic_json, collect_frame, frame_path, list_day, load_frame, parse_stamp, utcnow
from .model import ALGORITHM, categorical, forecast
from .replay import choose_sources, metadata_index

MAX_STORAGE_MB = 750
MAX_DOWNLOAD_BYTES = 12_000_000_000
INTERVAL_SECONDS = 300


class LimitReached(RuntimeError):
    pass


def session(root, expires_at, now):
    """A restart cannot extend the experiment or reset its network budget."""
    root = Path(root)
    path = root / "live-config.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if datetime.fromisoformat(value["expires_at"]) != expires_at:
            raise ValueError("Existing experiment deadline cannot be changed")
        return value
    if not now < expires_at <= now + timedelta(days=7):
        raise ValueError("New experiments must expire within seven days")
    value = {"started_at": now.isoformat(), "expires_at": expires_at.isoformat(),
             "max_storage_mb": MAX_STORAGE_MB, "max_download_bytes": MAX_DOWNLOAD_BYTES,
             "interval_seconds": INTERVAL_SECONDS, "algorithm": ALGORITHM,
             "approved_for_public_use": False,
             "model_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest()}
    atomic_json(path, value)
    return value


def finished(config, now):
    return now >= datetime.fromisoformat(config["expires_at"])


def guard_budget(root, used_download_bytes, next_bytes=0):
    if used_download_bytes + next_bytes > MAX_DOWNLOAD_BYTES:
        raise LimitReached("download_cap")
    used = sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
    # Reserve space for the next crop, metadata, forecast and status write.
    if used + 2_000_000 > MAX_STORAGE_MB * 1024**2:
        raise LimitReached("storage_cap")
    return used


def verify(root, now):
    """Only exact radar timestamp matches; no interpolation of verification truth."""
    root = Path(root)
    rate = metadata_index(root, "RATE")
    rows, pending, unsupported, records = [], 0, 0, 0
    from functools import lru_cache
    @lru_cache(maxsize=24)
    def read(key):
        return load_frame(root, "RATE", key)[0]
    for path in sorted((root / "prospective").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        for record in batch["forecasts"]:
            records += 1
            recorded = datetime.fromisoformat(batch["recorded_at"])
            for point in record["points"]:
                target = datetime.fromisoformat(point["valid_time"])
                if recorded >= target:
                    raise ValueError("Forecast was not recorded before verification time")
                if target.minute % 15 or target.second:
                    unsupported += 1
                    continue
                key = target.strftime("%Y%m%dT%H%M")
                if target > now or key not in rate:
                    pending += 1
                    continue
                meta = rate[key]
                cy, cx = meta["sites"][record["site"]]["centre"]
                value = read(key)[record["site"]][cy, cx]
                rows.append({"site": record["site"], **point,
                             "observed_rate": float(value) if np.isfinite(value) else None})
    summaries = []
    for site in SITES:
        for lead in (15, 30, 60):
            selected = [row for row in rows if row["site"] == site.id and row["lead_minutes"] == lead]
            summaries.append({"site": site.id, "lead_minutes": lead, "cases": len(selected),
                              "nowcast": categorical(selected),
                              "persistence": categorical(selected, "persistence_rate")})
    result = {"generated_at": now.isoformat(), "algorithm": ALGORITHM, "mode": "prospective",
              "approved_for_public_use": False, "forecast_site_records": records,
              "verified_points": len(rows), "pending_points": pending,
              "unsupported_non_15_minute_targets": unsupported, "results": summaries,
              "reference": "later OPERA RATE, not an independent rain gauge",
              "limitations": ["Overlapping forecasts are not independent events",
                              "5-minute and 1-minute onset timing are not verified by 15-minute RATE"]}
    atomic_json(root / "prospective-report.json", result)
    return result


def cycle(root, state, deadline):
    root = Path(root)
    now = utcnow()
    errors = []
    count = 0
    # Two catalog requests cover midnight. Only the recent two hours are fetched.
    catalog = []
    for day in sorted({now.date(), (now - timedelta(hours=2)).date()}):
        try:
            catalog.extend(list_day(day, "openradar-24h"))
        except Exception as exc:
            errors.append({"stage": "catalog", "error": type(exc).__name__})
    selected = [item for item in catalog if item["product"] in ("DBZH", "RATE")
                and now - timedelta(hours=2) <= parse_stamp(item["stamp"]) <= now]
    for item in selected:
        if utcnow() >= deadline:
            break
        path = frame_path(root, item["product"], item["stamp"])
        if path.exists() and path.with_suffix(".json").exists():
            continue
        guard_budget(root, state["reserved_download_bytes"], item["bytes"] * 3)
        # Reserve all three possible attempts before the request. Failed attempts
        # remain charged to the budget across restarts (a conservative upper bound).
        state["reserved_download_bytes"] += item["bytes"] * 3
        atomic_json(root / "live-budget.json", {"reserved_download_bytes": state["reserved_download_bytes"]})
        try:
            collect_frame(root, item)
            count += 1
        except Exception as exc:
            errors.append({"stage": "download", "product": item["product"],
                           "stamp": item["stamp"], "error": type(exc).__name__})
    if utcnow() >= deadline:
        return {"downloaded": count, "errors": errors, "forecast_sites": 0}
    dbzh, rate = metadata_index(root, "DBZH"), metadata_index(root, "RATE")
    issue = utcnow()  # Never backdate a prospective issue to the observation time.
    selection = choose_sources(dbzh, rate, issue, False)
    forecasts, unavailable = [], Counter()
    if selection:
        keys, rkey = selection
        images = [load_frame(root, "DBZH", key)[0] for key in keys]
        rain = load_frame(root, "RATE", rkey)[0]
        for site in SITES:
            try:
                forecasts.append(forecast([image[site.id] for image in images], [dbzh[key] for key in keys],
                                          rain[site.id], rate[rkey], site.id, issue,
                                          leads=tuple(range(5, 61, 5)), replay=False))
            except (ValueError, RuntimeError) as exc:
                unavailable[str(exc)] += 1
    else:
        unavailable["missing_available_sequence"] += len(SITES)
    recorded_at = utcnow()
    if any(datetime.fromisoformat(point["valid_time"]) <= recorded_at
           for record in forecasts for point in record["points"]):
        raise RuntimeError("Forecast computation exceeded its first target time")
    batch = {"issued_at": issue.isoformat(), "recorded_at": recorded_at.isoformat(),
             "forecasts": forecasts, "unavailable": dict(unavailable), "approved_for_public_use": False}
    filename = issue.strftime("%Y%m%dT%H%M%S%f") + ".json"
    # Exclusive creation preserves issued predictions; verification never rewrites them.
    directory = root / "prospective"
    directory.mkdir(exist_ok=True)
    with (directory / filename).open("x", encoding="utf-8") as out:
        json.dump(batch, out, ensure_ascii=False, allow_nan=False)
    report = verify(root, utcnow())
    return {"downloaded": count, "errors": errors, "forecast_sites": len(forecasts),
            "unavailable": dict(unavailable), "verified_points": report["verified_points"],
            "latest_dbzh": max(dbzh, default=None), "latest_rate": max(rate, default=None)}


def run(root, expires_at):
    import fcntl  # Deployment is Linux; prevents concurrent writers after a restart.
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = session(root, expires_at, utcnow())
        budget = root / "live-budget.json"
        state = json.loads(budget.read_text()) if budget.exists() else {"reserved_download_bytes": 0}
        last_success = None
        while not finished(config, utcnow()):
            started = utcnow()
            base = {"heartbeat_at": started.isoformat(), "expires_at": config["expires_at"],
                    "algorithm": ALGORITHM, "approved_for_public_use": False,
                    "last_success_at": last_success, "status": "collecting"}
            atomic_json(root / "live-status.json", base)
            try:
                used = guard_budget(root, state["reserved_download_bytes"])
                result = cycle(root, state, expires_at)
                if result["downloaded"] or (not result["errors"] and result.get("forecast_sites")):
                    last_success = utcnow().isoformat()
                base.update(result, storage_bytes=used, last_success_at=last_success,
                            status="degraded" if result["errors"] else "collecting")
            except LimitReached as exc:
                base.update(status="stopped", reason=str(exc), heartbeat_at=utcnow().isoformat())
                atomic_json(root / "live-status.json", base)
                print(json.dumps(base), flush=True)
                return
            except Exception as exc:
                base.update(status="degraded", error=type(exc).__name__)
            base.update(heartbeat_at=utcnow().isoformat(), reserved_download_bytes=state["reserved_download_bytes"])
            atomic_json(root / "live-status.json", base)
            print(json.dumps(base), flush=True)
            remaining = min(INTERVAL_SECONDS - (utcnow() - started).total_seconds(),
                            (expires_at - utcnow()).total_seconds())
            if remaining > 0:
                time.sleep(remaining)
        atomic_json(root / "live-status.json", {"status": "completed", "reason": "deadline",
                    "heartbeat_at": utcnow().isoformat(), "expires_at": config["expires_at"],
                    "approved_for_public_use": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expires-at", required=True)
    args = parser.parse_args()
    run(args.root, datetime.fromisoformat(args.expires_at))
