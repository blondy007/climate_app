"""Download a bounded, resumable case study; only regional crops are retained."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json

from .data import UTC, atomic_json, collect_frame, list_day, parse_stamp


def backfill(root, start, end):
    if end <= start or end-start > timedelta(days=4):
        raise ValueError("A case-study download is limited to four days")
    root = Path(root)
    selected = []
    day = start.date()
    while day <= end.date():
        cache = root / "catalogs" / (day.isoformat()+".json")
        catalog = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else list_day(day)
        atomic_json(cache, catalog)
        selected.extend(item for item in catalog if item["product"] in ("DBZH", "RATE") and start <= parse_stamp(item["stamp"]) < end)
        day += timedelta(days=1)
    total_bytes = sum(item["bytes"] for item in selected)
    if total_bytes > 4_500_000_000:
        raise ValueError("Case-study network cap exceeded")
    print(json.dumps({"phase":"backfill","files":len(selected),"maximum_download_bytes":total_bytes}),flush=True)
    done, errors = 0, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(collect_frame,root,item):item for item in selected}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                errors.append({"stamp":pending[future]["stamp"],"product":pending[future]["product"],"error":type(exc).__name__,"message":str(exc)[:130]})
            done += 1
            if done % 80 == 0:
                print(json.dumps({"downloaded_or_cached":done,"total":len(selected),"errors":len(errors)}),flush=True)
    atomic_json(root/"backfill.json",{"start":start.isoformat(),"end":end.isoformat(),"files":len(selected),"errors":errors,"source_bytes":total_bytes})
    print(json.dumps({"finished":done,"errors":len(errors)}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--start",required=True)
    parser.add_argument("--end",required=True)
    args=parser.parse_args()
    backfill(args.root,datetime.fromisoformat(args.start).replace(tzinfo=UTC),datetime.fromisoformat(args.end).replace(tzinfo=UTC))
