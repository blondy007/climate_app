"""Find wet episodes from systematic hourly accumulation samples, resumably."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
import argparse
import json

from .data import SITES, atomic_json, collect_frame, list_day, utcnow


def scan(root, days=7, workers=3):
    root = Path(root)
    selected = []
    today = utcnow().date()
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        catalog = list_day(day)
        atomic_json(root / "catalogs" / (day.isoformat() + ".json"), catalog)
        selected.extend(item for item in catalog if item["product"] == "ACRR" and item["stamp"].endswith("00"))
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect_frame, root, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"stamp": item["stamp"], "error": type(exc).__name__, "message": str(exc)[:120]})
            if (len(results) + len(errors)) % 20 == 0:
                print(json.dumps({"hourly_samples": len(results), "errors": len(errors), "total": len(selected)}), flush=True)
    results.sort(key=lambda row: row["nominal_end"])
    summary = {"generated_at": utcnow().isoformat(), "samples": len(results), "errors": errors, "sites": {}}
    for site in SITES:
        wet = [{"time": row["nominal_end"], **row["sites"][site.id]} for row in results
               if row["sites"][site.id]["nearby_max"] is not None and row["sites"][site.id]["nearby_max"] >= .1]
        summary["sites"][site.id] = {"wet_hourly_samples": len(wet), "wet": wet}
    atomic_json(root / "scan.json", summary)
    print(json.dumps({"samples": len(results), "errors": len(errors), "wet": {key: len(value["wet"]) for key, value in summary["sites"].items()}}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--days", type=int, default=7, choices=range(1, 15))
    args = parser.parse_args()
    scan(args.root, args.days)
