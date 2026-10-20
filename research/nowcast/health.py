"""Check capture progress, not the scientific validity of a forecast."""
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json

from .data import utcnow


def healthy(status, now):
    if status.get("status") not in ("collecting", "degraded"):
        return False
    heartbeat = datetime.fromisoformat(status["heartbeat_at"])
    if not timedelta(0) <= now - heartbeat < timedelta(minutes=20):
        return False
    success = status.get("last_success_at")
    return bool(success and timedelta(0) <= now - datetime.fromisoformat(success) < timedelta(minutes=30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    try:
        value = json.loads((Path(args.root) / "live-status.json").read_text())
        ok = healthy(value, utcnow())
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    raise SystemExit(0 if ok else 1)
