"""Optional production aggregate audit. Requires separate explicit approval.

Reads only provider names, run timestamps and future-point counts. Never exports
the database, coordinates, weather values, account tables or credential files.
Running it against production is not authorized merely by adding this file.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time


def aggregate_report(path, now=None):
    path=Path(path).resolve()
    now=now or datetime.now(timezone.utc)
    cutoff=now.isoformat()
    deadline=time.monotonic()+15
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as connection:
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        connection.set_progress_handler(lambda:int(time.monotonic()>deadline),10000)
        rows=connection.execute("""
            SELECT r.location_id,r.provider,r.fetched_at,COUNT(p.valid_time) future_points,
                   MIN(p.valid_time) first_valid_time,MAX(p.valid_time) last_valid_time
            FROM runs r JOIN locations l ON l.id=r.location_id
            LEFT JOIN points p ON p.run_id=r.id AND p.valid_time>?
            WHERE l.active=1 AND r.id IN (
                SELECT MAX(rr.id) FROM runs rr JOIN locations ll ON ll.id=rr.location_id
                WHERE ll.active=1 GROUP BY rr.location_id,rr.provider)
            GROUP BY r.id ORDER BY r.location_id,r.provider
        """,(cutoff,)).fetchall()
        count=connection.execute('SELECT count(*) FROM locations WHERE active=1').fetchone()[0]
    # Stable local numbers avoid emitting location IDs or saved names.
    identifiers={value:index+1 for index,value in enumerate(sorted({r['location_id'] for r in rows}))}
    return {'checked_at':cutoff,'active_places':count,'read_only':True,
            'latest_forecasts':[{'place_number':identifiers[r['location_id']],
                **{key:r[key] for key in ('provider','fetched_at','future_points','first_valid_time','last_valid_time')}}
                for r in rows]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--allow-production-aggregate-read',action='store_true',
                        help='Use only after explicit approval for this production read')
    args=parser.parse_args()
    if not args.allow_production_aggregate_read:
        parser.error('Explicitly approved aggregate access is required')
    print(json.dumps(aggregate_report(args.database)))
