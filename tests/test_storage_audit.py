"""Privacy and read-only behavior of the optional aggregate acceptance tool."""
import hashlib
from contextlib import closing
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('storage_audit',Path(__file__).resolve().parents[1]/'scripts/audit_forecast_storage.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)

class AggregateAuditTests(unittest.TestCase):
    def test_latest_active_runs_only_and_no_saved_identifiers_are_emitted(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic.sqlite3'
            with closing(sqlite3.connect(path)) as c:
                c.executescript("""
                    CREATE TABLE locations(id TEXT,active INTEGER);
                    CREATE TABLE runs(id INTEGER,location_id TEXT,provider TEXT,fetched_at TEXT);
                    CREATE TABLE points(run_id INTEGER,valid_time TEXT);
                    INSERT INTO locations VALUES('private-location-a',1),('private-removed-b',0);
                    INSERT INTO runs VALUES
                      (1,'private-location-a','weathernext3','2026-11-03T00:00:00+00:00'),
                      (2,'private-location-a','weathernext3','2026-11-04T00:00:00+00:00'),
                      (3,'private-removed-b','weathernext3','2026-11-04T00:00:00+00:00');
                    INSERT INTO points VALUES
                      (1,'2026-11-06T00:00:00+00:00'),
                      (2,'2026-11-05T00:00:00+00:00'),
                      (2,'2026-11-03T00:00:00+00:00'),
                      (3,'2026-11-05T00:00:00+00:00');
                """)
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            report=audit.aggregate_report(path,datetime(2026,11,4,tzinfo=timezone.utc))
            self.assertEqual(report['active_places'],1)
            self.assertEqual(len(report['latest_forecasts']),1)
            row=report['latest_forecasts'][0]
            self.assertEqual(row['future_points'],1)
            self.assertEqual(row['fetched_at'],'2026-11-04T00:00:00+00:00')
            self.assertEqual(row['last_valid_time'],'2026-11-05T00:00:00+00:00')
            self.assertNotIn('private-location',json.dumps(report))
            self.assertNotIn('private-removed',json.dumps(report))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_missing_database_is_never_created(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'missing.sqlite3'
            with self.assertRaises(sqlite3.OperationalError):
                audit.aggregate_report(path)
            self.assertFalse(path.exists())

if __name__=='__main__': unittest.main()
