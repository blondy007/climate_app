from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.nowcast.data import atomic_json
from research.nowcast.health import healthy
from research.nowcast.live import LimitReached, finished, guard_budget, session, verify


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)

    def test_restart_cannot_extend_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            end = self.now + timedelta(days=7)
            first = session(root, end, self.now)
            self.assertEqual(session(root, end, self.now + timedelta(days=1)), first)
            with self.assertRaisesRegex(ValueError, "cannot be changed"):
                session(root, end + timedelta(days=1), self.now + timedelta(days=1))
            self.assertFalse(finished(first, end - timedelta(seconds=1)))
            self.assertTrue(finished(first, end))

    def test_new_experiment_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            for end in (self.now, self.now + timedelta(days=8)):
                with self.assertRaises(ValueError):
                    session(root, end, self.now)

    def test_network_cap_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as root:
            saved = Path(root) / "kept.json"
            saved.write_text("original")
            with self.assertRaisesRegex(LimitReached, "download_cap"):
                guard_budget(root, 11_999_999_999, 2)
            self.assertEqual(saved.read_text(), "original")

    def test_health_requires_actual_progress_and_recent_heartbeat(self):
        status = {"status": "collecting", "heartbeat_at": self.now.isoformat(),
                  "last_success_at": self.now.isoformat()}
        self.assertTrue(healthy(status, self.now + timedelta(minutes=5)))
        self.assertFalse(healthy(status, self.now + timedelta(minutes=31)))
        self.assertFalse(healthy({**status, "last_success_at": None}, self.now))
        self.assertFalse(healthy({**status, "status": "completed"}, self.now))

    def test_verification_never_interpolates_truth_or_fills_missing_with_dry(self):
        with tempfile.TemporaryDirectory() as root:
            point = {"valid_time": (self.now + timedelta(minutes=15)).isoformat(),
                     "lead_minutes": 15, "rain_rate": 1, "persistence_rate": 0}
            record = {"site": "los-alcazares", "points": [point]}
            batch = {"recorded_at": self.now.isoformat(), "forecasts": [record]}
            atomic_json(Path(root) / "prospective/issued.json", batch)
            result = verify(root, self.now + timedelta(hours=1))
            self.assertEqual(result["pending_points"], 1)
            self.assertEqual(result["verified_points"], 0)
            key = "20261020T1215"
            metadata = {key: {"sites": {"los-alcazares": {"centre": [0, 0]}}}}
            with patch("research.nowcast.live.metadata_index", return_value=metadata), \
                 patch("research.nowcast.live.load_frame", return_value=({"los-alcazares": np.array([[np.nan]])}, {})):
                result = verify(root, self.now + timedelta(hours=1))
            self.assertEqual(result["results"][0]["nowcast"]["unavailable"], 1)
            self.assertEqual(result["results"][0]["nowcast"]["correct_dry"], 0)
            point["valid_time"] = (self.now + timedelta(minutes=5)).isoformat()
            atomic_json(Path(root) / "prospective/issued.json", batch)
            self.assertEqual(verify(root, self.now)["unsupported_non_15_minute_targets"], 1)

    def test_late_record_cannot_count_as_prospective(self):
        with tempfile.TemporaryDirectory() as root:
            batch = {"recorded_at": self.now.isoformat(), "forecasts": [
                {"site": "los-alcazares", "points": [{"valid_time": self.now.isoformat()}]}]}
            atomic_json(Path(root) / "prospective/late.json", batch)
            with self.assertRaisesRegex(ValueError, "before verification"):
                verify(root, self.now + timedelta(hours=1))
