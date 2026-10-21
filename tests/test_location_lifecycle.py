import os
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from climate_app.forecast.automatic import AUTOMATIC, device_location
from climate_app.forecast.domain import Forecast, utcnow
from climate_app.forecast.store import Store
from climate_app.ui import browser_location


class LocationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "locations.sqlite3"
        self.store = Store(self.path)
        self.first = self.store.add_location("Torrejón de Ardoz", 40.4594, -3.4697)
        self.second = self.store.add_location("Torrejón de Ardoz", 40.46, -3.47)
        self.now = utcnow()

    def archive(self, identity):
        self.store.save_forecast(identity, Forecast("ecmwf", self.now, [dict(
            valid_time=self.now + timedelta(hours=2), temperature=21, source_init_time=self.now)]))
        self.store.save_observations(identity, "test", [dict(valid_time=self.now, temperature=20)])
        self.store.set_alias(identity, "Alias")
        self.store.set_state(f"task:{identity}:ecmwf", {"status": "ok"})
        self.store.set_state(f"http:ecmwf:{identity}", {"data": "fixture"})

    def test_delete_only_the_chosen_inactive_duplicate_and_block_late_collector_writes(self):
        for identity in (self.first, self.second):
            self.archive(identity)
        self.store.reserve("open-meteo", "today", 32, 8000)
        self.store.set_state("collector", {"heartbeat": "fixture"})
        with self.assertRaises(ValueError):
            self.store.delete_location(self.first)
        self.store.deactivate(self.first)
        self.store.delete_location(self.first)
        self.store.save_forecast(self.first, Forecast("ecmwf", self.now, [dict(valid_time=self.now + timedelta(hours=3))]))
        self.store.save_observations(self.first, "test", [dict(valid_time=self.now)])
        self.store.set_state(f"task:{self.first}:ecmwf", {"late": True})
        self.store.set_state(f"http:ecmwf:{self.first}", {"late": True})
        self.assertEqual([loc.id for loc in self.store.locations(True)], [self.second])
        self.assertEqual(len(self.store.forecasts(self.second)), 1)
        self.assertEqual(len(self.store.observations(self.second)), 1)
        with self.store.connect() as conn:
            for table in ("runs", "points", "point_sources", "observations", "location_aliases"):
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(conn.execute("SELECT used FROM budget").fetchone()[0], 32)
        self.assertIsNone(self.store.state(f"task:{self.first}:ecmwf"))
        self.assertIsNone(self.store.state(f"http:ecmwf:{self.first}"))
        self.assertEqual(self.store.state("collector"), {"heartbeat": "fixture"})

    def test_failed_delete_rolls_back_the_entire_archive(self):
        self.archive(self.first)
        self.store.deactivate(self.first)
        with self.store.connect() as conn:
            conn.execute("CREATE TRIGGER block_test_delete BEFORE DELETE ON points BEGIN SELECT RAISE(ABORT,'fixture'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.delete_location(self.first)
        self.assertEqual(len(self.store.forecasts(self.first)), 1)
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM point_sources").fetchone()[0], 1)

    def test_last_deleted_place_does_not_reappear_after_restart_or_backup_restore(self):
        self.store.seed()
        for identity in (self.first, self.second):
            self.store.deactivate(identity)
            self.store.delete_location(identity)
        restarted = Store(self.path)
        restarted.seed()
        self.assertEqual(restarted.locations(True), [])
        restored = Store(restarted.backup())
        restored.seed()
        self.assertEqual(restored.locations(True), [])

    def test_preferences_restore_active_place_into_both_selection_keys_and_ignore_removed(self):
        with patch.object(browser_location, "_bridge", return_value={"kind": "preferences", "nonce": "restore", "place_id": self.second}), \
             patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(self.path)}), \
             patch("climate_app.auth.owner_config", return_value=None):
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="forecast_location").value, self.second)
        self.assertEqual(app.session_state["selected_forecast_location"], self.second)
        self.store.deactivate(self.second)
        state = {}
        with patch("climate_app.forecast.store.Store", return_value=self.store):
            browser_location._process_event({"kind": "preferences", "place_id": self.second}, state)
        self.assertNotIn("selected_forecast_location", state)
        state = {"browser_forecast_preferences_loaded": True, "selected_forecast_location": self.first}
        browser_location._process_event({"kind": "preferences", "automatic": True}, state)
        self.assertEqual(state["selected_forecast_location"], self.first)

    def test_name_updates_only_matching_automatic_location_and_survives_next_position(self):
        payload = dict(latitude=40.4594, longitude=-3.4697, accuracy=20,
                       timestamp=self.now.timestamp() * 1000, timezone="Europe/Madrid")
        location = device_location(payload, self.now)
        state = {"selected_forecast_location": AUTOMATIC, "device_location": location}
        event = dict(kind="place_name", latitude=location.latitude, longitude=location.longitude, name="Torrejón de Ardoz")
        browser_location._process_event(event, state)
        self.assertEqual(state["device_location"].name, "Torrejón de Ardoz")
        browser_location._process_event({"kind": "position", **payload}, state)
        self.assertEqual(state["device_location"].name, "Torrejón de Ardoz")
        browser_location._process_event({**event, "latitude": 37, "name": "Wrong city"}, state)
        self.assertEqual(state["device_location"].name, "Torrejón de Ardoz")
        state["selected_forecast_location"] = self.first
        browser_location._process_event({**event, "name": "Late city"}, state)
        self.assertEqual(state["device_location"].name, "Torrejón de Ardoz")

    def test_ui_delete_requires_owner_confirmation_and_can_be_cancelled(self):
        self.archive(self.first)
        self.archive(self.second)
        self.store.deactivate(self.second)
        with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(self.path)}), \
             patch("climate_app.auth.owner_config", return_value=None), \
             patch("climate_app.ui.forecast_dashboard.authenticated", return_value=False) as owner:
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
            app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
            app.button(key=f"delete_place_{self.second}").click().run(timeout=30)
            self.assertFalse(any(b.key == f"confirm_delete_place_{self.second}" for b in app.button))
            self.assertEqual(len(self.store.locations(True)), 2)
            app.button(key=f"cancel_delete_place_{self.second}").click().run(timeout=30)
            owner.return_value = True
            app.button(key=f"delete_place_{self.second}").click().run(timeout=30)
            self.assertEqual(len(self.store.locations(True)), 2)
            app.button(key=f"confirm_delete_place_{self.second}").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual([loc.id for loc in self.store.locations(True)], [self.first])
            self.assertEqual(len(self.store.forecasts(self.first)), 1)
