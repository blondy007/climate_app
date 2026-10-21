import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from climate_app.forecast.domain import Forecast, stamp, utcnow
from climate_app.forecast.refresh import refresh_saved_forecast
from climate_app.forecast.store import Store


class ManualRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "forecasts.sqlite3"
        self.store = Store(self.path)
        self.place = self.store.add_location("Torrejón", 40.46, -3.47)
        self.other = self.store.add_location("Otro", 37.76, -.84)
        self.now = utcnow()
        self.when = self.now + timedelta(hours=2)
        for identity in (self.place, self.other):
            for provider in ("ecmwf", "icon", "gfs", "metno", "weathernext3"):
                self.store.save_forecast(identity, Forecast(provider, self.now - timedelta(hours=1), [dict(
                    valid_time=self.when, temperature=100 if provider == "weathernext3" else 10,
                    humidity=60, wind_speed=12)]))

    def response(self, url, **kwargs):
        if "met.no" in url:
            data = {"properties": {"timeseries": [{"time": stamp(self.when), "data": {
                "instant": {"details": {"air_temperature": 24, "relative_humidity": 60,
                                        "wind_speed": 3}}}}]}}
        else:
            data = {"hourly": {"time": [self.when.timestamp()], "temperature_2m": [24],
                               "relative_humidity_2m": [60], "wind_speed_10m": [12]}}
        return Mock(status_code=200, headers={}, json=lambda: data)

    def test_refresh_bypasses_open_cache_updates_only_chosen_place_and_public_consensus(self):
        self.store.set_state(f"http:ecmwf:{self.place}", {
            "data": {"hourly": {"time": [self.when.timestamp()], "temperature_2m": [11]}},
            "expires": stamp(self.now + timedelta(hours=3))})
        with patch("climate_app.forecast.providers.requests.get", side_effect=self.response) as get:
            result = refresh_saved_forecast(self.store, self.place)
        self.assertEqual(set(result.updated), {"ecmwf", "icon", "gfs", "metno"})
        self.assertEqual(get.call_count, 4)
        current = self.store.forecasts(self.place)
        self.assertEqual(current[current.provider == "ecmwf"].temperature.tolist(), [24])
        self.assertEqual(current[current.provider == "climateapp"].temperature.tolist(), [24])
        self.assertEqual(current[current.provider == "weathernext3"].temperature.tolist(), [100])
        self.assertEqual(len(self.store.forecasts(self.other, latest=False)), 5)
        self.assertIsNone(self.store.state("collector"))
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT used FROM budget WHERE service='open-meteo'").fetchone()[0], 96)
        self.assertEqual(len(self.store.forecasts(self.place, latest=False)), 10)

    def test_cooldown_is_atomic_shared_persistent_and_rejects_inactive_or_unknown_places(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _: Store(self.path).claim_manual_refresh(self.place, self.now), range(2)))
        self.assertEqual(claims.count(None), 1)
        with patch("climate_app.forecast.providers.requests.get") as get:
            result = refresh_saved_forecast(Store(self.path), self.place, self.now + timedelta(minutes=1))
        self.assertIsNotNone(result.retry_at)
        get.assert_not_called()
        self.assertIsNone(self.store.claim_manual_refresh(self.place, self.now + timedelta(minutes=16)))
        self.store.deactivate(self.other)
        for identity in (self.other, "unknown"):
            with self.assertRaises(ValueError):
                self.store.claim_manual_refresh(identity)

    def test_metno_upstream_expiry_and_legacy_cache_are_respected_without_new_snapshots(self):
        for upstream in (None, format_datetime(self.now + timedelta(hours=1), usegmt=True)):
            with self.subTest(upstream=upstream), tempfile.TemporaryDirectory() as temp:
                store = Store(Path(temp) / "test.sqlite3")
                identity = store.add_location("Test", 40, -3)
                store.set_state(f"http:metno:{identity}", {
                    "data": self.response("https://api.met.no").json(),
                    "expires": stamp(self.now + timedelta(hours=3)), "provider_expires": upstream})
                with patch("climate_app.forecast.providers.requests.get", side_effect=self.response) as get:
                    result = refresh_saved_forecast(store, identity)
                self.assertEqual(result.cached, ["metno"])
                self.assertEqual(get.call_count, 3)
                self.assertFalse((store.forecasts(identity).provider == "metno").any())

    def test_partial_failure_keeps_previous_provider_and_total_failure_keeps_entire_archive(self):
        def response(url, **kwargs):
            return Mock(status_code=503) if "dwd-icon" in url else self.response(url, **kwargs)
        with patch("climate_app.forecast.providers.requests.get", side_effect=response):
            result = refresh_saved_forecast(self.store, self.place)
        self.assertEqual(set(result.errors), {"icon"})
        self.assertEqual(self.store.forecasts(self.place).query("provider == 'icon'").temperature.tolist(), [10])
        before = self.store.forecasts(self.other, latest=False)
        with patch("climate_app.forecast.providers.requests.get", return_value=Mock(status_code=503)):
            failed = refresh_saved_forecast(self.store, self.other)
        self.assertEqual(len(failed.errors), 4)
        self.assertEqual(failed.updated, [])
        self.assertTrue(before.equals(self.store.forecasts(self.other, latest=False)))

    def test_daily_quota_cannot_be_bypassed_by_manual_refresh(self):
        self.store.reserve("open-meteo", self.now.date().isoformat(), 8000, 8000)
        with patch("climate_app.forecast.providers.requests.get", side_effect=self.response) as get:
            result = refresh_saved_forecast(self.store, self.place)
        self.assertEqual(set(result.errors), {"ecmwf", "icon", "gfs"})
        self.assertEqual(result.updated, ["metno"])
        self.assertEqual(get.call_count, 1)
        self.assertTrue(all("Cuota preventiva" in value for value in result.errors.values()))

    def test_ui_refreshes_the_selected_place_and_reports_cooldown_and_recovery_from_empty(self):
        with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(self.path)}), \
             patch("climate_app.auth.owner_config", return_value=None), \
             patch("climate_app.forecast.providers.requests.get", side_effect=self.response) as get:
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
            app.radio(key="forecast_location").set_value(self.place).run(timeout=30)
            app.button(key="refresh_saved_forecast").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="refresh_saved_forecast").disabled)
            self.assertTrue(any("Previsión consultada" in message.value for message in app.success))
            self.assertEqual(get.call_count, 4)
            self.assertFalse(any(button.key == "refresh_device_location" for button in app.button))
            empty = self.store.add_location("Sin previsión", 42, -4)
            app.run(timeout=30)
            self.assertFalse(app.exception)
            app.radio(key="forecast_location").set_value(empty).run(timeout=30)
            app.button(key="refresh_saved_forecast").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any("Previsión de Sin previsión" in m.value for m in app.markdown))


if __name__ == "__main__":
    unittest.main()
