import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from climate_app.forecast.automatic import AUTOMATIC, SessionCache, automatic_forecasts, device_location
from climate_app.forecast.domain import Forecast, ProviderError, utcnow
from climate_app.forecast.store import Store
from climate_app.ui import browser_location


class AutomaticLocationTests(unittest.TestCase):
    def payload(self, **changes):
        return dict(latitude=37.744126, longitude=-.850456, accuracy=25,
                    timestamp=utcnow().timestamp() * 1000, timezone="Europe/Madrid", **changes)

    def test_invalid_stale_and_inaccurate_positions_do_not_become_a_forecast(self):
        now = utcnow()
        for changes in ({"latitude": float("nan")}, {"longitude": 181}, {"accuracy": 20_000},
                        {"accuracy": None}, {"timestamp": (now - timedelta(minutes=11)).timestamp() * 1000},
                        {"timestamp": (now + timedelta(minutes=2)).timestamp() * 1000}):
            payload = self.payload()
            payload.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                device_location(payload, now)

    def test_only_approximate_coordinates_are_used_and_timezone_is_validated(self):
        payload = self.payload()
        location = device_location(payload)
        self.assertEqual((location.latitude, location.longitude), (37.74, -.85))
        payload.update(latitude=37.744127, timezone="not-a-zone")
        self.assertEqual(device_location(payload).timezone, "UTC")

    def test_queries_use_shared_quota_but_gps_and_forecasts_are_not_archived(self):
        now = utcnow()
        location = device_location(self.payload())
        def response(url, **kwargs):
            if "met.no" in url:
                data = {"properties": {"timeseries": [{"time": (now + timedelta(hours=1)).isoformat(),
                        "data": {"instant": {"details": {"air_temperature": 20, "relative_humidity": 60,
                                                       "wind_speed": 3, "cloud_area_fraction": 20}}}}]}}
            else:
                self.assertEqual(kwargs["params"]["latitude"], 37.74)
                self.assertEqual(kwargs["params"]["longitude"], -.85)
                data = {"hourly": {"time": [int((now + timedelta(hours=1)).timestamp())],
                                   "temperature_2m": [20], "relative_humidity_2m": [60],
                                   "wind_speed_10m": [10], "wind_direction_10m": [90]}}
            return SimpleNamespace(status_code=200, json=lambda: data, headers={})
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "private-test.sqlite3")
            store.seed()
            session = {}
            with patch("climate_app.forecast.providers.requests.get", side_effect=response) as request:
                frame, errors = automatic_forecasts(location, store, session, now)
                again, _ = automatic_forecasts(location, store, session, now + timedelta(minutes=1))
            self.assertEqual(request.call_count, 4)
            self.assertFalse(errors)
            self.assertEqual(set(frame.provider), {"ecmwf", "icon", "gfs", "metno", "climateapp"})
            self.assertTrue(frame.equals(again))
            self.assertEqual(len(store.locations()), 2)
            with store.connect() as connection:
                for table in ("runs", "points"):
                    self.assertEqual(connection.execute("SELECT count(*) FROM " + table).fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT count(*) FROM state WHERE key!='locations_seeded'").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT used FROM budget WHERE service='open-meteo'").fetchone()[0], 96)

    def test_partial_provider_failure_preserves_available_sources_and_finite_consensus(self):
        now = utcnow()
        location = device_location(self.payload())
        def fetch(provider, *_):
            if provider == "icon":
                raise ProviderError("Proveedor temporalmente no disponible")
            return Forecast(provider, now, [dict(valid_time=now + timedelta(hours=1), temperature=20)])
        with patch("climate_app.forecast.automatic.fetch_open_meteo", side_effect=fetch), \
             patch("climate_app.forecast.automatic.fetch_metno", side_effect=ProviderError("Sin datos")):
            frame, errors = automatic_forecasts(location, SimpleNamespace(), {}, now)
        self.assertEqual(set(errors), {"icon", "metno"})
        self.assertEqual(set(frame.provider), {"ecmwf", "gfs", "climateapp"})
        self.assertEqual(frame[frame.provider == "climateapp"].temperature.iloc[0], 20)

    def test_all_provider_failure_is_cached_briefly_and_never_becomes_dry_weather(self):
        now = utcnow()
        location = device_location(self.payload())
        session = {}
        with patch("climate_app.forecast.automatic.fetch_open_meteo", side_effect=ProviderError("No disponible")) as fetch, \
             patch("climate_app.forecast.automatic.fetch_metno", side_effect=ProviderError("No disponible")):
            frame, errors = automatic_forecasts(location, SimpleNamespace(), session, now)
            automatic_forecasts(location, SimpleNamespace(), session, now + timedelta(seconds=10))
        self.assertTrue(frame.empty)
        self.assertEqual(len(errors), 4)
        self.assertEqual(fetch.call_count, 3)

    def test_session_http_cache_is_bounded_and_never_persists_payloads(self):
        shared = SimpleNamespace(reserve=lambda *args: True)
        cache = SessionCache(shared, {})
        for index in range(100):
            cache.set_state(str(index), {"data": index})
        self.assertEqual(len(cache.cache), 16)
        self.assertTrue(cache.reserve("open-meteo", "fixture", 32, 8000))

    def test_model_preference_has_a_visible_source_fallback_and_no_private_leak(self):
        choose = browser_location.choose_provider
        self.assertEqual(choose(["ecmwf", "climateapp"], "weathernext3"), "climateapp")
        self.assertEqual(choose(["climateapp_private", "climateapp"]), "climateapp_private")
        self.assertEqual(choose(["ecmwf", "icon"], "icon"), "icon")
        self.assertIsNone(choose([]))

    def test_late_browser_reply_cannot_override_a_manual_location(self):
        event = {"kind": "position", "nonce": "fixture-event", **self.payload()}
        state = {"selected_forecast_location": "saved-city"}
        with patch.object(browser_location, "_bridge", return_value=event), \
             patch.object(browser_location.st, "session_state", state):
            browser_location.sync_browser_location()
        self.assertNotIn("device_location", state)

    def test_permission_denial_clears_previous_location_and_has_a_recovery_message(self):
        state = {"selected_forecast_location": AUTOMATIC, "device_location": device_location(self.payload())}
        with patch.object(browser_location, "_bridge", return_value={"kind": "error", "nonce": "denied-1", "reason": "denied"}), \
             patch.object(browser_location.st, "session_state", state):
            browser_location.sync_browser_location()
        self.assertNotIn("device_location", state)
        self.assertIn("permisos", state["device_location_error"])


if __name__ == "__main__":
    unittest.main()
