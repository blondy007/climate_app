import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from climate_app.auth import password_hash, verify_password
from climate_app.forecast.domain import Forecast, ProviderError, utcnow
from climate_app.forecast.store import Store
from climate_app.forecast.worker import collect_cycle
from climate_app.shared.utils import canonical_station_id
from climate_app.data.transformations import normalize_meteostat_df

import pandas as pd


class AuthTests(unittest.TestCase):
    def test_password_hash_is_salted_and_invalid_password_is_rejected(self):
        value = password_hash("test-password-at-least-twelve")
        self.assertTrue(verify_password("test-password-at-least-twelve", value))
        self.assertFalse(verify_password("wrong-password", value))
        self.assertNotEqual(value, password_hash("test-password-at-least-twelve"))
        self.assertFalse(verify_password("anything", "malformed"))

    def test_public_view_never_contains_google_or_private_consensus(self):
        from climate_app.forecast.access import visible_forecasts
        data = pd.DataFrame({"provider": ["ecmwf", "weathernext3", "climateapp", "climateapp_private"]})
        with patch("climate_app.forecast.access.private_view", return_value=False):
            self.assertEqual(visible_forecasts(data).provider.tolist(), ["ecmwf", "climateapp"])
        with patch("climate_app.forecast.access.private_view", return_value=True):
            self.assertEqual(len(visible_forecasts(data)), 4)


class HistoricalDataTests(unittest.TestCase):
    def test_model_substitutes_are_not_used_as_ground_truth(self):
        from climate_app.forecast.observations import parse_observation_csv
        now = utcnow()
        data = pd.DataFrame([dict(year=now.year, month=now.month, day=now.day, hour=0,
                                 temp=20, temp_source="metar", prcp=8, prcp_source="dwd_mosmix",
                                 wspd=10, wspd_source="metar")])
        result = parse_observation_csv(data, now - timedelta(days=1), now + timedelta(days=1))[0]
        self.assertEqual(result["temperature"], 20)
        self.assertEqual(result["wind_speed"], 10)
        self.assertTrue(pd.isna(result["precipitation"]))

    def test_meteostat_wind_is_already_kmh_and_lost_station_zero_is_repaired(self):
        data = pd.DataFrame({"station": [8227], "datetime": ["2025-01-01T12:00:00Z"],
                             "temp": [8], "wspd": [10], "rhum": [60]})
        normal = normalize_meteostat_df(data)
        self.assertEqual(normal.wspd_kmh.iloc[0], 10)
        self.assertEqual(normal.station.iloc[0], "08227")
        self.assertEqual(normal.Lugar.iloc[0], "Torrejón de Ardoz")
        self.assertEqual(canonical_station_id("8433"), "08433")


class WorkerTests(unittest.TestCase):
    def test_provider_failure_does_not_stop_collection_or_public_private_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "forecasts.sqlite3")
            location_id = store.add_location("Test", 40.4, -3.7, station_id="08227", station_name="Madrid", station_distance_km=1)

            def open_forecast(provider, *_):
                now = utcnow()
                return Forecast(provider, now, [dict(valid_time=now + timedelta(hours=2), temperature=20)])

            def google_forecast(*_):
                now = utcnow()
                return Forecast("weathernext3", now, [dict(valid_time=now + timedelta(hours=2), temperature=100)])

            with patch("climate_app.forecast.worker.fetch_open_meteo", side_effect=open_forecast), \
                 patch("climate_app.forecast.worker.fetch_metno", side_effect=ProviderError("HTTP 503")), \
                 patch("climate_app.forecast.worker.configured", return_value=True), \
                 patch("climate_app.forecast.worker.fetch_weathernext", side_effect=google_forecast), \
                 patch("climate_app.forecast.worker.fetch_observations", return_value=[]):
                collect_cycle(store, include_climate=False)
            frame = store.forecasts(location_id)
            self.assertEqual(set(frame.provider), {"ecmwf", "icon", "gfs", "weathernext3", "climateapp", "climateapp_private"})
            self.assertEqual(set(frame[frame.provider == "climateapp"].temperature), {20})
            self.assertTrue((frame[frame.provider == "climateapp_private"].temperature >= 20).all())
            self.assertEqual(store.state(f"task:{location_id}:metno")["status"], "error")
            self.assertTrue((Path(temp) / "forecasts-backup.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
