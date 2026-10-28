import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from climate_app.forecast.domain import Location, ProviderError
from climate_app.forecast.store import Store
from climate_app.forecast.weathernext import configured, fetch_weathernext, parse_rows
from climate_app.forecast.weathernext_ee import SURFACE, STATION, decode_region, merge_grids, require_noncommercial, fetch_point

NOW = datetime(2026, 10, 28, 12, 15, tzinfo=timezone.utc)


def table(bands, init, valid, values, lat=37.8, lon=-.8):
    return [["id", "longitude", "latitude", "time", *bands, "source_init_ms"],
            ["native", lon, lat, valid.timestamp() * 1000,
             *[values.get(name) for name in bands.values()], init.timestamp() * 1000]]


class EarthEngineTests(unittest.TestCase):
    def test_earth_engine_needs_no_bigquery_dataset(self):
        with patch.dict(os.environ, {"WEATHERNEXT_PROJECT": "climate-test", "WEATHERNEXT_BACKEND": "earthengine", "WEATHERNEXT_DATASET": ""}):
            self.assertTrue(configured())

    def test_commercial_unknown_and_inaccessible_registrations_are_rejected(self):
        session = Mock()
        for status, payload in [(403, {}), (200, {}), (200, {"registrationState": "REGISTERED_COMMERCIALLY"})]:
            session.get.return_value.status_code = status
            session.get.return_value.json.return_value = payload
            with self.assertRaises(ProviderError):
                require_noncommercial(session, "climate-test")
        session.get.return_value.json.return_value = {"registrationState": "REGISTERED_NOT_COMMERCIALLY"}
        require_noncommercial(session, "climate-test")

    def test_billing_gate_precedes_earth_engine_and_never_falls_back_to_bigquery(self):
        for billing in (True, False):
            with patch.dict(os.environ, {"WEATHERNEXT_PROJECT": "climate-test", "WEATHERNEXT_BACKEND": "earthengine"}), \
                 patch("google.auth.default", return_value=(Mock(), None)), \
                 patch("google.auth.transport.requests.AuthorizedSession") as session, \
                 patch("climate_app.forecast.weathernext_ee.fetch_point") as fetch, \
                 patch("google.cloud.bigquery.Client") as bigquery:
                response = session.return_value.__enter__.return_value.get.return_value
                response.status_code = 200
                response.json.return_value = {"billingEnabled": billing, "registrationState": "REGISTERED_NOT_COMMERCIALLY"}
                if billing:
                    with self.assertRaises(ProviderError):
                        fetch_weathernext(Location("test", "Test", 37.8, -.8), Mock())
                    fetch.assert_not_called()
                else:
                    fetch_weathernext(Location("test", "Test", 37.8, -.8), Mock())
                    fetch.assert_called_once()
                bigquery.assert_not_called()

    def test_native_grids_times_and_units_survive_independent_station_run(self):
        valid = NOW + timedelta(hours=1)
        surface_init, station_init = NOW - timedelta(hours=1), NOW - timedelta(hours=2)
        surface = decode_region(table(SURFACE, surface_init, valid,
            {"temp_k": 290, "temp_p10_k": 289, "temp_p90_k": 291, "dew_k": 280,
             "wind_mps": 5, "u": -5, "v": 0, "precip_m": .002, "cloud_fraction": .5}), SURFACE, NOW)
        station = decode_region(table(STATION, station_init, valid,
            {"temp_k": 293.15, "temp_p10_k": 292.15, "temp_p90_k": 294.15, "dew_k": 283.15},
            lat=37.75, lon=-.85), STATION, NOW)
        point = parse_rows(merge_grids(surface, station), NOW).points[0]
        self.assertEqual(point["valid_time"], valid)
        self.assertEqual(point["source_init_time"], surface_init)
        self.assertEqual(point["temperature_init_time"], station_init)
        self.assertEqual((point["surface_latitude"], point["temperature_latitude"]), (37.8, 37.75))
        self.assertAlmostEqual(point["temperature"], 20)
        self.assertAlmostEqual(point["temperature_p10"], 19)
        self.assertAlmostEqual(point["temperature_p90"], 21)
        self.assertEqual((point["precipitation"], point["wind_speed"], point["wind_direction"], point["cloud_cover"]), (2, 18, 90, 50))
        self.assertIsNotNone(point["humidity"])
        station[0]["temp_k"] = None
        fallback = merge_grids(surface, station)[0]
        self.assertEqual(fallback["temp_k"], 290)
        self.assertEqual(fallback["temperature_init_time"], surface_init)
        self.assertEqual(fallback["temperature_latitude"], 37.8)

    def test_stale_future_initializations_and_out_of_horizon_data_are_rejected(self):
        for init, valid in [(NOW - timedelta(hours=25), NOW + timedelta(hours=1)),
                            (NOW + timedelta(hours=1), NOW + timedelta(hours=2)),
                            (NOW, NOW), (NOW, NOW + timedelta(hours=361))]:
            with self.assertRaises(ProviderError):
                decode_region(table(SURFACE, init, valid, {"temp_k": 290}), SURFACE, NOW)

    def test_daily_limit_is_persistent_and_stops_before_earth_engine_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "isolated.sqlite3")
            self.assertTrue(store.reserve("earthengine-forecasts", NOW.strftime("%Y-%m-%d"), 240, 240))
            with patch("ee.Initialize") as initialize:
                with self.assertRaises(ProviderError):
                    fetch_point(Location("test", "Test", 37.8, -.8), Store(store.path), Mock(), "climate-test", NOW)
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
