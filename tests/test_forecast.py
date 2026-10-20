import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from climate_app.forecast.analysis import climatology, combine, daily_summary, paired_samples, scores
from climate_app.forecast.domain import Forecast, Location, ProviderError, VARIABLES
from climate_app.forecast.providers import parse_metno, parse_open_meteo
from climate_app.forecast.store import Store
from climate_app.forecast.weathernext import parse_rows, query_sql, require_unbilled

NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def frame(rows):
    result = pd.DataFrame(rows)
    for variable in (*VARIABLES, "temperature_p10", "temperature_p90"):
        if variable not in result:
            result[variable] = np.nan
    for column in ("valid_time", "fetched_at", "init_time"):
        if column not in result:
            result[column] = pd.NaT
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def forecast_row(provider, when, temp, fetched=NOW, **extra):
    return dict(location_id="test", provider=provider, valid_time=when, temperature=temp,
                fetched_at=fetched, **extra)


class StorageTests(unittest.TestCase):
    def test_alias_preserves_identity_archive_and_survives_backup_and_reactivation(self):
        self.store.save_forecast(self.location, Forecast("gfs", NOW, [dict(valid_time=NOW + timedelta(hours=2), temperature=17)]))
        self.store.save_observations(self.location, "08227", [dict(valid_time=NOW - timedelta(hours=1), temperature=16)])
        self.store.set_alias(self.location, "  Casa  ")
        restored = Store(self.store.backup())
        location = restored.locations()[0]
        self.assertEqual((location.id, location.name, location.alias, location.display_name), (self.location, "Madrid", "Casa", "Casa · Madrid"))
        self.assertEqual(restored.forecasts(self.location, since=NOW).temperature.tolist(), [17])
        self.assertEqual(restored.observations(self.location, since=NOW - timedelta(days=1)).temperature.tolist(), [16])
        restored.deactivate(self.location)
        restored.add_location("Madrid", 40.4, -3.7)
        self.assertEqual(restored.locations()[0].alias, "Casa")
        restored.set_alias(self.location, "")
        self.assertEqual(restored.locations()[0].display_name, "Madrid")
        with restored.connect() as conn:
            self.assertEqual(len(conn.execute("SELECT * FROM locations").fetchone()), 9)

    def test_alias_rejects_invalid_input_and_unknown_place(self):
        for alias in ("x" * 61, "Casa\nTrabajo"):
            with self.assertRaises(ValueError):
                self.store.set_alias(self.location, alias)
        with self.assertRaises(ValueError):
            self.store.set_alias("missing", "Casa")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "forecast.sqlite3")
        self.location = self.store.add_location("Madrid", 40.4, -3.7)

    def test_snapshots_are_immutable_and_past_rows_are_rejected(self):
        data = Forecast("ecmwf", NOW, [dict(valid_time=NOW + timedelta(hours=1), temperature=20),
                                       dict(valid_time=NOW - timedelta(hours=1), temperature=99)])
        self.store.save_forecast(self.location, data)
        data.points[0]["temperature"] = 80
        self.store.save_forecast(self.location, data)
        saved = self.store.forecasts(self.location, since=NOW - timedelta(days=1), latest=False)
        self.assertEqual(saved.temperature.tolist(), [20])

    def test_latest_does_not_splice_an_old_long_range_tail(self):
        self.store.save_forecast(self.location, Forecast("ecmwf", NOW, [dict(valid_time=NOW + timedelta(days=5), temperature=20)]))
        self.store.save_forecast(self.location, Forecast("ecmwf", NOW + timedelta(hours=1), [dict(valid_time=NOW + timedelta(hours=2), temperature=21)]))
        latest = self.store.forecasts(self.location, since=NOW)
        self.assertEqual(latest.temperature.tolist(), [21])

    def test_budget_and_location_limits_are_persistent(self):
        self.assertTrue(self.store.reserve("api", "today", 8, 10))
        self.assertFalse(Store(self.store.path).reserve("api", "today", 3, 10))
        for n in range(9):
            self.store.add_location(str(n), n, n)
        with self.assertRaises(ValueError):
            self.store.add_location("eleventh", -50, -50)

    def test_snapshot_restores_all_forecasts_and_observations(self):
        self.store.save_forecast(self.location, Forecast("gfs", NOW, [dict(valid_time=NOW + timedelta(hours=2), temperature=17)]))
        self.store.save_observations(self.location, "08227", [dict(valid_time=NOW - timedelta(hours=1), temperature=16)])
        backup = self.store.backup()
        restored = Store(backup)
        with restored.connect() as conn:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(restored.forecasts(self.location, since=NOW).temperature.tolist(), [17])
        self.assertEqual(restored.observations(self.location, since=NOW - timedelta(days=1)).temperature.tolist(), [16])

    def test_deactivation_preserves_history_and_is_reversible(self):
        self.store.deactivate(self.location)
        self.assertEqual(self.store.locations(), [])
        self.assertEqual(len(self.store.locations(include_inactive=True)), 1)
        self.assertEqual(self.store.add_location("Madrid", 40.4, -3.7), self.location)

    def test_native_run_and_grid_provenance_survives_snapshot_restore(self):
        point = dict(valid_time=NOW + timedelta(hours=1), temperature=20,
                     source_init_time=NOW - timedelta(hours=1),
                     temperature_init_time=NOW - timedelta(hours=2),
                     surface_latitude=40.45, surface_longitude=-3.45,
                     temperature_latitude=40.475, temperature_longitude=-3.475)
        self.store.save_forecast(self.location, Forecast("weathernext3", NOW, [point]))
        restored = Store(self.store.backup())
        archived = restored.forecasts(self.location, since=NOW, latest=False)
        row = archived.iloc[0]
        self.assertEqual(row.source_init_time, NOW - timedelta(hours=1))
        self.assertEqual(row.temperature_init_time, NOW - timedelta(hours=2))
        self.assertEqual(row.surface_latitude, 40.45)
        self.assertEqual(row.temperature_longitude, -3.475)
        self.assertNotIn("metadata", archived.columns)

    def test_old_release_can_still_insert_twelve_column_points(self):
        self.store.save_forecast(self.location, Forecast("gfs", NOW,
            [dict(valid_time=NOW + timedelta(hours=1), temperature=20)]))
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(len(connection.execute("PRAGMA table_info(points)").fetchall()), 12)
            run_id = connection.execute("SELECT id FROM runs LIMIT 1").fetchone()[0]
            connection.execute("INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, (NOW + timedelta(hours=2)).isoformat(), 21, *[None] * 9))
        row = self.store.forecasts(self.location, since=NOW).iloc[-1]
        self.assertEqual(row.temperature, 21)
        self.assertTrue(pd.isna(row.source_init_time))


class ProviderTests(unittest.TestCase):
    def test_metno_rain_is_aligned_with_end_of_interval(self):
        result = parse_metno({"properties": {"timeseries": [{"time": NOW.isoformat(), "data": {
            "instant": {"details": {"air_temperature": 20, "wind_speed": 5}},
            "next_1_hours": {"details": {"precipitation_amount": 2}},
            "next_6_hours": {"details": {"precipitation_amount": 99}},
        }}]}}, NOW)
        self.assertEqual(result.points[0]["wind_speed"], 18)
        self.assertNotIn("precipitation", result.points[0])
        self.assertEqual(result.points[1]["valid_time"], NOW + timedelta(hours=1))
        self.assertEqual(result.points[1]["precipitation"], 2)
        self.assertIsNone(result.points[1]["precip_probability"])

    def test_missing_model_values_are_not_zero(self):
        data = {"hourly": {"time": [(NOW + timedelta(hours=1)).timestamp()], "temperature_2m": [20]}}
        point = parse_open_meteo("gfs", data, NOW).points[0]
        self.assertIsNone(point["precipitation"])
        self.assertIsNone(point["humidity"])

    def test_weathernext_unit_conversions_and_native_uncertainty(self):
        result = parse_rows([dict(time=NOW + timedelta(hours=1), temp_k=293.15, temp_p10_k=292.15,
                                 temp_p90_k=294.15, dew_k=283.15, wind_mps=5, u=-5, v=0,
                                 precip_m=0.002, cloud_fraction=0.5)], NOW)
        point = result.points[0]
        self.assertAlmostEqual(point["temperature"], 20)
        self.assertAlmostEqual(point["temperature_p10"], 19)
        self.assertAlmostEqual(point["temperature_p90"], 21)
        self.assertAlmostEqual(point["wind_speed"], 18)
        self.assertAlmostEqual(point["wind_direction"], 90)
        self.assertAlmostEqual(point["precipitation"], 2)
        self.assertAlmostEqual(point["cloud_cover"], 50)
        self.assertNotIn("precip_probability", point)

    def test_weathernext_keeps_different_native_runs_per_valid_hour(self):
        surface = NOW - timedelta(hours=1)
        station = NOW - timedelta(hours=2)
        rows = [dict(time=NOW + timedelta(hours=1), init_time=surface,
                     temperature_init_time=station, temp_k=293.15,
                     surface_latitude=40.45, temperature_latitude=40.475),
                dict(time=NOW + timedelta(days=5), init_time=NOW - timedelta(hours=6),
                     temperature_init_time=NOW - timedelta(hours=6), temp_k=292.15)]
        result = parse_rows(rows, NOW)
        self.assertEqual(result.points[0]["source_init_time"], surface)
        self.assertEqual(result.points[0]["temperature_init_time"], station)
        self.assertEqual(result.points[0]["temperature_latitude"], 40.475)
        self.assertEqual(result.points[1]["source_init_time"], NOW - timedelta(hours=6))

    def test_zero_cost_gate_requires_explicit_disabled_billing(self):
        for payload, status in [({"billingEnabled": True}, 200), ({}, 200), ({"billingEnabled": False}, 403)]:
            session = Mock()
            session.get.return_value.status_code = status
            session.get.return_value.json.return_value = payload
            with self.assertRaises(ProviderError):
                require_unbilled(session, "climate-test")
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"billingEnabled": False}
        require_unbilled(session, "climate-test")

    def test_sql_identifiers_cannot_inject_sql(self):
        with self.assertRaises(ProviderError):
            query_sql("project.dataset`; DROP TABLE anything;")
        self.assertIn("station_head_temperature_2m_p50", query_sql("climate-test.weather"))


class VerificationTests(unittest.TestCase):
    def test_circular_wind_and_missing_values(self):
        forecasts = frame([forecast_row("gfs", NOW + timedelta(hours=1), 20, wind_direction=350),
                           forecast_row("icon", NOW + timedelta(hours=1), 22, wind_direction=10)])
        merged = combine(forecasts, as_of=NOW).iloc[0]
        self.assertAlmostEqual(merged.temperature, 21)
        self.assertLess(min(merged.wind_direction, 360 - merged.wind_direction), 0.001)
        self.assertTrue(pd.isna(merged.precipitation))
        self.assertTrue(pd.isna(merged.temperature_p10))
        self.assertFalse(merged.trained)

    def test_learned_weights_favor_better_model_only_with_sufficient_evidence(self):
        forecasts = frame([forecast_row("gfs", NOW + timedelta(hours=1), 20),
                           forecast_row("icon", NOW + timedelta(hours=1), 30)])
        evidence = pd.DataFrame([dict(provider=p, variable="temperature", horizon="0–6 h", mae=mae, n=60, days=4)
                                 for p, mae in [("gfs", 1), ("icon", 4)]])
        self.assertLess(combine(forecasts, evidence, NOW).temperature.iloc[0], 25)
        evidence.loc[evidence.provider == "icon", "n"] = 10
        self.assertEqual(combine(forecasts, evidence, NOW).temperature.iloc[0], 25)

    def test_hindcasts_and_observations_not_yet_available_are_excluded(self):
        target = NOW - timedelta(hours=1)
        forecasts = frame([forecast_row("gfs", target, 20, NOW - timedelta(hours=2)),
                           forecast_row("icon", target, 99, NOW)])
        observations = frame([dict(location_id="test", station_id="08227", valid_time=target,
                                    fetched_at=NOW + timedelta(hours=1), temperature=21)])
        self.assertTrue(scores(forecasts, observations, NOW).empty)
        observations["fetched_at"] = pd.Timestamp(NOW)
        result = scores(forecasts, observations, NOW)
        self.assertEqual(result.provider.tolist(), ["gfs"])
        self.assertEqual(result.mae.tolist(), [1])

    def test_repeated_snapshots_do_not_inflate_sample_count(self):
        target = NOW - timedelta(hours=1)
        forecasts = frame([forecast_row("gfs", target, 20, NOW - timedelta(hours=n)) for n in (2, 3, 4)])
        observations = frame([dict(location_id="test", station_id="08227", valid_time=target,
                                    fetched_at=NOW, temperature=21)])
        self.assertEqual(scores(forecasts, observations, NOW).n.tolist(), [1])

    def test_daily_rain_missing_is_not_reported_as_dry(self):
        forecasts = frame([forecast_row("gfs", NOW + timedelta(hours=n), 20) for n in (1, 2)])
        summary = daily_summary(forecasts, "Europe/Madrid")
        self.assertTrue(pd.isna(summary.precipitation.iloc[0]))
        self.assertEqual(summary.rain_hours.iloc[0], 0)

    def test_climatology_crosses_new_year_and_excludes_current_year(self):
        target = pd.Timestamp("2026-01-02T12:00:00Z")
        rows = [dict(valid_time=pd.Timestamp(f"{year}-12-{day:02d}T12:00:00Z"), temperature=10,
                     fetched_at=NOW) for year in (2023, 2024, 2025) for day in range(20, 32)]
        observations = frame(rows)
        result = climatology(observations, target, 15, "UTC")
        self.assertEqual(result["mean"], 10)
        self.assertEqual(result["delta"], 5)
        self.assertEqual(result["percentile"], 100)


if __name__ == "__main__":
    unittest.main()
