import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from climate_app.forecast.analysis import climatology, daily_summary, scores
from climate_app.forecast.domain import Location
from climate_app.forecast.thermal import with_feels_like
from climate_app.ui.appearance import PALETTES
from climate_app.ui.forecast_dashboard import temperature_figure


class ThermalTests(unittest.TestCase):
    def test_wind_units_and_true_zero_inputs(self):
        points = pd.DataFrame({"temperature": [20, 10], "humidity": [0, 0], "wind_speed": [0, 36]}, index=[5, 9])
        result = with_feels_like(points)
        self.assertEqual(result.feels_like.tolist(), [16, -1])
        self.assertNotIn("feels_like", points)
        self.assertEqual(result.index.tolist(), [5, 9])

    def test_hot_humid_and_cold_windy_reference_values(self):
        result = with_feels_like(pd.DataFrame({"temperature": [30, 0], "humidity": [70, 80], "wind_speed": [18, 36]}))
        self.assertAlmostEqual(result.feels_like.iloc[0], 32.268334, places=5)
        self.assertAlmostEqual(result.feels_like.iloc[1], -9.38828, places=5)

    def test_missing_and_invalid_inputs_never_become_air_temperature(self):
        rows = [dict(temperature=20, humidity=50, wind_speed=10) for _ in range(10)]
        for row, (field, value) in zip(rows, [("temperature", None), ("humidity", None), ("wind_speed", None),
                                             ("temperature", np.inf), ("humidity", -1), ("humidity", 101),
                                             ("wind_speed", -1), ("wind_speed", np.inf), ("temperature", -237.7),
                                             ("humidity", "invalid")]):
            row[field] = value
        self.assertTrue(with_feels_like(pd.DataFrame(rows)).feels_like.isna().all())
        self.assertTrue(with_feels_like(pd.DataFrame({"temperature": [20]})).feels_like.isna().all())
        self.assertTrue(with_feels_like(pd.DataFrame()).empty)

    def test_daily_extremes_use_same_hour_inputs_and_count_partial_coverage(self):
        data = pd.DataFrame({"valid_time": pd.date_range("2026-10-24T22:00Z", periods=3, freq="h"),
                             "temperature": [20, 30, 40], "humidity": [0, 0, np.nan], "wind_speed": [0, 36, 0],
                             "wind_gust": np.nan, "precipitation": np.nan, "precip_probability": np.nan})
        day = daily_summary(data, "Europe/Madrid").iloc[0]
        self.assertEqual((day.minimum, day.maximum), (20, 40))
        self.assertEqual((day.feels_minimum, day.feels_maximum, day.feels_hours), (16, 19, 2))

    def test_climatology_uses_same_index_for_observations_and_excludes_incomplete_hours(self):
        rows = [dict(valid_time=pd.Timestamp(f"{year}-10-{day:02d}T12:00Z"), temperature=20,
                     humidity=0, wind_speed=0) for year in (2023, 2024, 2025) for day in range(10, 21)]
        rows[0]["humidity"] = np.nan
        result = climatology(pd.DataFrame(rows), pd.Timestamp("2026-10-20T12:00Z"), 20, "UTC", variable="feels_like")
        self.assertEqual((result["mean"], result["delta"], result["n"]), (16, 4, 32))
        for row in rows:
            row["wind_speed"] = np.nan
        self.assertIsNone(climatology(pd.DataFrame(rows), pd.Timestamp("2026-10-20T12:00Z"), 20, "UTC", variable="feels_like"))

    def test_evaluation_derives_both_sides_and_requires_complete_observed_inputs(self):
        now = pd.Timestamp("2026-10-20T12:00Z")
        shared = dict(location_id="test", temperature=10, humidity=0, wind_direction=0,
                      precipitation=0, temperature_p10=np.nan, temperature_p90=np.nan)
        forecast, observed = [], []
        for hours in (1, 2):
            target = now - pd.Timedelta(hours=hours)
            forecast.append(dict(shared, provider="gfs", valid_time=target, wind_speed=36, fetched_at=target-pd.Timedelta(hours=2)))
            observed.append(dict(shared, valid_time=target, wind_speed=0, fetched_at=now))
        observed[1]["humidity"] = np.nan
        result = scores(pd.DataFrame(forecast), pd.DataFrame(observed), as_of=now).set_index("variable")
        self.assertEqual((result.loc["feels_like", "mae"], result.loc["feels_like", "n"]), (7, 1))
        self.assertEqual(result.loc["temperature", "n"], 2)

    def test_feels_like_chart_never_reuses_native_temperature_interval(self):
        frame = with_feels_like(pd.DataFrame(dict(provider=["gfs"], valid_time=[pd.Timestamp("2026-10-20T12:00Z")],
                   temperature=[20], humidity=[50], wind_speed=[10], temperature_p10=[18], temperature_p90=[22])))
        with patch("climate_app.ui.forecast_dashboard.palette", return_value=PALETTES["light"]):
            fig = temperature_figure(frame, frame, "gfs", Location("x", "Madrid", 40, -3),
                                     variable="feels_like", include_feels_like=False)
        self.assertEqual(len(fig.data), 1)
        self.assertAlmostEqual(fig.data[0].y[0], frame.feels_like.iloc[0])
        self.assertFalse(fig.data[0].connectgaps)


if __name__ == "__main__":
    unittest.main()
