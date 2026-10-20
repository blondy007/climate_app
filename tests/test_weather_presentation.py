import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from climate_app.forecast.domain import Location
from climate_app.ui.design import hero_html, hourly_strip, weather_kind


class PresentationTests(unittest.TestCase):
    def test_upcoming_hours_are_a_time_window_when_a_model_has_gaps(self):
        now = pd.Timestamp.now(tz="UTC")
        frame = pd.DataFrame({"valid_time": [now + pd.Timedelta(hours=h) for h in (1, 13, 25, 37)],
                              "temperature": 12, "cloud_cover": 0, "precipitation": 0, "precip_probability": 0})
        with patch("climate_app.ui.design.st.markdown") as render:
            hourly_strip(frame, "Europe/Madrid")
        self.assertEqual(render.call_args.args[0].count('class="hour-cell"'), 2)

    def test_selected_day_keeps_all_25_hours_and_identifies_repeated_local_hours(self):
        times = pd.date_range("2026-10-25", "2026-10-26", tz="Europe/Madrid", freq="h", inclusive="left").tz_convert("UTC")
        frame = pd.DataFrame({"valid_time": times, "temperature": 12, "cloud_cover": 0, "precipitation": 0, "precip_probability": 0})
        with patch("climate_app.ui.design.st.markdown") as render:
            hourly_strip(frame, "Europe/Madrid", title="Domingo · 25/10", selected=True)
        html = render.call_args.args[0]
        self.assertEqual(html.count('class="hour-cell"'), 25)
        self.assertIn("02:00 CEST", html)
        self.assertIn("02:00 CET", html)

    def test_missing_conditions_do_not_invent_sun_or_rain(self):
        self.assertEqual(weather_kind({"cloud_cover": None, "precipitation": None})[0], "unknown")
        self.assertEqual(weather_kind({"cloud_cover": 0, "precipitation": 0, "precip_probability": 80})[0], "sun")

    def test_freezing_rain_is_not_presented_as_a_snow_forecast(self):
        self.assertEqual(weather_kind({"temperature": -1, "precipitation": .5})[0], "rain")

    def test_location_html_is_escaped_and_forecast_is_identified(self):
        when = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
        location = Location("test", '<img src=x onerror="alert(1)">', 40, -3)
        frame = pd.DataFrame([dict(valid_time=pd.Timestamp(when), temperature=20,
                                   temperature_p10=float("nan"), temperature_p90=float("nan"))])
        output = hero_html(location, frame.iloc[0], frame, "Consenso", when)
        self.assertIn("&lt;img", output)
        self.assertNotIn("<img", output)
        self.assertIn("PRÓXIMA HORA", output)
        self.assertNotIn("P10–P90", output)


if __name__ == "__main__":
    unittest.main()
