import unittest
from datetime import datetime, timezone

import pandas as pd

from climate_app.forecast.domain import Location
from climate_app.ui.design import hero_html, weather_kind


class PresentationTests(unittest.TestCase):
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
