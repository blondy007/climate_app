import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from climate_app.forecast.domain import Forecast, utcnow
from climate_app.forecast.store import Store


class AppTests(unittest.TestCase):
    def test_google_requires_owner_login_and_logout_hides_it_again(self):
        from climate_app.auth import password_hash
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            now = utcnow()
            location = store.locations()[0]
            for provider, value in [("climateapp", 20), ("weathernext3", 30)]:
                store.save_forecast(location.id, Forecast(provider, now, [dict(valid_time=now + timedelta(hours=2), temperature=value)], metadata={"trained": False}))
            test_password = "local-test-fixture-not-a-real-password"
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path), "CLIMATE_OWNER_USERNAME": "test-owner",
                                        "CLIMATE_OWNER_PASSWORD_HASH": password_hash(test_password), "CLIMATE_HTTPS_VERIFIED": "1"}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                model = next(select for select in app.selectbox if select.label == "Previsión principal")
                self.assertNotIn("Google WeatherNext 3", model.options)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.text_input(key="owner_username").set_value("test-owner")
                app.text_input(key="owner_password").set_value(test_password)
                next(button for button in app.button if button.label == "Entrar").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                model = next(select for select in app.selectbox if select.label == "Previsión principal")
                self.assertIn("Google WeatherNext 3", model.options)
                model.set_value("weathernext3").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any('hero-value' in element.value and '30°' in element.value for element in app.markdown))
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                next(button for button in app.button if button.label == "Cerrar sesión").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                model = next(select for select in app.selectbox if select.label == "Previsión principal")
                self.assertNotIn("Google WeatherNext 3", model.options)

    def test_forecast_and_empty_evaluation_render_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            now = utcnow()
            for location in store.locations():
                for provider in ("ecmwf", "icon", "climateapp"):
                    points = [dict(valid_time=(now + timedelta(hours=hour)).replace(minute=0, second=0, microsecond=0),
                                   temperature=20 + hour / 100, precipitation=0, humidity=60,
                                   wind_speed=12, wind_direction=90) for hour in range(1, 50)]
                    store.save_forecast(location.id, Forecast(provider, now, points, metadata={"trained": False}))
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any('hero-value' in element.value and '20°' in element.value for element in app.markdown))
                next(radio for radio in app.radio if radio.label == "Horizonte").set_value(360).run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                next(box for box in app.checkbox if box.label == "Comparar modelos").set_value(False).run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertGreater(len(app.get("download_button")), 0)
                other_place = store.locations()[1].id
                app.selectbox(key="forecast_location").set_value(other_place).run(timeout=30)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(app.selectbox(key="forecast_location").value, other_place)
                app.radio(key="main_view").set_value("🎯 Aciertos").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                app.radio(key="main_view").set_value("📊 Histórico").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any(button.label == "Eliminar estación" for button in app.button))
                self.assertTrue(any(select.label == "Ciudad(es)" for select in app.multiselect))


if __name__ == "__main__":
    unittest.main()
