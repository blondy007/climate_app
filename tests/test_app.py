import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from climate_app.forecast.automatic import AUTOMATIC
from climate_app.forecast.domain import Forecast, Location, utcnow
from climate_app.forecast.store import Store


class AppTests(unittest.TestCase):
    def test_hero_refresh_relocates_and_recovery_survives_without_a_forecast(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            saved = store.locations()[0]
            now = utcnow()
            store.save_forecast(saved.id, Forecast("ecmwf", now, [
                dict(valid_time=now + timedelta(hours=h), temperature=18, humidity=60, wind_speed=15)
                for h in (1, 2)]))
            frame = store.forecasts(saved.id, since=now)
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}), \
                 patch("climate_app.ui.forecast_dashboard.automatic_forecasts", return_value=(frame, {})):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
                app.session_state["selected_forecast_location"] = AUTOMATIC
                app.session_state["device_location"] = Location("auto:test", "Ubicación actual", 40, -3)
                app.run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(any('hero-can-refresh' in m.value and 'Solo durante esta visita' in m.value
                                    for m in app.markdown if '<section class="weather-hero' in m.value))
                app.button(key="refresh_device_location").click().run(timeout=30)
                self.assertEqual(app.session_state["device_location_request"], 1)
                app.radio(key="forecast_location").set_value(saved.id).run(timeout=30)
                self.assertFalse(any(b.key == "refresh_device_location" for b in app.button))
                del app.session_state["device_location"]
                app.session_state["device_location_error"] = "Permiso denegado"
                app.radio(key="forecast_location").set_value(AUTOMATIC).run(timeout=30)
                self.assertEqual(app.button(key="refresh_device_location").label, "Volver a localizar")
                app.button(key="refresh_device_location").click().run(timeout=30)
                self.assertEqual(app.session_state["device_location_request"], 2)
                self.assertNotIn("device_location_error", app.session_state)
                self.assertFalse(app.exception)

    def test_duplicate_places_alias_and_undo_preserve_the_selected_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            first, second = store.locations()
            third_id = store.add_location(second.name, second.latitude + .01, second.longitude)
            now = utcnow()
            for location in store.locations():
                store.save_forecast(location.id, Forecast("ecmwf", now, [dict(valid_time=now + timedelta(hours=2), temperature=21)]))
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                options = app.radio(key="forecast_location").options
                self.assertEqual(len(options), len(set(options)))
                self.assertTrue(any("40.4594" in label for label in options))
                app.radio(key="forecast_location").set_value(second.id).run(timeout=30)
                next(button for button in app.button if button.label == "Mis lugares").click().run(timeout=30)
                self.assertFalse(any(element.key.startswith("alias_input_") for element in app.text_input if element.key))
                app.button(key=f"edit_alias_{second.id}").click().run(timeout=30)
                app.text_input(key=f"alias_input_{second.id}").set_value("Trabajo")
                next(button for button in app.button if button.label == "Guardar alias").click().run(timeout=30)
                self.assertEqual(Store(path).locations()[1].alias, "Trabajo")
                app.button(key=f"remove_place_{second.id}").click().run(timeout=30)
                self.assertEqual(len(store.locations()), 3)
                app.button(key=f"cancel_remove_place_{second.id}").click().run(timeout=30)
                self.assertEqual(len(store.locations()), 3)
                app.button(key=f"remove_place_{second.id}").click().run(timeout=30)
                app.button(key=f"confirm_remove_place_{second.id}").click().run(timeout=30)
                self.assertTrue(any(button.label == "Deshacer" for button in app.button))
                app.button(key="undo_remove_place").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(store.locations()), 3)
                self.assertEqual(store.locations()[1].alias, "Trabajo")
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(app.radio(key="forecast_location").value, second.id)
                self.assertIn("Trabajo · Torrejón de Ardoz", app.radio(key="forecast_location").options)
                self.assertTrue(any('Trabajo · Torrejón de Ardoz' in element.value and 'hero-top' in element.value for element in app.markdown))
                self.assertEqual(len(store.forecasts(second.id)), 1)

    def test_inline_alias_editor_cancels_switches_places_and_restores_original_name(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            first, second = store.locations()
            store.set_alias(first.id, "Casa")
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.button(key=f"edit_alias_{first.id}").click().run(timeout=30)
                app.text_input(key=f"alias_input_{first.id}").set_value("Sin guardar")
                next(b for b in app.button if b.label == "Cancelar alias").click().run(timeout=30)
                self.assertEqual(store.locations()[0].alias, "Casa")
                app.button(key=f"edit_alias_{first.id}").click().run(timeout=30)
                self.assertEqual(app.text_input(key=f"alias_input_{first.id}").value, "Casa")
                app.button(key=f"edit_alias_{second.id}").click().run(timeout=30)
                self.assertEqual([e.key for e in app.text_input if e.key and e.key.startswith("alias_input_")],
                                 [f"alias_input_{second.id}"])
                app.button(key=f"edit_alias_{first.id}").click().run(timeout=30)
                app.text_input(key=f"alias_input_{first.id}").set_value("")
                next(b for b in app.button if b.label == "Guardar alias").click().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertFalse(store.locations()[0].alias)
                self.assertEqual(store.locations()[0].display_name, first.name)
                self.assertFalse(any(e.key and e.key.startswith("alias_input_") for e in app.text_input))

    def test_day_selection_changes_hours_and_resets_when_context_no_longer_matches(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            now = utcnow().replace(minute=0, second=0, microsecond=0)
            for location in store.locations():
                for provider in ("ecmwf", "icon"):
                    store.save_forecast(location.id, Forecast(provider, now, [dict(valid_time=now + timedelta(hours=h), temperature=h, precipitation=0) for h in range(1, 361)]))
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                next(radio for radio in app.radio if radio.label == "Horizonte").set_value(360).run(timeout=30)
                chosen = (pd.Timestamp(now).tz_convert("Europe/Madrid") + pd.Timedelta(days=5)).date().isoformat()
                app.button(key=f"forecast_day_{chosen}").click().run(timeout=30)
                self.assertEqual(app.session_state["forecast_selected_day"], chosen)
                hours = next(element.value for element in app.markdown if 'id="hourly-section"' in element.value)
                start = pd.Timestamp(chosen, tz="Europe/Madrid")
                expected_hours = len(pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left"))
                self.assertEqual(hours.count('class="hour-cell"'), expected_hours)
                self.assertNotIn("Próximas horas", hours)
                self.assertTrue(any("PRÓXIMA HORA" in element.value for element in app.markdown))
                app.button(key="reset_forecast_day").click().run(timeout=30)
                self.assertIsNone(app.session_state["forecast_selected_day"])
                app.button(key=f"forecast_day_{chosen}").click().run(timeout=30)
                next(radio for radio in app.radio if radio.label == "Horizonte").set_value(48).run(timeout=30)
                self.assertIsNone(app.session_state["forecast_selected_day"])
                first_day = pd.Timestamp(now + timedelta(hours=1)).tz_convert("Europe/Madrid").date().isoformat()
                app.button(key=f"forecast_day_{first_day}").click().run(timeout=30)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.radio(key="forecast_model_preference").set_value("icon").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertIsNone(app.session_state["forecast_selected_day"])
                app.button(key=f"forecast_day_{first_day}").click().run(timeout=30)
                app.radio(key="forecast_location").set_value(store.locations()[1].id).run(timeout=30)
                self.assertIsNone(app.session_state["forecast_selected_day"])
                self.assertEqual(len(app.exception), 0, str(app.exception))

    def test_places_can_be_removed_and_restored_without_losing_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            now = utcnow()
            first, second = store.locations()
            for location in (first, second):
                store.save_forecast(location.id, Forecast("ecmwf", now, [dict(valid_time=now + timedelta(hours=2), temperature=21)]))
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                app.radio(key="forecast_location").set_value(second.id).run(timeout=30)
                next(button for button in app.button if button.label == "Mis lugares").click().run(timeout=30)
                self.assertEqual(app.radio(key="main_view").value, "⚙️ Ajustes")
                app.button(key=f"remove_place_{first.id}").click().run(timeout=30)
                app.button(key=f"confirm_remove_place_{first.id}").click().run(timeout=30)
                self.assertEqual([loc.id for loc in store.locations()], [second.id])
                self.assertEqual(len(store.forecasts(first.id)), 1)
                self.assertEqual(app.session_state["selected_forecast_location"], second.id)
                app.button(key=f"remove_place_{second.id}").click().run(timeout=30)
                app.button(key=f"confirm_remove_place_{second.id}").click().run(timeout=30)
                self.assertEqual(store.locations(), [])
                self.assertEqual(len(store.locations(include_inactive=True)), 2)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any("Tu próximo lugar favorito" in element.value for element in app.markdown))
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.button(key=f"restore_place_{second.id}").click().run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(app.radio(key="forecast_location").value, second.id)
                self.assertTrue(any('21°' in element.value for element in app.markdown))

    def test_dark_theme_survives_navigation_and_updates_charts(self):
        import json
        from climate_app.ui.appearance import PALETTES
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.sqlite3"
            store = Store(path)
            store.seed()
            now = utcnow()
            store.save_forecast(store.locations()[0].id, Forecast("ecmwf", now, [dict(valid_time=now + timedelta(hours=2), temperature=21)]))
            with patch.dict(os.environ, {"CLIMATE_FORECAST_DB": str(path)}):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.radio(key="appearance_picker").set_value("dark").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(app.session_state["appearance"], "dark")
                figure = json.loads(app.get("plotly_chart")[0].proto.spec)
                self.assertEqual(figure["layout"]["paper_bgcolor"], PALETTES["dark"]["surface"])
                self.assertEqual(figure["layout"]["font"]["color"], PALETTES["dark"]["ink"])
                app.radio(key="main_view").set_value("📊 Histórico").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                self.assertEqual(app.radio(key="appearance_picker").value, "dark")
                app.radio(key="appearance_picker").set_value("light").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                figure = json.loads(app.get("plotly_chart")[0].proto.spec)
                self.assertEqual(figure["layout"]["paper_bgcolor"], PALETTES["light"]["surface"])
                self.assertEqual(len(app.exception), 0, str(app.exception))

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
                self.assertFalse(any(select.label == "Previsión principal" for select in app.selectbox))
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                self.assertNotIn("WeatherNext 3", app.radio(key="forecast_model_preference").options)
                app.text_input(key="owner_username").set_value("test-owner")
                app.text_input(key="owner_password").set_value(test_password)
                next(button for button in app.button if button.label == "Entrar").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                model = app.radio(key="forecast_model_preference")
                self.assertIn("WeatherNext 3", model.options)
                model.set_value("weathernext3").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any('hero-value' in element.value and '30°' in element.value for element in app.markdown))
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                next(button for button in app.button if button.label == "Cerrar sesión").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertNotIn("WeatherNext 3", app.radio(key="forecast_model_preference").options)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertFalse(any("30°" in element.value and "hero-value" in element.value for element in app.markdown))

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
                self.assertTrue(any('hero-feels' in element.value and 'Sensación' in element.value for element in app.markdown))
                self.assertTrue(any('hour-feels' in element.value for element in app.markdown))
                self.assertTrue(any('day-feels' in element.value for element in app.markdown))
                self.assertTrue(any('Sensación mínima °C' in table.value.columns for table in app.dataframe))
                self.assertTrue(any('Sensación térmica' in table.value.columns for table in app.dataframe))
                next(radio for radio in app.radio if radio.label == "Horizonte").set_value(360).run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                next(box for box in app.checkbox if box.label == "Comparar modelos").set_value(False).run(timeout=30)
                app.radio(key="forecast_comparison_variable").set_value("feels_like").run(timeout=30)
                app.radio(key="forecast_climate_variable").set_value("feels_like").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertGreater(len(app.get("download_button")), 0)
                other_place = store.locations()[1].id
                app.radio(key="forecast_location").set_value(other_place).run(timeout=30)
                app.radio(key="main_view").set_value("⚙️ Ajustes").run(timeout=30)
                app.radio(key="main_view").set_value("🌤️ Previsión").run(timeout=30)
                self.assertEqual(app.radio(key="forecast_location").value, other_place)
                app.radio(key="main_view").set_value("🎯 Aciertos").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                app.radio(key="main_view").set_value("📊 Histórico").run(timeout=30)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any(button.label == "Eliminar estación" for button in app.button))
                self.assertTrue(any(select.label == "Ciudad(es)" for select in app.multiselect))


if __name__ == "__main__":
    unittest.main()
