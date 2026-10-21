"""Native browser permission bridge. No coordinates in URLs or browser storage."""
from pathlib import Path
import hashlib

import streamlit as st
from streamlit.components.v1 import declare_component
from streamlit.components.v1 import html

from climate_app.forecast.automatic import AUTOMATIC, device_location
from climate_app.forecast.domain import stamp, utcnow

PREFERENCES = {"auto": "Consenso recomendado", "climateapp": "Consenso sin WeatherNext",
               "ecmwf": "ECMWF IFS", "icon": "DWD ICON", "gfs": "NOAA GFS",
               "metno": "MET Norway", "weathernext3": "WeatherNext 3"}
_component_path = Path(__file__).with_name("browser_component")
def _bridge(**kwargs):
    version = hashlib.sha256((_component_path / "index.html").read_bytes()).hexdigest()[:10]
    component = declare_component("climateapp_browser_" + version, path=str(_component_path))
    return component(**kwargs)


def sync_browser_location():
    from climate_app.auth import restore_owner_session
    restore_owner_session()
    state = st.session_state
    event = _bridge(automatic=state.get("selected_forecast_location") == AUTOMATIC,
                    weather_view=state.get("main_view", "🌤️ Previsión") in ("🌤️ Previsión", "🎯 Aciertos"),
                    model=state.get("preferred_forecast_provider", "auto"),
                    restore=not state.get("browser_forecast_preferences_loaded", False),
                    auth_action=state.get("owner_cookie_action"),
                    request=state.get("device_location_request", 0), default=None, key="browser_location_bridge")
    html((_component_path / "watchdog.html").read_text(encoding="utf-8"), height=0)
    events = event.get("events", []) if isinstance(event, dict) and "events" in event else [event]
    if not isinstance(events, list):
        return
    seen = state.setdefault("browser_processed_events", [])
    for item in events[-16:]:
        if not isinstance(item, dict) or not isinstance(item.get("nonce"), str) or item["nonce"] in seen:
            continue
        seen.append(item["nonce"])
        del seen[:-32]
        _process_event(item, state)


def _process_event(event, state):
    if not isinstance(event, dict):
        return
    if event.get("kind") in ("auth", "auth_lost", "auth_error", "auth_logout"):
        from climate_app.auth import clear_owner_session, fingerprint, owner_config
        from climate_app.browser_auth import BrowserSessions
        pending = state.get("owner_cookie_action", {})
        matching_action = pending.get("nonce") == event.get("request")
        if event["kind"] == "auth_logout" and matching_action:
            clear_owner_session()
            state.pop("owner_cookie_action", None)
            state["owner_logout_pending"] = False
        elif event["kind"] == "auth_error" and matching_action:
            state["owner_cookie_warning"] = "Tu sesión está abierta en esta pestaña, pero no hemos podido recordarla. Comprueba que el navegador permita las cookies de esta web."
            if pending.get("kind") == "remember":
                BrowserSessions().cancel_exchange(pending.get("code"))
            state.pop("owner_cookie_action", None)
        elif event["kind"] == "auth_lost" and not pending and state.get("owner_browser_session"):
            clear_owner_session()
        elif event["kind"] == "auth" and not state.get("owner_logout_pending"):
            config = owner_config()
            if config:
                proof = BrowserSessions().consume_proof(event.get("proof"), fingerprint(config))
                if proof:
                    state["owner_authenticated"] = fingerprint(config)
                    state["owner_login_time"] = stamp(utcnow())
                    state["owner_browser_session"] = proof["session_hash"]
                    state["owner_browser_expires"] = proof["expires_at"]
                    state.pop("owner_cookie_warning", None)
                    if matching_action:
                        state.pop("owner_cookie_action", None)
        return
    if event.get("kind") == "preferences" and not state.get("browser_forecast_preferences_loaded"):
        state["browser_forecast_preferences_loaded"] = True
        model = event.get("model")
        if isinstance(model, str) and model in PREFERENCES:
            state["preferred_forecast_provider"] = model
        if event.get("automatic") is True:
            state["selected_forecast_location"] = AUTOMATIC
    if state.get("selected_forecast_location") != AUTOMATIC:
        return  # A late GPS reply must never undo a manual place selection.
    if event.get("kind") == "position":
        try:
            location = device_location(event)
            state["device_location"] = location
            state["device_location_error"] = None
        except ValueError as exc:
            state.pop("device_location", None)
            state["device_location_error"] = str(exc)
    elif event.get("kind") == "error":
        state.pop("device_location", None)
        state["device_location_error"] = {
            "denied": "El navegador no permite acceder a tu ubicación. Puedes habilitarla en los permisos de esta web o elegir un lugar guardado.",
            "timeout": "El dispositivo ha tardado demasiado en localizarte. Comprueba que la ubicación esté activada y vuelve a intentarlo.",
            "unsupported": "Este navegador no ofrece ubicación. Elige uno de tus lugares guardados.",
            "https": "La ubicación necesita abrir la aplicación mediante HTTPS.",
        }.get(event.get("reason"), "No se ha podido obtener la ubicación. Comprueba la señal y vuelve a intentarlo.")


def choose_provider(available, preference="auto"):
    available = list(available)
    if preference != "auto" and preference in available:
        return preference
    return next((p for p in ("climateapp_private", "climateapp", "ecmwf", "icon", "gfs", "metno", "weathernext3")
                 if p in available), None)
