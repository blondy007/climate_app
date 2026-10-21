"""Single-owner login. Configure locally with an interactive, hidden password prompt."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
from datetime import timedelta

import streamlit as st

from climate_app.data.paths import DATA_DIR
from climate_app.forecast.domain import instant, stamp, utcnow

OWNER_FILE = DATA_DIR / "auth-owner.json"
HTTPS_MARKER = DATA_DIR / "https-verified.json"
ITERATIONS = 600_000


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest}"


def verify_password(password, encoded):
    try:
        algorithm, iterations, salt, expected = encoded.split("$")
        count = int(iterations)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= count <= 2_000_000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), count).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def owner_config():
    username = os.environ.get("CLIMATE_OWNER_USERNAME", "")
    encoded = os.environ.get("CLIMATE_OWNER_PASSWORD_HASH", "")
    if username and encoded:
        return {"username": username, "password_hash": encoded}
    if OWNER_FILE.exists():
        try:
            config = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
            if config.get("username") and config.get("password_hash"):
                return config
        except (OSError, ValueError):
            pass
    return None


def fingerprint(config):
    return hashlib.sha256((config["username"] + config["password_hash"]).encode()).hexdigest()


def authenticated():
    config = owner_config()
    issued = st.session_state.get("owner_login_time")
    valid = bool(config and issued and utcnow() - instant(issued) < timedelta(hours=24)
                 and st.session_state.get("owner_authenticated") == fingerprint(config))
    session = st.session_state.get("owner_browser_session")
    if valid and session:
        from climate_app.browser_auth import BrowserSessions
        valid = BrowserSessions().valid(session, fingerprint(config))
    return valid


def clear_owner_session():
    for key in ("owner_authenticated", "owner_login_time", "owner_browser_session", "owner_browser_expires"):
        st.session_state.pop(key, None)


def restore_owner_session():
    """Read the initial connection cookie server-side, once per web session."""
    state = st.session_state
    if state.get("owner_cookie_checked"):
        return
    state["owner_cookie_checked"] = True
    if state.get("owner_logout_pending") or state.get("owner_authenticated"):
        return
    from climate_app.browser_auth import BrowserSessions, COOKIE
    config = owner_config()
    cookie = st.context.cookies.get(COOKIE)
    restored = BrowserSessions().identify(cookie, fingerprint(config)) if config and cookie else None
    if restored:
        state["owner_authenticated"] = fingerprint(config)
        state["owner_login_time"] = stamp(utcnow())
        state["owner_browser_session"] = restored["session_hash"]
        state["owner_browser_expires"] = restored["expires_at"]


def owner_login(store):
    with st.expander("Acceso personal · WeatherNext", expanded=False):
        if authenticated():
            st.success("Sesión personal iniciada")
            if st.session_state.get("owner_browser_expires"):
                expiry = instant(st.session_state["owner_browser_expires"])
                st.caption(f"Este navegador recordará tu acceso hasta el {expiry:%d/%m/%Y}, salvo que cierres sesión antes.")
            if st.session_state.get("owner_cookie_warning"):
                st.warning(st.session_state["owner_cookie_warning"])
            if st.button("Cerrar sesión"):
                from climate_app.browser_auth import BrowserSessions
                sessions = BrowserSessions(store)
                sessions.revoke(st.session_state.get("owner_browser_session"))
                pending = st.session_state.get("owner_cookie_action", {})
                if pending.get("kind") == "remember":
                    sessions.cancel_exchange(pending.get("code"))
                clear_owner_session()
                st.session_state["owner_logout_pending"] = True
                st.session_state["owner_cookie_action"] = {"kind": "logout", "nonce": secrets.token_urlsafe(12)}
                st.rerun()
            return
        st.caption("Inicia sesión para consultar WeatherNext y el consenso completo.")
        config = owner_config()
        if not config:
            st.info("El acceso personal está pendiente de configurar por el propietario.")
            return
        if os.environ.get("CLIMATE_HTTPS_VERIFIED") != "1" and not HTTPS_MARKER.exists():
            st.info("El acceso personal se habilitará al verificar la conexión HTTPS.")
            return
        with st.form("owner_login", clear_on_submit=True):
            username = st.text_input("Usuario", max_chars=80, key="owner_username")
            password = st.text_input("Contraseña", type="password", max_chars=256, key="owner_password")
            remember = st.checkbox("Mantener sesión en este dispositivo", value=True,
                                   help="Recuerda tu acceso durante 30 días. Puedes revocarlo con Cerrar sesión.")
            submit = st.form_submit_button("Entrar")
        if submit:
            now = utcnow()
            window = str(int(now.timestamp()) // 900)
            if not store.reserve("owner-login", window, 1, 5):
                st.error("Demasiados intentos. Inténtalo dentro de unos minutos.")
                return
            correct_password = verify_password(password, config["password_hash"])
            if hmac.compare_digest(username.encode(), config["username"].encode()) and correct_password:
                st.session_state["owner_authenticated"] = fingerprint(config)
                st.session_state["owner_login_time"] = stamp(now)
                st.session_state["owner_logout_pending"] = False
                st.session_state.pop("owner_cookie_warning", None)
                if remember:
                    from climate_app.browser_auth import BrowserSessions
                    code = BrowserSessions(store).issue_exchange(fingerprint(config))
                    st.session_state["owner_cookie_action"] = {"kind": "remember", "code": code,
                                                              "nonce": secrets.token_urlsafe(12)}
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


def main():
    parser = argparse.ArgumentParser(description="Configurar el acceso del propietario sin mostrar contraseñas")
    parser.add_argument("command", choices=["configure", "verify-https"])
    args = parser.parse_args()
    if args.command == "verify-https":
        import requests
        from urllib.parse import urlsplit
        origin = os.environ.get("CLIMATE_PUBLIC_URL", "https://climate-app.51-170-61-130.sslip.io").rstrip("/")
        if urlsplit(origin).scheme != "https":
            raise SystemExit("Se requiere una URL HTTPS")
        result = requests.get(origin + "/_stcore/health", timeout=20)
        result.raise_for_status()
        if urlsplit(result.url).scheme != "https" or urlsplit(result.url).hostname != urlsplit(origin).hostname:
            raise SystemExit("Redirección HTTPS inesperada")
        HTTPS_MARKER.parent.mkdir(parents=True, exist_ok=True)
        HTTPS_MARKER.write_text(json.dumps({"origin": origin, "verified_at": stamp(utcnow())}), encoding="utf-8")
        print("HTTPS y certificado verificados. El formulario de acceso personal ya puede habilitarse.")
    if args.command == "configure":
        import getpass
        username = input("Usuario: ").strip()
        password = getpass.getpass("Contraseña (mínimo 12 caracteres): ")
        confirmation = getpass.getpass("Repetir contraseña: ")
        if not username or len(username) > 80 or len(password) < 12 or password != confirmation:
            raise SystemExit("Datos inválidos; no se ha modificado la cuenta")
        OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = OWNER_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"username": username, "password_hash": password_hash(password)}), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(OWNER_FILE)
        print("Cuenta configurada. La contraseña no se ha mostrado ni guardado en texto plano.")


if __name__ == "__main__":
    main()
