"""Revocable remembered owner sessions. The browser cookie is Secure and HttpOnly."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import timedelta
from urllib.parse import urlsplit

from tornado.web import HTTPError, RequestHandler

from climate_app.forecast.domain import stamp, utcnow
from climate_app.forecast.store import Store

COOKIE = "__Host-climateapp_owner"
SESSION_DAYS = 30


def token_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", value):
        return None
    return hashlib.sha256(value.encode()).hexdigest()


class BrowserSessions:
    def __init__(self, store=None):
        self.store = store or Store()

    def _issue(self, connection, kind, owner, expiry, session_hash=None):
        token = secrets.token_urlsafe(32)
        connection.execute("INSERT INTO owner_browser_tokens VALUES (?,?,?,?,?)",
                           (token_hash(token), kind, owner, session_hash, stamp(expiry)))
        return token

    def issue_exchange(self, owner, now=None):
        now = now or utcnow()
        with self.store.connect() as connection:
            connection.execute("DELETE FROM owner_browser_tokens WHERE expires_at<=?", (stamp(now),))
            return self._issue(connection, "exchange", owner, now + timedelta(seconds=90))

    def exchange(self, code, owner, now=None):
        now = now or utcnow()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM owner_browser_tokens WHERE token_hash=? AND kind='exchange'",
                                     (token_hash(code),)).fetchone()
            if not row or row["owner_fingerprint"] != owner or row["expires_at"] <= stamp(now):
                return None
            expires = now + timedelta(days=SESSION_DAYS)
            cookie = self._issue(connection, "session", owner, expires)
            connection.execute("UPDATE owner_browser_tokens SET kind='consumed_exchange',session_hash=? WHERE token_hash=?",
                               (token_hash(cookie), row["token_hash"]))
            proof = self._issue(connection, "proof", owner, now + timedelta(seconds=60), token_hash(cookie))
            return {"cookie": cookie, "proof": proof, "expires_at": stamp(expires)}

    def restore(self, cookie, owner, now=None):
        now = now or utcnow()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM owner_browser_tokens WHERE token_hash=? AND kind='session' "
                                     "AND owner_fingerprint=? AND expires_at>?",
                                     (token_hash(cookie), owner, stamp(now))).fetchone()
            if not row:
                return None
            connection.execute("DELETE FROM owner_browser_tokens WHERE kind='proof' AND expires_at<=?", (stamp(now),))
            proof = self._issue(connection, "proof", owner, now + timedelta(seconds=60), row["token_hash"])
            return {"proof": proof, "expires_at": row["expires_at"]}

    def consume_proof(self, proof, owner, now=None):
        now = now or utcnow()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM owner_browser_tokens WHERE token_hash=? AND kind='proof' "
                                     "AND owner_fingerprint=? AND expires_at>?",
                                     (token_hash(proof), owner, stamp(now))).fetchone()
            if not row:
                return None
            connection.execute("DELETE FROM owner_browser_tokens WHERE token_hash=?", (row["token_hash"],))
            session = connection.execute("SELECT expires_at FROM owner_browser_tokens WHERE token_hash=? "
                                         "AND kind='session' AND owner_fingerprint=? AND expires_at>?",
                                         (row["session_hash"], owner, stamp(now))).fetchone()
            return {"session_hash": row["session_hash"], "expires_at": session["expires_at"]} if session else None

    def valid(self, session_hash, owner, now=None):
        with self.store.connect() as connection:
            return bool(connection.execute("SELECT 1 FROM owner_browser_tokens WHERE token_hash=? AND kind='session' "
                                            "AND owner_fingerprint=? AND expires_at>?",
                                            (session_hash, owner, stamp(now or utcnow()))).fetchone())

    def identify(self, cookie, owner, now=None):
        """Restore directly from the HttpOnly cookie on the trusted web connection."""
        with self.store.connect() as connection:
            row = connection.execute("SELECT token_hash,expires_at FROM owner_browser_tokens WHERE token_hash=? "
                                     "AND kind='session' AND owner_fingerprint=? AND expires_at>?",
                                     (token_hash(cookie), owner, stamp(now or utcnow()))).fetchone()
            return {"session_hash": row["token_hash"], "expires_at": row["expires_at"]} if row else None

    def revoke(self, session_hash):
        with self.store.connect() as connection:
            connection.execute("DELETE FROM owner_browser_tokens WHERE token_hash=? OR session_hash=?",
                               (session_hash, session_hash))

    def cancel_exchange(self, code):
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT session_hash FROM owner_browser_tokens WHERE token_hash=? "
                                     "AND kind IN ('exchange','consumed_exchange')", (token_hash(code),)).fetchone()
            connection.execute("DELETE FROM owner_browser_tokens WHERE token_hash=?", (token_hash(code),))
            if row and row["session_hash"]:
                connection.execute("DELETE FROM owner_browser_tokens WHERE token_hash=? OR session_hash=?",
                                   (row["session_hash"], row["session_hash"]))


class BrowserAuthHandler(RequestHandler):
    def initialize(self, sessions=None):
        self.sessions = sessions or BrowserSessions()

    def set_default_headers(self):
        self.set_header("Cache-Control", "no-store")
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("X-Content-Type-Options", "nosniff")

    def prepare(self):
        forwarded = self.request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
        secure = self.request.protocol == "https" or forwarded == "https"
        host = urlsplit("//" + self.request.host).hostname
        if not secure and host not in ("localhost", "127.0.0.1", "::1"):
            raise HTTPError(403)
        origin = self.request.headers.get("Origin")
        expected = ("https" if secure else "http") + "://" + self.request.host
        if (origin and origin != expected) or self.request.headers.get("Sec-Fetch-Site") == "cross-site":
            raise HTTPError(403)
        if self.request.method == "POST":
            if origin != expected or self.request.headers.get("X-ClimateApp-Request") != "owner-session":
                raise HTTPError(403)
            if self.request.headers.get("Content-Type", "").split(";")[0] != "application/json" or len(self.request.body) > 4096:
                raise HTTPError(400)

    def current_owner(self):
        from climate_app.auth import fingerprint, owner_config
        config = owner_config()
        return fingerprint(config) if config else None

    def get(self, action):
        if action != "session":
            raise HTTPError(404)
        csrf = self.xsrf_token.decode()
        owner = self.current_owner()
        restored = self.sessions.restore(self.get_cookie(COOKIE), owner) if owner else None
        self.write({"authenticated": bool(restored), "csrf": csrf, **(restored or {})})

    def post(self, action):
        if action == "logout":
            self.sessions.revoke(token_hash(self.get_cookie(COOKIE)))
            self.clear_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="Strict")
            self.write({"authenticated": False})
            return
        if action != "remember":
            raise HTTPError(404)
        try:
            body = json.loads(self.request.body)
        except (ValueError, UnicodeDecodeError):
            raise HTTPError(400) from None
        if not isinstance(body, dict):
            raise HTTPError(400)
        owner = self.current_owner()
        issued = self.sessions.exchange(body.get("code"), owner) if owner else None
        if not issued:
            raise HTTPError(401)
        self.sessions.revoke(token_hash(self.get_cookie(COOKIE)))
        self.set_cookie(COOKIE, issued["cookie"], max_age=SESSION_DAYS * 86400, path="/",
                        secure=True, httponly=True, samesite="Strict")
        self.write({"authenticated": True, "proof": issued["proof"], "expires_at": issued["expires_at"]})


def install_routes():
    """Register on the pinned Streamlit server, sharing its port and XSRF protection."""
    from streamlit.web.server.server import Server
    if getattr(Server, "_climateapp_auth_installed", False):
        return
    original = Server._create_app
    def create_app(server):
        application = original(server)
        application.add_handlers(r".*", [(r"/_climate/auth/(session|remember|logout)", BrowserAuthHandler)])
        return application
    Server._create_app = create_app
    Server._climateapp_auth_installed = True
