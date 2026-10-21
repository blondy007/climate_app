import json
import tempfile
import unittest
from datetime import timedelta
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import patch

from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

from climate_app.browser_auth import BrowserAuthHandler, BrowserSessions, COOKIE, token_hash
from climate_app.forecast.domain import utcnow
from climate_app.forecast.store import Store

OWNER = "owner-fingerprint-fixture"


class BrowserSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sessions.sqlite3"
        self.sessions = BrowserSessions(Store(self.path))

    def login(self, now=None):
        code = self.sessions.issue_exchange(OWNER, now)
        return self.sessions.exchange(code, OWNER, now)

    def test_only_one_time_codes_and_proofs_can_establish_a_browser_session(self):
        code = self.sessions.issue_exchange(OWNER)
        self.assertIsNone(self.sessions.exchange("forged-code", OWNER))
        issued = self.sessions.exchange(code, OWNER)
        self.assertIsNone(self.sessions.exchange(code, OWNER))
        self.assertIsNone(self.sessions.consume_proof("forged-proof", OWNER))
        proof = self.sessions.consume_proof(issued["proof"], OWNER)
        self.assertTrue(self.sessions.valid(proof["session_hash"], OWNER))
        self.assertIsNone(self.sessions.consume_proof(issued["proof"], OWNER))

    def test_remembered_login_survives_process_recreation_but_expires_after_thirty_days(self):
        now = utcnow()
        issued = self.login(now)
        restarted = BrowserSessions(Store(self.path))
        self.assertTrue(restarted.restore(issued["cookie"], OWNER, now + timedelta(days=29)))
        self.assertIsNone(restarted.restore(issued["cookie"], OWNER, now + timedelta(days=31)))
        self.assertIsNone(restarted.restore(issued["cookie"], "changed-password-fingerprint", now))

    def test_logout_invalidates_cookie_and_any_outstanding_restore_proof(self):
        issued = self.login()
        restored = self.sessions.restore(issued["cookie"], OWNER)
        self.sessions.revoke(token_hash(issued["cookie"]))
        self.assertIsNone(self.sessions.restore(issued["cookie"], OWNER))
        self.assertIsNone(self.sessions.consume_proof(restored["proof"], OWNER))

    def test_logout_cancels_a_login_in_flight_even_after_cookie_issuance(self):
        code = self.sessions.issue_exchange(OWNER)
        issued = self.sessions.exchange(code, OWNER)
        self.sessions.cancel_exchange(code)
        self.assertIsNone(self.sessions.restore(issued["cookie"], OWNER))
        self.assertIsNone(self.sessions.consume_proof(issued["proof"], OWNER))

    def test_stale_grants_and_proofs_fail_closed(self):
        now = utcnow()
        code = self.sessions.issue_exchange(OWNER, now)
        self.assertIsNone(self.sessions.exchange(code, OWNER, now + timedelta(minutes=2)))
        issued = self.login(now)
        self.assertIsNone(self.sessions.consume_proof(issued["proof"], OWNER, now + timedelta(minutes=2)))

    def test_backup_restores_sessions_without_storing_raw_cookie_or_codes(self):
        issued = self.login()
        snapshot = self.sessions.store.backup(Path(self.temp.name) / "restored.sqlite3")
        restored = BrowserSessions(Store(snapshot))
        self.assertTrue(restored.restore(issued["cookie"], OWNER))
        with restored.store.connect() as connection:
            rows = [tuple(row) for row in connection.execute("SELECT * FROM owner_browser_tokens")]
        saved = json.dumps(rows)
        self.assertNotIn(issued["cookie"], saved)
        self.assertNotIn(issued["proof"], saved)

    def test_browser_bridge_requires_valid_proof_and_processes_a_batch_only_once(self):
        from climate_app.ui import browser_location
        from climate_app.auth import fingerprint
        config = {"username": "fixture-owner", "password_hash": "fixture-hash"}
        owner = fingerprint(config)
        issued = self.sessions.exchange(self.sessions.issue_exchange(owner), owner)
        state = {}
        with patch.object(browser_location.st, "session_state", state), \
             patch("climate_app.auth.st.session_state", state), \
             patch("climate_app.auth.owner_config", return_value=config), \
             patch("climate_app.browser_auth.BrowserSessions", return_value=self.sessions):
            browser_location._process_event({"kind": "auth", "proof": "forged-proof"}, state)
            self.assertNotIn("owner_authenticated", state)
            batch = {"events": [{"kind": "preferences", "nonce": "pref", "model": "icon"},
                                {"kind": "auth", "nonce": "auth", "proof": issued["proof"]}]}
            with patch.object(browser_location, "_bridge", return_value=batch):
                browser_location.sync_browser_location()
                browser_location.sync_browser_location()
            self.assertEqual(state["preferred_forecast_provider"], "icon")
            self.assertEqual(state["owner_authenticated"], owner)
            self.assertEqual(len(state["browser_processed_events"]), 2)


class FixtureAuthHandler(BrowserAuthHandler):
    def current_owner(self):
        return OWNER


class BrowserAuthHTTPTests(AsyncHTTPTestCase):
    def get_app(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sessions = BrowserSessions(Store(Path(self.temp.name) / "http-sessions.sqlite3"))
        return Application([(r"/_climate/auth/(session|remember|logout)", FixtureAuthHandler,
                             {"sessions": self.sessions})], xsrf_cookies=True)

    def csrf(self):
        response = self.fetch("/_climate/auth/session")
        cookies = SimpleCookie()
        for line in response.headers.get_list("Set-Cookie"):
            cookies.load(line)
        return json.loads(response.body)["csrf"], cookies["_xsrf"].value

    def post(self, action, body, csrf=None, cookie=None, **extra):
        if csrf is None:
            csrf, xsrf_cookie = self.csrf()
            cookie = "_xsrf=" + xsrf_cookie + ("; " + cookie if cookie else "")
        headers = {"Origin": self.get_url("").rstrip("/"), "Content-Type": "application/json",
                   "X-ClimateApp-Request": "owner-session", "X-Xsrftoken": csrf, "Cookie": cookie or "", **extra}
        return self.fetch("/_climate/auth/" + action, method="POST", headers=headers, body=json.dumps(body))

    def test_cookie_flags_restore_and_logout_over_http_endpoints(self):
        code = self.sessions.issue_exchange(OWNER)
        response = self.post("remember", {"code": code})
        self.assertEqual(response.code, 200)
        cookie = next(line for line in response.headers.get_list("Set-Cookie") if line.startswith(COOKIE + "="))
        for flag in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/", "Max-Age=2592000"):
            self.assertIn(flag, cookie)
        self.assertNotIn("Domain=", cookie)
        cookie_header = cookie.split(";", 1)[0]
        restored = self.fetch("/_climate/auth/session", headers={"Cookie": cookie_header})
        self.assertTrue(json.loads(restored.body)["authenticated"])
        self.assertNotIn("cookie", json.loads(restored.body))
        self.assertEqual(restored.headers["Cache-Control"], "no-store")
        self.assertEqual(self.post("logout", {}, cookie=cookie_header).code, 200)
        self.assertFalse(json.loads(self.fetch("/_climate/auth/session", headers={"Cookie": cookie_header}).body)["authenticated"])

    def test_csrf_and_origin_protections_are_required(self):
        code = self.sessions.issue_exchange(OWNER)
        self.assertEqual(self.post("remember", {"code": code}, Origin="https://other.example").code, 403)
        self.assertEqual(self.fetch("/_climate/auth/remember", method="POST", body="{}").code, 403)
        self.assertEqual(self.post("remember", {"code": code}, **{"X-ClimateApp-Request": ""}).code, 403)
        self.assertEqual(self.post("remember", {"code": code}).code, 200)
        self.assertEqual(self.post("remember", {"code": code}).code, 401)

    def test_arbitrary_browser_claim_or_insecure_public_request_does_not_authenticate(self):
        self.assertEqual(self.post("remember", {"authenticated": True}).code, 401)
        response = self.fetch("/_climate/auth/session", headers={"Host": "weather.example"})
        self.assertEqual(response.code, 403)


if __name__ == "__main__":
    unittest.main()
