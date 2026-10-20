import unittest
from unittest.mock import patch

from climate_app.ui import appearance


class AppearanceTests(unittest.TestCase):
    def sync(self, state, params):
        with patch.object(appearance.st, "session_state", state), patch.object(appearance.st, "query_params", params), patch.object(appearance, "html"), patch.object(appearance.st, "radio"), patch.object(appearance.st, "markdown"):
            appearance.sync_appearance()
            return appearance.palette()

    def test_explicit_url_choice_restores_the_palette(self):
        state = {}
        self.assertEqual(self.sync(state, {"theme": "dark"}), appearance.PALETTES["dark"])
        self.assertEqual(state["appearance"], "dark")
        self.assertEqual(self.sync(state, {"theme": "light"}), appearance.PALETTES["light"])

    def test_automatic_uses_the_browser_scheme_for_initial_charts(self):
        for scheme in ("light", "dark"):
            self.assertEqual(self.sync({}, {"theme": "auto", "scheme": scheme}), appearance.PALETTES[scheme])

    def test_invalid_or_unavailable_preference_has_a_safe_default(self):
        for params in ({}, {"theme": "other", "scheme": "other"}):
            self.assertEqual(self.sync({}, params), appearance.PALETTES["light"])
