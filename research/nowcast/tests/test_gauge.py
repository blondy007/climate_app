from datetime import datetime, timezone
import unittest
from research.nowcast.gauge import parse_series


class GaugeTests(unittest.TestCase):
    def fixture(self):
        # In October, midnight on the local chart is 22:00 UTC the day before.
        epoch = int(datetime(2026, 10, 16, 22, tzinfo=timezone.utc).timestamp())
        raw = (epoch + 7200) * 1000
        return ("tipoDatosMHDMA: {text: 'I'} codigo:'06A18P01', um:'mm', "
                f"priFec:{epoch}, ultFec:{epoch + 600}, acuVal:0.3, data: ["
                f"{{x:{raw}, y:0.10, y0:0.10, ac:0.10, fb:'1', pr:'A'}},"
                f"{{x:{raw+300000}, y:null, y0:null, ac:null, fb:'0', pr:'null'}},"
                f"{{x:{raw+600000}, y:0.20, y0:0.20, ac:0.30, fb:'1', pr:'A'}}]")

    def test_chart_local_clock_converted_and_checked_against_internal_epochs(self):
        result = parse_series(self.fixture())
        self.assertEqual(result["records"][0]["utc"], "2026-10-16T22:00:00+00:00")
        self.assertIsNone(result["records"][1]["increment_mm"])
        self.assertEqual(result["missing_or_unusable"], 1)
        self.assertEqual(result["total_mm"], .3)

    def test_changed_units_or_clock_are_rejected(self):
        for changed in (self.fixture().replace("um:'mm'", "um:'m'"),
                        self.fixture().replace("priFec:", "priFec:1")):
            with self.assertRaises(ValueError):
                parse_series(changed)

    def test_changed_or_unparsed_data_cannot_silently_disappear(self):
        for changed in (self.fixture().replace("acuVal:0.3", "acuVal:9"),
                        self.fixture().replace("y0:null", "y0:unknown")):
            with self.assertRaises(ValueError):
                parse_series(changed)
