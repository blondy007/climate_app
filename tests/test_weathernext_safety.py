import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from climate_app.forecast.domain import Location, ProviderError, utcnow
from climate_app.forecast.store import Store
from climate_app.forecast.weathernext import fetch_weathernext


class QuerySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "forecasts.sqlite3")
        self.location = Location("test", "Madrid", 40.4, -3.7)
        self.environment = patch.dict(os.environ, {"WEATHERNEXT_PROJECT": "climate-test",
                                                    "WEATHERNEXT_DATASET": "climate-test.weather"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_enabled_billing_never_creates_a_bigquery_client(self):
        with patch("google.auth.default", return_value=(Mock(), "climate-test")), \
             patch("google.auth.transport.requests.AuthorizedSession") as session, \
             patch("google.cloud.bigquery.Client") as client:
            response = session.return_value.__enter__.return_value.get.return_value
            response.status_code = 200
            response.json.return_value = {"billingEnabled": True}
            with self.assertRaises(ProviderError):
                fetch_weathernext(self.location, self.store)
            client.assert_not_called()

    def test_unknown_query_size_never_executes_a_real_query(self):
        with patch("google.auth.default", return_value=(Mock(), "climate-test")), \
             patch("google.auth.transport.requests.AuthorizedSession") as session, \
             patch("google.cloud.bigquery.Client") as client:
            response = session.return_value.__enter__.return_value.get.return_value
            response.status_code = 200
            response.json.return_value = {"billingEnabled": False}
            client.return_value.query.return_value.total_bytes_processed = None
            with self.assertRaises(ProviderError):
                fetch_weathernext(self.location, self.store)
            self.assertEqual(client.return_value.query.call_count, 1)
            self.assertTrue(client.return_value.query.call_args.kwargs["job_config"].dry_run)

    def test_completed_query_refunds_only_unused_reserved_bytes(self):
        now = utcnow()
        dry = Mock(total_bytes_processed=1024)
        job = Mock(total_bytes_processed=1024, total_bytes_billed=1024)
        job.result.return_value = [dict(time=now + timedelta(hours=2), init_time=now - timedelta(hours=1),
                                       temperature_init_time=now - timedelta(hours=1), temp_k=293.15)]
        with patch("google.auth.default", return_value=(Mock(), "climate-test")), \
             patch("google.auth.transport.requests.AuthorizedSession") as session, \
             patch("google.cloud.bigquery.Client") as client:
            response = session.return_value.__enter__.return_value.get.return_value
            response.status_code = 200
            response.json.return_value = {"billingEnabled": False}
            client.return_value.query.side_effect = [dry, job]
            forecast = fetch_weathernext(self.location, self.store)
            self.assertEqual(forecast.provider, "weathernext3")
            real_config = client.return_value.query.call_args.kwargs["job_config"]
            self.assertEqual(real_config.maximum_bytes_billed, 2 * 1024 ** 3)
        with self.store.connect() as connection:
            used = connection.execute("SELECT used FROM budget WHERE service='bigquery-bytes'").fetchone()[0]
        self.assertEqual(used, 1024)


if __name__ == "__main__":
    unittest.main()
