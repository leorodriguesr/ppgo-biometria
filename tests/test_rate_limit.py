import sys
import unittest
from unittest.mock import MagicMock

sys.modules.setdefault("insightface", MagicMock())
sys.modules.setdefault("insightface.app", MagicMock())

from main import (
    AUTH_FAILURE_LIMIT,
    _auth_failures,
    _auth_lock,
    _rate_hits,
    _rate_lock,
    app,
    auth_block_remaining,
    health,
    rate_limit_allows,
    register_auth_failure,
)


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        with _rate_lock:
            _rate_hits.clear()

    def test_blocks_after_the_configured_limit(self):
        for _ in range(3):
            self.assertTrue(rate_limit_allows("ip:10.0.0.1", 3, now=1000))
        self.assertFalse(rate_limit_allows("ip:10.0.0.1", 3, now=1000))

    def test_window_expiry_allows_again(self):
        self.assertTrue(rate_limit_allows("ip:10.0.0.2", 1, now=1000))
        self.assertFalse(rate_limit_allows("ip:10.0.0.2", 1, now=1000))
        self.assertTrue(rate_limit_allows("ip:10.0.0.2", 1, now=1061))

    def test_keys_are_independent(self):
        self.assertTrue(rate_limit_allows("ip:10.0.0.3", 1, now=1000))
        self.assertTrue(rate_limit_allows("token:abc", 1, now=1000))

    def test_zero_disables_the_limit(self):
        self.assertTrue(rate_limit_allows("ip:10.0.0.4", 0, now=1000))
        self.assertTrue(rate_limit_allows("ip:10.0.0.4", 0, now=1000))


class ReportHardeningTests(unittest.TestCase):
    def setUp(self):
        with _auth_lock:
            _auth_failures.clear()

    def test_auth_failures_block_temporarily(self):
        for _ in range(AUTH_FAILURE_LIMIT):
            register_auth_failure("10.1.1.1", now=2000)
        self.assertGreater(auth_block_remaining("10.1.1.1", now=2000), 0)
        self.assertEqual(auth_block_remaining("10.1.1.1", now=2901), 0)

    def test_health_hides_model_details(self):
        import asyncio

        payload = asyncio.run(health())
        self.assertEqual(set(payload), {"status"})
        self.assertIn(payload["status"], {"ok", "degraded"})

    def test_public_docs_are_disabled(self):
        self.assertIsNone(app.docs_url)
        self.assertIsNone(app.openapi_url)
        schema = app.openapi()
        operation = schema["paths"]["/generate-embedding"]["post"]
        self.assertEqual(
            operation["security"],
            [{"BearerAuth": []}, {"TokenHeader": []}],
        )


if __name__ == "__main__":
    unittest.main()
