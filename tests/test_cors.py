import sys
import unittest
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.modules.setdefault("insightface", MagicMock())
sys.modules.setdefault("insightface.app", MagicMock())

from main import configure_cors, parse_cors_origins


class CorsConfigTests(unittest.TestCase):
    def test_empty_value_disables_cors(self):
        self.assertEqual(parse_cors_origins(""), [])
        self.assertEqual(parse_cors_origins(" , "), [])

    def test_wildcard_is_ignored(self):
        self.assertEqual(parse_cors_origins("*"), [])
        self.assertEqual(
            parse_cors_origins("https://app.exemplo.gov.br, *"),
            ["https://app.exemplo.gov.br"],
        )

    def test_explicit_origins_are_normalized(self):
        self.assertEqual(
            parse_cors_origins(" https://app.exemplo.gov.br/ ,https://homo.exemplo.gov.br "),
            ["https://app.exemplo.gov.br", "https://homo.exemplo.gov.br"],
        )

    def test_middleware_is_not_installed_without_origins(self):
        application = FastAPI()
        configure_cors(application, [])
        self.assertFalse(any(item.cls is CORSMiddleware for item in application.user_middleware))

    def test_allowlist_does_not_enable_credentials_or_wildcard(self):
        application = FastAPI()
        configure_cors(application, ["https://app.exemplo.gov.br"])
        cors = next(item for item in application.user_middleware if item.cls is CORSMiddleware)
        self.assertEqual(cors.kwargs["allow_origins"], ["https://app.exemplo.gov.br"])
        self.assertFalse(cors.kwargs["allow_credentials"])
        self.assertNotIn("*", cors.kwargs["allow_origins"])
        self.assertNotIn("*", cors.kwargs["allow_methods"])
        self.assertNotIn("*", cors.kwargs["allow_headers"])


if __name__ == "__main__":
    unittest.main()
