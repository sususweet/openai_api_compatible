"""Unit tests for custom auth header helpers."""

import unittest
from unittest.mock import patch

from models.common_openai import (
    AUTH_TYPE_API_KEY_HEADER,
    auth_requests_session,
    build_auth_header,
    build_request_headers,
    get_auth_type,
    merge_auth_into_headers,
    uses_default_openai_auth,
)


class TestAuthHeaders(unittest.TestCase):
    def test_default_openai_auth(self):
        credentials = {"api_key": "secret-key"}
        self.assertEqual(build_auth_header(credentials), ("Authorization", "Bearer secret-key"))
        self.assertTrue(uses_default_openai_auth(credentials))

    def test_api_key_header_auth_type(self):
        credentials = {
            "api_key": "secret-key",
            "auth_type": AUTH_TYPE_API_KEY_HEADER,
            "auth_header_name": "X-API-Key",
        }
        self.assertEqual(get_auth_type(credentials), AUTH_TYPE_API_KEY_HEADER)
        self.assertEqual(build_auth_header(credentials), ("X-API-Key", "secret-key"))
        self.assertFalse(uses_default_openai_auth(credentials))

    def test_api_key_header_strips_whitespace(self):
        credentials = {
            "api_key": "  secret-key  ",
            "auth_type": AUTH_TYPE_API_KEY_HEADER,
            "auth_header_name": "X-API-Key",
        }
        self.assertEqual(build_auth_header(credentials), ("X-API-Key", "secret-key"))

    def test_merge_auth_replaces_sdk_authorization_for_custom_header(self):
        credentials = {
            "api_key": "secret-key",
            "auth_type": AUTH_TYPE_API_KEY_HEADER,
            "auth_header_name": "X-API-Key",
        }
        merged = merge_auth_into_headers(
            credentials,
            {"Content-Type": "application/json", "Authorization": "Bearer secret-key"},
        )
        self.assertEqual(merged["X-API-Key"], "secret-key")
        self.assertNotIn("Authorization", merged)

    def test_build_request_headers_omits_auth_without_api_key(self):
        headers = build_request_headers(
            {"api_key": ""},
            {"Content-Type": "application/json"},
        )
        self.assertEqual(headers, {"Content-Type": "application/json"})

    @patch("requests.post")
    def test_auth_requests_session_injects_x_api_key(self, mock_post):
        import requests

        credentials = {
            "api_key": "secret-key",
            "auth_type": AUTH_TYPE_API_KEY_HEADER,
            "auth_header_name": "X-API-Key",
        }
        with auth_requests_session(credentials):
            requests.post(
                "https://api.example.com/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": "Bearer secret-key"},
                json={"model": "test"},
            )

        mock_post.assert_called_once()
        sent_headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(sent_headers["X-API-Key"], "secret-key")
        self.assertNotIn("Authorization", sent_headers)


if __name__ == "__main__":
    unittest.main()
