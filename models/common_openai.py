import json
import logging
from collections.abc import Mapping
from contextlib import contextmanager

import httpx
import openai
import requests
from httpx import Timeout
from openai import OpenAI

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

AUTH_TYPE_OPENAI_BEARER = "openai_bearer"
AUTH_TYPE_API_KEY_HEADER = "api_key_header"

logger = logging.getLogger(__name__)

_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "x-api-key", "api-key", "x-token", "apikey"}
)


def _mask_header_value(value: str) -> str:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _mask_headers(headers: Mapping[str, str]) -> dict[str, str]:
    masked: dict[str, str] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in _SENSITIVE_HEADER_NAMES or (
            "api" in key_lower and "key" in key_lower
        ):
            masked[key] = _mask_header_value(str(value))
        else:
            masked[key] = value
    return masked


def log_validation_auth_config(credentials: Mapping) -> None:
    logger.info(
        "Validate credentials auth config: auth_type=%s auth_header_name=%s "
        "auth_header_prefix=%s api_key_set=%s",
        get_auth_type(credentials),
        credentials.get("auth_header_name"),
        credentials.get("auth_header_prefix"),
        bool(_normalize_api_key(credentials)),
    )


def log_validation_http_request(
    url: str,
    headers: Mapping[str, str] | None = None,
    json_body: dict | None = None,
    stream: bool = False,
) -> None:
    safe_headers = _mask_headers(headers or {})
    message_parts = [
        "Validate credentials HTTP request:",
        f"POST {url}",
        f"headers={json.dumps(safe_headers, ensure_ascii=False)}",
    ]
    if json_body is not None:
        message_parts.append(f"json={json.dumps(json_body, ensure_ascii=False)}")
    if stream:
        message_parts.append("stream=true")
    logger.info(" | ".join(message_parts))


def _normalize_api_key(credentials: Mapping) -> str | None:
    api_key = credentials.get("api_key")
    if api_key is None:
        return None
    api_key = str(api_key).strip()
    return api_key or None


def get_auth_type(credentials: Mapping) -> str:
    explicit = credentials.get("auth_type")
    if explicit in (AUTH_TYPE_OPENAI_BEARER, AUTH_TYPE_API_KEY_HEADER):
        return explicit

    header_name = (credentials.get("auth_header_name") or "").strip()
    if header_name and header_name != "Authorization":
        return AUTH_TYPE_API_KEY_HEADER

    prefix = credentials.get("auth_header_prefix")
    if prefix is not None and prefix != "" and prefix != "Bearer ":
        return AUTH_TYPE_API_KEY_HEADER

    return AUTH_TYPE_OPENAI_BEARER


def build_auth_header(credentials: Mapping) -> tuple[str, str] | tuple[None, None]:
    api_key = _normalize_api_key(credentials)
    if not api_key:
        return None, None

    auth_type = get_auth_type(credentials)
    if auth_type == AUTH_TYPE_OPENAI_BEARER:
        return "Authorization", f"Bearer {api_key}"

    header_name = (credentials.get("auth_header_name") or "X-API-Key").strip() or "X-API-Key"
    prefix = credentials.get("auth_header_prefix") or ""
    value = f"{prefix}{api_key}" if prefix else api_key
    return header_name, value


def uses_default_openai_auth(credentials: Mapping) -> bool:
    return get_auth_type(credentials) == AUTH_TYPE_OPENAI_BEARER


def _strip_authorization_headers(headers: dict[str, str]) -> None:
    for key in list(headers.keys()):
        if key.lower() == "authorization":
            del headers[key]


def merge_auth_into_headers(
    credentials: Mapping,
    headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    merged: dict[str, str] = dict(headers or {})
    if not uses_default_openai_auth(credentials):
        _strip_authorization_headers(merged)

    extra_headers = credentials.get("extra_headers")
    if extra_headers:
        merged.update(extra_headers)

    header_name, header_value = build_auth_header(credentials)
    if header_name is not None:
        merged[header_name] = header_value

    return merged


def build_request_headers(
    credentials: Mapping,
    base_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return merge_auth_into_headers(credentials, base_headers)


def _patch_request_kwargs(credentials: Mapping, kwargs: dict) -> None:
    kwargs["headers"] = merge_auth_into_headers(credentials, kwargs.get("headers"))


@contextmanager
def auth_requests_session(credentials: Mapping, log_requests: bool = False):
    """Patch requests so SDK hardcoded Authorization Bearer can be overridden."""
    original_post = requests.post
    original_session_request = requests.Session.request

    def patched_post(url, *args, **kwargs):
        _patch_request_kwargs(credentials, kwargs)
        if log_requests:
            log_validation_http_request(
                url,
                headers=kwargs.get("headers"),
                json_body=kwargs.get("json"),
                stream=bool(kwargs.get("stream")),
            )
        return original_post(url, *args, **kwargs)

    def patched_session_request(self, method, url, *args, **kwargs):
        if str(method).upper() == "POST":
            _patch_request_kwargs(credentials, kwargs)
            if log_requests:
                log_validation_http_request(
                    url,
                    headers=kwargs.get("headers"),
                    json_body=kwargs.get("json"),
                    stream=bool(kwargs.get("stream")),
                )
        return original_session_request(self, method, url, *args, **kwargs)

    requests.post = patched_post
    requests.Session.request = patched_session_request
    try:
        yield
    finally:
        requests.post = original_post
        requests.Session.request = original_session_request


def create_openai_client(endpoint_url: str, credentials: Mapping) -> OpenAI:
    auth_headers = build_request_headers(credentials)
    if uses_default_openai_auth(credentials):
        return OpenAI(
            api_key=_normalize_api_key(credentials),
            base_url=endpoint_url,
            default_headers=auth_headers,
        )

    def auth_request_hook(request: httpx.Request) -> None:
        for key in list(request.headers.keys()):
            if key.lower() == "authorization":
                del request.headers[key]
        for name, value in auth_headers.items():
            request.headers[name] = value

    http_client = httpx.Client(
        event_hooks={"request": [auth_request_hook]},
        timeout=httpx.Timeout(300.0, connect=5.0),
    )
    return OpenAI(
        api_key="custom-auth-placeholder",
        base_url=endpoint_url,
        http_client=http_client,
        default_headers=auth_headers,
    )


class _CommonOpenAI:
    def _to_credential_kwargs(self, credentials: Mapping) -> dict:
        """
        Transform credentials to kwargs for model instance

        :param credentials:
        :return:
        """
        credentials_kwargs = {
            "api_key": credentials["api_key"],
            "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }

        if credentials.get("endpoint_url"):
            openai_api_base = credentials["endpoint_url"].rstrip("/")
            credentials_kwargs["base_url"] = openai_api_base + "/v1"

        if "openai_organization" in credentials:
            credentials_kwargs["organization"] = credentials["openai_organization"]

        return credentials_kwargs

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [openai.APIConnectionError, openai.APITimeoutError],
            InvokeServerUnavailableError: [openai.InternalServerError],
            InvokeRateLimitError: [openai.RateLimitError],
            InvokeAuthorizationError: [openai.AuthenticationError, openai.PermissionDeniedError],
            InvokeBadRequestError: [
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ],
        }
