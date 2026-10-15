"""Shared helpers for the generated API Shakedown test runners.

Environment variables:
  SHAKEDOWN_TARGET_URL     Server under test, including any base path
                           (default: the generated WireMock stubs).
  SHAKEDOWN_SELECT_STATUS  "1" to send X-Shakedown-Status, so the stubs return each
                           case's expected status. Leave unset for real servers.
  SHAKEDOWN_API_KEY        Credential for apiKey and bearer security (default: test-key).
  SHAKEDOWN_BASIC_AUTH     "user:password" for HTTP basic security.
  SHAKEDOWN_TIMEOUT        Request timeout in seconds (default: 10).
"""

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from openapi_schema_validator import OAS30ReadValidator, OAS31Validator

SELECT_STATUS = os.environ.get("SHAKEDOWN_SELECT_STATUS") == "1"
API_KEY = os.environ.get("SHAKEDOWN_API_KEY", "test-key")
BASIC_AUTH = os.environ.get("SHAKEDOWN_BASIC_AUTH", "user:password")
TIMEOUT = float(os.environ.get("SHAKEDOWN_TIMEOUT", "10"))


def load_cases(runner_file: str, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(runner_file).with_name(name).read_text("utf-8"))
    return data


def target_url(data: dict[str, Any]) -> str:
    return str(os.environ.get("SHAKEDOWN_TARGET_URL", data["default_target"])).rstrip("/")


def build_request(case: dict[str, Any], status_header: str, select: str | None) -> dict[str, Any]:
    """httpx request kwargs for a case. `select` is the stub status to ask for, if any."""
    path = case["path"]
    for name, value in case["path_params"].items():
        path = path.replace("{" + name + "}", quote(str(value), safe=""))
    headers = dict(case["headers"])
    # query_pairs is the query exactly as OpenAPI serializes it (exploded objects, repeated
    # array items, ...); harnesses generated before it existed only have `query`.
    # An empty list is meaningful (e.g. an exploded empty object sends nothing).
    raw_pairs = case["query_pairs"] if "query_pairs" in case else list(case["query"].items())
    params: list[tuple[str, str]] = [(str(key), str(value)) for key, value in raw_pairs]
    cookies: dict[str, str] = {}
    for auth in case["auth"]:
        if auth["scheme"] == "bearer":
            headers["Authorization"] = f"Bearer {API_KEY}"
        elif auth["scheme"] == "basic":
            headers["Authorization"] = "Basic " + base64.b64encode(BASIC_AUTH.encode()).decode()
        elif auth["location"] == "query":
            params.append((auth["name"], API_KEY))
        else:
            {"header": headers, "cookie": cookies}[auth["location"]][auth["name"]] = API_KEY
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    if SELECT_STATUS and select:
        headers[status_header] = select

    request: dict[str, Any] = {"method": case["method"].upper(), "url": path, "params": params}
    content_type = case.get("body_content_type") or "application/json"
    if case.get("raw_body") is not None:
        request["content"] = case["raw_body"]
        headers["Content-Type"] = content_type
    elif case.get("body") is not None:
        body = case["body"]
        if "json" in content_type:
            request["json"] = body
        elif content_type == "application/x-www-form-urlencoded" and isinstance(body, dict):
            request["data"] = body
        else:
            request["content"] = body if isinstance(body, str) else json.dumps(body)
            headers["Content-Type"] = content_type
    request["headers"] = headers
    return request


def status_matches(actual: int, expected: str, others: list[str]) -> bool:
    if expected.isdigit():
        return actual == int(expected)
    if expected[0].isdigit():  # e.g. "4XX"
        return str(actual)[0] == expected[0]
    # "default": any status the operation does not document explicitly
    return not any(status_matches(actual, s, []) for s in others if s != "default")


def schema_problems(schema: dict[str, Any], value: Any, openapi_version: str) -> list[str]:
    validator: Any = OAS30ReadValidator if openapi_version.startswith("3.0") else OAS31Validator
    try:
        errors = list(validator(schema).iter_errors(value))
    except Exception:  # e.g. a circular $ref that cannot be resolved here
        return []
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}" for e in errors]
