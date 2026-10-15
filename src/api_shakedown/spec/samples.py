"""Deterministic sample values from a schema: the fallback when no LLM body is usable."""

import contextlib
import json
from typing import Any, Literal

from api_shakedown.spec.models import ApiSpec, Body
from api_shakedown.spec.schema_check import schema_errors

MAX_DEPTH = 8

_STRING_FORMATS = {
    "date-time": "2024-01-01T12:00:00Z",
    "date": "2024-01-01",
    "time": "12:00:00",
    "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "email": "user@example.com",
    "uri": "https://example.com/resource",
    "url": "https://example.com/resource",
    "hostname": "example.com",
    "ipv4": "192.0.2.1",
    "ipv6": "2001:db8::1",
    "byte": "ZXhhbXBsZQ==",
    "password": "s3cret-Passw0rd",
}


def valid_sample(
    schema: dict[str, Any] | None,
    example: Any,
    openapi_version: str,
    direction: Literal["read", "write"] = "read",
) -> Any:
    """The first candidate that validates against `schema`.

    Real specs often carry examples that contradict their own schemas: JSON written as a
    string, or nested `example`s of the wrong type. Candidates, in order: the example as
    given, the example parsed from JSON text, a schema sample that uses nested examples,
    and one that ignores them. If none validates, the last one is returned.
    """
    if schema is None:
        return example
    candidates: list[Any] = []
    if example is not None:
        candidates.append(example)
        if isinstance(example, str):
            with contextlib.suppress(ValueError):
                candidates.append(json.loads(example))
    candidates += [sample_from_schema(schema), sample_from_schema(schema, use_examples=False)]
    for candidate in candidates:
        if candidate is not None and not schema_errors(
            schema, candidate, openapi_version, direction
        ):
            return candidate
    return candidates[-1]


def sample_from_schema(schema: Any, _depth: int = 0, *, use_examples: bool = True) -> Any:
    """A sample value for `schema`. `use_examples=False` ignores example/examples."""
    if not isinstance(schema, dict) or _depth > MAX_DEPTH or schema.get("x-circular"):
        return None
    for key in ("example", "default", "const") if use_examples else ("default", "const"):
        if key in schema:
            return schema[key]
    if use_examples and isinstance(schema.get("examples"), list) and schema["examples"]:
        return schema["examples"][0]
    if schema.get("enum"):
        return schema["enum"][0]

    if "allOf" in schema:
        parts = [
            sample_from_schema(sub, _depth + 1, use_examples=use_examples)
            for sub in schema["allOf"]
        ]
        if all(isinstance(part, dict) for part in parts):
            merged: dict[str, Any] = {}
            for part in parts:
                merged.update(part)
            return merged
        return next((part for part in parts if part is not None), None)
    for key in ("oneOf", "anyOf"):
        if schema.get(key):
            return sample_from_schema(schema[key][0], _depth + 1, use_examples=use_examples)

    kind = _type_of(schema)
    if kind == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        result = {}
        for name, prop in properties.items():
            value = sample_from_schema(prop, _depth + 1, use_examples=use_examples)
            if value is not None or name in required:
                result[name] = value
        return result
    if kind == "array":
        item = sample_from_schema(schema.get("items"), _depth + 1, use_examples=use_examples)
        count = max(1, int(schema.get("minItems", 1)))
        return [item] * count if item is not None else []
    if kind == "string":
        return _string(schema)
    if kind in ("integer", "number"):
        return _number(schema, integer=kind == "integer")
    if kind == "boolean":
        return True
    return None


def _type_of(schema: dict[str, Any]) -> str | None:
    kind = schema.get("type")
    if isinstance(kind, list):  # OpenAPI 3.1: ["string", "null"]
        kind = next((k for k in kind if k != "null"), None)
    if kind is None:
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
    return kind if isinstance(kind, str) else None


def _string(schema: dict[str, Any]) -> str:
    value = _STRING_FORMATS.get(str(schema.get("format")), "string")
    min_length = int(schema.get("minLength", 0))
    max_length = schema.get("maxLength")
    if len(value) < min_length:
        value = value + "x" * (min_length - len(value))
    if max_length is not None:
        value = value[: int(max_length)]
    return value


def _number(schema: dict[str, Any], *, integer: bool) -> int | float:
    low = schema.get("minimum")
    high = schema.get("maximum")
    if isinstance(schema.get("exclusiveMinimum"), (int, float)):  # 3.1 numeric form
        low = schema["exclusiveMinimum"] + 1
    value = low if low is not None else 1
    if high is not None and value > high:
        value = high
    return int(value) if integer else float(value)


def example_problems(spec: ApiSpec) -> list[str]:
    """Where the spec's own JSON examples contradict its schemas (worth telling its authors)."""
    notes = []
    for op in spec.operations:
        bodies: list[tuple[str, Body | None, Literal["read", "write"]]] = [
            (f"{op.id} request body", op.request_body, "write"),
            *((f"{op.id} {r.status}", r.body, "read") for r in op.responses),
        ]
        for label, body, direction in bodies:
            if body is None or body.schema_ is None or "json" not in body.content_type:
                continue
            version, schema = spec.openapi_version, body.schema_
            if body.example is not None:
                if not schema_errors(schema, body.example, version, direction):
                    continue
                parsed = _json_text(body.example)
                if parsed is not None and not schema_errors(schema, parsed, version, direction):
                    notes.append(f"{label}: the spec's example is JSON written as text; parsed it.")
                else:
                    notes.append(
                        f"{label}: the spec's example does not match its own schema; "
                        "generated a valid one instead."
                    )
                continue
            nested = sample_from_schema(schema)
            if nested is not None and schema_errors(schema, nested, version, direction):
                notes.append(
                    f"{label}: examples inside the spec's schema contradict it; ignored them."
                )
    return notes


def _json_text(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None
