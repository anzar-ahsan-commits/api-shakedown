"""Deterministic sample values from a schema: the fallback when no LLM body is usable."""

from typing import Any

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


def sample_from_schema(schema: Any, _depth: int = 0) -> Any:
    if not isinstance(schema, dict) or _depth > MAX_DEPTH or schema.get("x-circular"):
        return None
    for key in ("example", "default", "const"):
        if key in schema:
            return schema[key]
    if isinstance(schema.get("examples"), list) and schema["examples"]:
        return schema["examples"][0]
    if schema.get("enum"):
        return schema["enum"][0]

    if "allOf" in schema:
        parts = [sample_from_schema(sub, _depth + 1) for sub in schema["allOf"]]
        if all(isinstance(part, dict) for part in parts):
            merged: dict[str, Any] = {}
            for part in parts:
                merged.update(part)
            return merged
        return next((part for part in parts if part is not None), None)
    for key in ("oneOf", "anyOf"):
        if schema.get(key):
            return sample_from_schema(schema[key][0], _depth + 1)

    kind = _type_of(schema)
    if kind == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        result = {}
        for name, prop in properties.items():
            value = sample_from_schema(prop, _depth + 1)
            if value is not None or name in required:
                result[name] = value
        return result
    if kind == "array":
        item = sample_from_schema(schema.get("items"), _depth + 1)
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
