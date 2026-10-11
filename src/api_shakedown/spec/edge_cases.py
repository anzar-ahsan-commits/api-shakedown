"""Deterministic edge-case values for a schema: boundaries, wrong types, odd strings.

Whether each value is accepted is decided by actually validating it against the schema,
so `valid` is always correct for the spec, never guessed.
"""

from typing import Any, Literal

from pydantic import BaseModel

from api_shakedown.spec.schema_check import schema_errors

LONG_STRING_LENGTH = 10_000
_FORMAT_VIOLATIONS = {
    "date-time": "2024-13-45T25:61:00Z",
    "date": "2024-02-30x",
    "uuid": "not-a-uuid",
    "email": "not-an-email",
    "uri": "not a uri",
    "ipv4": "999.1.1.1",
}


class EdgeCase(BaseModel):
    value: Any
    valid: bool
    description: str


def edge_cases(
    schema: dict[str, Any], openapi_version: str, direction: Literal["read", "write"] = "write"
) -> list[EdgeCase]:
    candidates = _candidates(schema)
    seen: set[str] = set()
    cases = []
    for value, description in candidates:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        valid = not schema_errors(schema, value, openapi_version, direction)
        valid = valid and _fits_format(schema, value)
        cases.append(EdgeCase(value=value, valid=valid, description=description))
    return cases


def object_edge_cases(schema: dict[str, Any], openapi_version: str) -> dict[str, list[EdgeCase]]:
    """Edge cases per top-level property of an object schema (allOf parts merged)."""
    return {
        name: edge_cases(prop, openapi_version)
        for name, prop in properties_of(schema).items()
        if isinstance(prop, dict)
    }


def required_properties(schema: dict[str, Any]) -> set[str]:
    required = set(schema.get("required") or [])
    for part in schema.get("allOf") or []:
        if isinstance(part, dict):
            required |= required_properties(part)
    return required


def properties_of(schema: dict[str, Any]) -> dict[str, Any]:
    properties = dict(schema.get("properties") or {})
    for part in schema.get("allOf") or []:
        if isinstance(part, dict):
            properties |= properties_of(part)
    return properties


def _candidates(schema: dict[str, Any]) -> list[tuple[Any, str]]:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    out: list[tuple[Any, str]] = [(None, "null")]

    if schema.get("enum"):
        out += [(v, f"enum member {v!r}") for v in schema["enum"]]
        out.append(("NOT_A_MEMBER", "value outside the enum"))
    if kind == "string" or (kind is None and "enum" not in schema and "properties" not in schema):
        out += _string_candidates(schema)
    if kind in ("integer", "number"):
        out += _number_candidates(schema, integer=kind == "integer")
    if kind == "boolean":
        out += [(True, "true"), (False, "false"), ("true", "boolean as string"), (0, "0")]
    if kind == "array":
        min_items, max_items = schema.get("minItems"), schema.get("maxItems")
        out += [([], "empty array"), ("not-an-array", "string instead of array")]
        if isinstance(max_items, int):
            out.append(([None] * (max_items + 1), f"maxItems+1 ({max_items + 1}) items"))
        if isinstance(min_items, int) and min_items > 0:
            out.append(([None] * (min_items - 1), f"minItems-1 ({min_items - 1}) items"))
    if kind == "object" or "properties" in schema:
        out += [({}, "empty object"), ([], "array instead of object")]
    if kind not in ("string", None):
        out.append(("a string", "string instead of " + str(kind)))
    if kind == "string":
        out.append((12345, "number instead of string"))
    return out


def _string_candidates(schema: dict[str, Any]) -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = [
        ("", "empty string"),
        ("   ", "whitespace only"),
        ("Zoë 名前 🐶", "non-ASCII and emoji"),
        ("'; DROP TABLE users;--", "SQL injection probe"),
        ("<script>alert(1)</script>", "HTML/script injection probe"),
    ]
    min_length, max_length = schema.get("minLength"), schema.get("maxLength")
    if isinstance(min_length, int):
        out.append(("a" * min_length, f"exactly minLength ({min_length})"))
        if min_length > 0:
            out.append(("a" * (min_length - 1), f"minLength-1 ({min_length - 1})"))
    if isinstance(max_length, int):
        out.append(("a" * max_length, f"exactly maxLength ({max_length})"))
        out.append(("a" * (max_length + 1), f"maxLength+1 ({max_length + 1})"))
    else:
        out.append(("a" * LONG_STRING_LENGTH, f"very long ({LONG_STRING_LENGTH} chars)"))
    violation = _FORMAT_VIOLATIONS.get(str(schema.get("format")))
    if violation:
        out.append((violation, f"invalid {schema['format']}"))
    return out


_INT_RANGES = {"int32": (-(2**31), 2**31 - 1), "int64": (-(2**63), 2**63 - 1)}


def _fits_format(schema: dict[str, Any], value: Any) -> bool:
    """Validators ignore int32/int64 ranges; enforce them here."""
    bounds = _INT_RANGES.get(str(schema.get("format")))
    if bounds is None or not isinstance(value, int) or isinstance(value, bool):
        return True
    return bounds[0] <= value <= bounds[1]


def _number_candidates(schema: dict[str, Any], *, integer: bool) -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = [(0, "zero"), (-1, "negative")]
    low, high = schema.get("minimum"), schema.get("maximum")
    if isinstance(low, (int, float)):
        out += [(low, f"minimum ({low})"), (low - 1, f"minimum-1 ({low - 1})")]
    if isinstance(high, (int, float)):
        out += [(high, f"maximum ({high})"), (high + 1, f"maximum+1 ({high + 1})")]
    out += [(2**31, "2^31 (int32 overflow)"), (2**63, "2^63 (int64 overflow)")]
    if integer:
        out.append((1.5, "fraction for an integer"))
    out.append(("1", "number as string"))
    return out
