"""Put query parameter values on the wire the way OpenAPI says they travel.

OpenAPI's default for query parameters is `style: form` with `explode: true`, which means
an object parameter does not appear under its own name at all: `{"q": "x", "page": 2}`
for a parameter named `searchParameters` is sent as `?q=x&page=2`. Arrays repeat the
name (`?tag=a&tag=b`), `deepObject` uses `name[key]=value`, and parameters defined with
`content` (e.g. application/json) are sent as one serialized value.
"""

import json
from collections.abc import Mapping
from typing import Any

from api_shakedown.spec.models import Operation, Parameter


def query_pairs(op: Operation, values: Mapping[str, Any]) -> list[list[str]]:
    """Query string pairs for `values` (keyed by parameter name), in a JSON-friendly form."""
    params = {p.name: p for p in op.parameters if p.location == "query"}
    pairs: list[list[str]] = []
    for name, value in values.items():
        param = params.get(name) or Parameter(name=name, location="query")
        pairs += [[key, text] for key, text in serialize(param, value)]
    return pairs


def serialize(param: Parameter, value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    name = param.name
    if param.content_type:
        return [(name, value if isinstance(value, str) else json.dumps(value))]
    style = param.style or "form"
    explode = param.explode if param.explode is not None else style == "form"
    if isinstance(value, dict):
        if style == "deepObject":
            return [(f"{name}[{key}]", _scalar(item)) for key, item in value.items()]
        if explode:
            return [(str(key), _scalar(item)) for key, item in value.items() if item is not None]
        flat = ",".join(f"{key},{_scalar(item)}" for key, item in value.items())
        return [(name, flat)]
    if isinstance(value, list):
        if explode and style == "form":
            return [(name, _scalar(item)) for item in value]
        separator = {"spaceDelimited": " ", "pipeDelimited": "|"}.get(style, ",")
        return [(name, separator.join(_scalar(item) for item in value))]
    return [(name, _scalar(value))]


def appears_by_name(param: Parameter) -> bool:
    """Whether a request carrying this parameter has a query key equal to its name."""
    if param.content_type:
        return True
    style = param.style or "form"
    explode = param.explode if param.explode is not None else style == "form"
    is_object = (param.schema_ or {}).get("type") == "object" or "properties" in (
        param.schema_ or {}
    )
    if style == "deepObject":
        return False
    return not (is_object and style == "form" and explode)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return "" if value is None else str(value)
