"""Turn a raw OpenAPI document into an `ApiSpec`."""

import re
from collections.abc import Iterable
from typing import Any, cast, get_args

from api_shakedown.spec.models import (
    ApiSpec,
    Body,
    Operation,
    Parameter,
    ParameterLocation,
    RequestBody,
    Response,
)
from api_shakedown.spec.refs import RefResolver

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_LOCATIONS = set(get_args(ParameterLocation))


def normalize(document: dict[str, Any], warnings: Iterable[str] = ()) -> ApiSpec:
    resolver = RefResolver(document)
    info = document.get("info") or {}
    global_security = document.get("security") or []
    notes = list(warnings)
    operations: list[Operation] = []
    used_ids: set[str] = set()

    for path, raw_item in document["paths"].items():
        item = resolver.resolve(raw_item)
        if not isinstance(item, dict):
            continue
        shared_params = item.get("parameters") or []
        for method in HTTP_METHODS:
            raw = item.get(method)
            if not isinstance(raw, dict):
                continue
            op_id = _unique_id(raw.get("operationId") or _derive_id(method, path), used_ids, notes)
            operations.append(
                Operation(
                    id=op_id,
                    method=method,
                    path=path,
                    summary=raw.get("summary"),
                    description=raw.get("description"),
                    tags=[str(t) for t in raw.get("tags") or []],
                    parameters=_parameters([*shared_params, *(raw.get("parameters") or [])], notes),
                    request_body=_request_body(raw.get("requestBody")),
                    responses=_responses(raw.get("responses") or {}),
                    security=_security_names(raw.get("security", global_security)),
                    deprecated=bool(raw.get("deprecated", False)),
                )
            )

    components = document.get("components") or {}
    return ApiSpec(
        title=str(info.get("title") or "Untitled API"),
        version=str(info.get("version") or "unknown"),
        openapi_version=str(document["openapi"]),
        description=info.get("description"),
        servers=[
            str(s["url"])
            for s in document.get("servers") or []
            if isinstance(s, dict) and "url" in s
        ],
        security_schemes=resolver.resolve(components.get("securitySchemes") or {}),
        schemas={
            name: resolved
            for name, schema in (components.get("schemas") or {}).items()
            if isinstance(resolved := resolver.resolve(schema), dict)
        },
        operations=operations,
        warnings=[*notes, *resolver.warnings],
    )


def _derive_id(method: str, path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")
    return f"{method}_{slug or 'root'}"


def _unique_id(candidate: str, used: set[str], notes: list[str]) -> str:
    op_id, n = candidate, 2
    while op_id in used:
        op_id, n = f"{candidate}_{n}", n + 1
    if op_id != candidate:
        notes.append(f"Duplicate operationId {candidate!r}; renamed to {op_id!r}.")
    used.add(op_id)
    return op_id


def _parameters(raw_params: list[Any], notes: list[str]) -> list[Parameter]:
    """Merge path-level and operation-level parameters; the operation's win."""
    merged: dict[tuple[str, str], Parameter] = {}
    for raw in raw_params:
        if not isinstance(raw, dict) or "name" not in raw:
            continue
        location = cast(ParameterLocation, raw.get("in"))
        if location not in _LOCATIONS:
            notes.append(f"Skipped parameter {raw['name']!r} with unknown location {location!r}.")
            continue
        schema = raw.get("schema")
        if schema is None and isinstance(raw.get("content"), dict):
            body = _body(raw["content"])
            schema = body.schema_ if body else None
        merged[(raw["name"], location)] = Parameter(
            name=raw["name"],
            location=location,
            required=location == "path" or bool(raw.get("required", False)),
            description=raw.get("description"),
            schema=schema,
        )
    return list(merged.values())


def _body(content: Any) -> Body | None:
    if not isinstance(content, dict) or not content:
        return None
    types = list(content)
    preferred = next(
        (t for t in types if t.split(";")[0].strip() == "application/json"),
        next((t for t in types if "json" in t), types[0]),
    )
    media = content[preferred] if isinstance(content[preferred], dict) else {}
    example = media.get("example")
    if example is None and isinstance(media.get("examples"), dict) and media["examples"]:
        first = next(iter(media["examples"].values()))
        example = first.get("value") if isinstance(first, dict) else None
    return Body(
        content_type=preferred,
        schema=media.get("schema"),
        example=example,
        other_content_types=[t for t in types if t != preferred],
    )


def _request_body(raw: Any) -> RequestBody | None:
    if not isinstance(raw, dict):
        return None
    body = _body(raw.get("content"))
    if body is None:
        return None
    return RequestBody(**body.model_dump(), required=bool(raw.get("required", False)))


def _responses(raw: dict[str, Any]) -> list[Response]:
    return [
        Response(
            status=str(status),
            description=response.get("description"),
            headers=_headers(response.get("headers")),
            body=_body(response.get("content")),
        )
        for status, response in raw.items()
        if isinstance(response, dict)
    ]


def _headers(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    return {
        name: header.get("schema") or {}
        for name, header in raw.items()
        if isinstance(header, dict) and name.lower() != "content-type"  # ignored per spec
    }


def _security_names(requirements: Any) -> list[str]:
    names: list[str] = []
    for requirement in requirements if isinstance(requirements, list) else []:
        if isinstance(requirement, dict):
            names.extend(name for name in requirement if name not in names)
    return names
