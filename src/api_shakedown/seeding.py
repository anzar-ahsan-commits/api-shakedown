"""Seed steps: requests that create the test dataset's records on a real server.

Contract cases use dataset ids (`/collections/products` expects `products` to exist), so
against an empty server they fail with 404. With `SHAKEDOWN_SEED=1` the contract runner
first sends these create requests, then deletes what it created when the tests finish.

The steps are derived by code from the spec and the dataset, never by the LLM: for each
bound resource, a create operation is found (`POST` to the collection with the key in the
body, else `PUT` on the item path), each record is trimmed to the request body schema and
checked against it. Parents are seeded before children; resources that can't be created
this way are listed in the notes instead.
"""

import re
from typing import Any

from api_shakedown.contracts import SeedRequest, SeedStep, auth_requirements
from api_shakedown.spec.edge_cases import properties_of, required_properties
from api_shakedown.spec.models import ApiSpec, Operation
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Dataset, ParamBinding

_PARAM = re.compile(r"\{([^}]+)\}")


def seed_plan(spec: ApiSpec, dataset: Dataset) -> tuple[list[SeedStep], list[str]]:
    """Seed steps for every bound resource that can be created, plus notes on the rest."""
    ops = {(op.method, op.path): op for op in spec.operations}
    steps: list[SeedStep] = []
    notes: list[str] = []
    for binding in sorted(dataset.bindings, key=lambda b: b.path.count("/")):
        entity = dataset.entities[binding.entity]
        if not entity.records:
            continue
        create = _create_operation(spec, binding)
        if create is None:
            notes.append(f"seed: {binding.path} not seeded: {_why_no_create(spec, binding)}.")
            continue
        parents = _parent_values(dataset, binding)
        if parents is None:
            notes.append(f"seed: {binding.path} needs a parent resource that isn't seeded.")
            continue
        delete = ops.get(("delete", binding.path))
        problems: list[str] = []
        planned: list[SeedStep] = []
        for record in entity.records:
            key = record.get(binding.field)
            if key is None:
                continue
            params = parents | {binding.param: key}
            body, errors = _body(spec, create, record)
            if errors:
                problems += errors
                continue
            minimal = _minimal_body(spec, create, body, binding.field)
            planned.append(
                SeedStep(
                    entity=entity.name,
                    record_id=str(key),
                    create=_request(spec, create, params, body),
                    fallback=_request(spec, create, params, minimal) if minimal else None,
                    delete=_request(spec, delete, params, None) if delete else None,
                )
            )
        if problems:
            notes.append(
                f"seed: {entity.name} records don't fit {create.method.upper()} {create.path} "
                f"({problems[0]}); not seeded."
            )
        else:
            steps += planned
    return steps, notes


def _create_operation(spec: ApiSpec, binding: ParamBinding) -> Operation | None:
    """POST to the collection when its body carries the key, else PUT on the item path."""
    collection = binding.path.removesuffix("/{" + binding.param + "}")
    for op in spec.operations:
        if (
            op.method == "post"
            and op.path == collection
            and op.request_body
            and _usable(op)
            and binding.field in properties_of(op.request_body.schema_ or {})
        ):
            return op
    for op in spec.operations:
        if op.method == "put" and op.path == binding.path and _usable(op):
            return op
    return None


def _why_no_create(spec: ApiSpec, binding: ParamBinding) -> str:
    collection = binding.path.removesuffix("/{" + binding.param + "}")
    if any(op.method == "post" and op.path == collection for op in spec.operations):
        return (
            f"POST {collection} doesn't take `{binding.field}`, so the server assigns it "
            "and the test data's ids can't be reproduced"
        )
    return "no create operation found"


def _usable(op: Operation) -> bool:
    """Has a body to send, and no required query or header inputs we'd have to invent."""
    required = [p for p in op.parameters if p.required and p.location in ("query", "header")]
    return op.request_body is not None and not required


def _parent_values(dataset: Dataset, binding: ParamBinding) -> dict[str, Any] | None:
    """Values for the path's other parameters: the first record of each parent resource."""
    values: dict[str, Any] = {}
    for name in _PARAM.findall(binding.path)[:-1]:
        parent = next(
            (
                b
                for b in dataset.bindings
                if b.param == name and binding.path.startswith(b.path + "/")
            ),
            None,
        )
        ids = dataset.entities[parent.entity].ids() if parent else []
        if not ids:
            return None
        values[name] = ids[0]
    return values


def _body(spec: ApiSpec, op: Operation, record: dict[str, Any]) -> tuple[Any, list[str]]:
    """The record trimmed to the request schema (dropping read-only response fields)."""
    assert op.request_body is not None
    schema = op.request_body.schema_
    if not schema:
        return record, []
    properties = properties_of(schema)
    body = {k: v for k, v in record.items() if k in properties} if properties else record
    return body, schema_errors(schema, body, spec.openapi_version, "write")


def _minimal_body(spec: ApiSpec, op: Operation, body: Any, key: str) -> dict[str, Any] | None:
    """Only the required fields (and the key), to retry with when the full body is refused:
    optional fields often point at other records the server doesn't have."""
    schema = op.request_body.schema_ if op.request_body else None
    if not schema or not isinstance(body, dict):
        return None
    keep = required_properties(schema) | {key}
    minimal = {k: v for k, v in body.items() if k in keep}
    if minimal == body or schema_errors(schema, minimal, spec.openapi_version, "write"):
        return None
    return minimal


def _request(spec: ApiSpec, op: Operation, params: dict[str, Any], body: Any) -> SeedRequest:
    names = set(_PARAM.findall(op.path))
    return SeedRequest(
        method=op.method,
        path=op.path,
        path_params={k: v for k, v in params.items() if k in names},
        body=body,
        body_content_type=op.request_body.content_type if op.request_body else None,
        auth=auth_requirements(spec, op),
    )
