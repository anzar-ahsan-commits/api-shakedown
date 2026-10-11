"""Build WireMock stub mappings from a normalized spec.

Request matching is derived from the spec in code: LLMs only supply response content.

- The happy-path response of each operation (lowest 2xx) matches on method + path.
- Every other documented response is selected by sending `X-Shakedown-Status: <status>`,
  e.g. `X-Shakedown-Status: 404`, so tests and chaos scenarios can trigger it on demand.
- With a test dataset, known ids return their own record and unknown ids return 404.
"""

import json
import re
from typing import Any, Literal
from urllib.parse import quote, urlparse

from pydantic import BaseModel, Field

from api_shakedown.spec.models import ApiSpec, Operation, Response
from api_shakedown.spec.samples import sample_from_schema
from api_shakedown.testdata import Dataset, Entity

STATUS_HEADER = "X-Shakedown-Status"
DEFAULT_STUB_URL = "http://localhost:8080"  # WireMock's default port
# WireMock: lower number wins. Chaos overrides everything while a scenario is active.
CHAOS_PRIORITY = 1
SELECTED_PRIORITY = 2
RECORD_PRIORITY = 4
HAPPY_PRIORITY = 5

ResponseKey = tuple[str, str]  # (operation id, status label)


class ResponseContent(BaseModel):
    body: Any = None
    headers: dict[str, str] = Field(default_factory=dict)
    source: Literal["llm", "dataset", "sample"] = "sample"  # who wrote the body


class StubSet(BaseModel):
    mappings: list[dict[str, Any]]
    notes: list[str] = Field(default_factory=list)  # e.g. where samples replaced LLM output
    status_header: str = STATUS_HEADER

    def to_wiremock(self) -> dict[str, Any]:
        """The JSON document WireMock loads from `mappings/*.json`."""
        return {"mappings": self.mappings}


def expected_responses(operations: list[Operation]) -> list[tuple[Operation, Response]]:
    return [(op, response) for op in operations for response in op.responses]


def happy_response(op: Operation) -> Response | None:
    success = sorted((r for r in op.responses if r.status.startswith("2")), key=lambda r: r.status)
    return success[0] if success else (op.responses[0] if op.responses else None)


def sample_content(response: Response) -> ResponseContent:
    """Schema-derived content for a response with no usable LLM output."""
    body = None
    if response.body is not None:
        body = response.body.example
        if body is None:
            body = sample_from_schema(response.body.schema_)
    headers = {name: _header_value(schema) for name, schema in response.headers.items()}
    return ResponseContent(body=body, headers=headers)


def dataset_content(
    response: Response, dataset: Dataset | None, creating: bool = False
) -> ResponseContent | None:
    """Dataset records for a success response whose body is an entity (or a list of one).

    A create call (`creating`) returns a new record rather than an existing one.
    """
    if dataset is None or response.body is None or not response.status.startswith("2"):
        return None
    match = dataset.entity_for(response.body.schema_)
    if match is None or not match[0].records:
        return None
    entity, is_array = match
    if is_array:
        body: Any = entity.records
    else:
        body = entity.new_record() if creating else entity.records[0]
    headers = sample_content(response).headers
    return ResponseContent(body=body, headers=headers, source="dataset")


def build_stub_set(
    spec: ApiSpec,
    contents: dict[ResponseKey, ResponseContent],
    notes: list[str] | None = None,
    dataset: Dataset | None = None,
) -> StubSet:
    """One mapping per documented response; missing contents are filled with samples.

    With a dataset, entity responses use its records, every known id of a bound path
    parameter gets its own mapping returning that record, and unknown ids get the
    operation's documented 404 instead of the happy response.
    """
    prefix = base_path(spec)
    mappings = []
    for op in spec.operations:
        happy = happy_response(op)

        def content_for(response: Response, op: Operation = op) -> ResponseContent:
            return (
                dataset_content(response, dataset, creating=op.method == "post")
                or contents.get((op.id, response.status))
                or sample_content(response)
            )

        for response in op.responses:
            if response is not happy:
                mappings.append(
                    _mapping(prefix, op, response, content_for(response), SELECTED_PRIORITY)
                )
        if happy is None:
            continue

        bound = _bound_param(op, dataset)
        not_found = next((r for r in op.responses if r.status == "404"), None)
        if bound is None or dataset is None:
            mappings.append(_mapping(prefix, op, happy, content_for(happy), HAPPY_PRIORITY))
            continue
        param, entity = bound
        for record_id in entity.ids():
            content = content_for(happy)
            if dataset.entity_for(happy.body.schema_ if happy.body else None) == (entity, False):
                content = content.model_copy(update={"body": entity.record(record_id)})
            mappings.append(
                _mapping(prefix, op, happy, content, RECORD_PRIORITY, pinned={param: record_id})
            )
        fallback = not_found or happy
        mappings.append(_mapping(prefix, op, fallback, content_for(fallback), HAPPY_PRIORITY))
    return StubSet(mappings=mappings, notes=list(notes or []))


def _bound_param(op: Operation, dataset: Dataset | None) -> tuple[str, Entity] | None:
    if dataset is None:
        return None
    for param in (p for p in op.parameters if p.location == "path"):
        binding = dataset.binding(op.path, param.name)
        if binding is not None and dataset.entities[binding.entity].ids():
            return param.name, dataset.entities[binding.entity]
    return None


def base_path(spec: ApiSpec) -> str:
    """Path prefix from the first server URL, e.g. `/v1`. Templated URLs are ignored."""
    if not spec.servers or "{" in spec.servers[0]:
        return ""
    return urlparse(spec.servers[0]).path.rstrip("/")


def path_regex(op: Operation, prefix: str = "", pinned: dict[str, Any] | None = None) -> str:
    """Java-compatible regex for `op.path`, with path parameters constrained by type.

    `pinned` parameters match one exact value instead.
    """
    schemas = {p.name: p.schema_ for p in op.parameters if p.location == "path"}
    pinned = pinned or {}

    def piece(part: str) -> str:
        if not part.startswith("{"):
            return re.escape(part)
        name = part[1:-1]
        if name in pinned:
            return re.escape(quote(str(pinned[name]), safe=""))
        return _param_regex(schemas.get(name))

    return "".join(piece(part) for part in re.split(r"(\{[^}]+\})", prefix + op.path))


def status_code(label: str) -> int:
    """`404` -> 404, `4XX` -> 400, `default` -> 500."""
    if label.isdigit():
        return int(label)
    if len(label) == 3 and label[0].isdigit():
        return int(label[0]) * 100
    return 500


def request_matcher(
    op: Operation, prefix: str, pinned: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Method + path (+ required query params) for an operation."""
    request: dict[str, Any] = {
        "method": op.method.upper(),
        "urlPathPattern": path_regex(op, prefix, pinned),
    }
    required_query = [p.name for p in op.parameters if p.location == "query" and p.required]
    if required_query:
        request["queryParameters"] = {name: {"matches": ".+"} for name in required_query}
    return request


def response_definition(response: Response, content: ResponseContent) -> dict[str, Any]:
    """WireMock `response` block for a documented response."""
    reply: dict[str, Any] = {"status": status_code(response.status)}
    headers: dict[str, str] = {}
    if response.body is not None and content.body is not None:
        headers["Content-Type"] = response.body.content_type
        if "json" in response.body.content_type:
            reply["jsonBody"] = content.body
        elif isinstance(content.body, str):
            reply["body"] = content.body
        else:
            reply["body"] = json.dumps(content.body)
    headers.update({k: v for k, v in content.headers.items() if k in response.headers})
    if headers:
        reply["headers"] = headers
    return reply


def _mapping(
    prefix: str,
    op: Operation,
    response: Response,
    content: ResponseContent,
    priority: int,
    pinned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = priority == SELECTED_PRIORITY
    request = request_matcher(op, prefix, pinned)
    if selected:
        request["headers"] = {STATUS_HEADER: {"equalTo": response.status}}
    name = f"{op.id} {response.status}"
    if pinned:
        name += " " + " ".join(f"{k}={v}" for k, v in pinned.items())
    elif priority == HAPPY_PRIORITY and response.status == "404":
        name += " (unknown id)"
    return {
        "name": name,
        "priority": priority,
        "request": request,
        "response": response_definition(response, content),
        "metadata": {
            "shakedown": {
                "operationId": op.id,
                "status": response.status,
                "selectedBy": {STATUS_HEADER: response.status} if selected else None,
                "source": content.source,
            }
        },
    }


def _param_regex(schema: dict[str, Any] | None) -> str:
    if not schema:
        return "[^/]+"
    if schema.get("enum"):
        return "(?:" + "|".join(re.escape(str(v)) for v in schema["enum"]) + ")"
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    non_negative = isinstance(schema.get("minimum"), (int, float)) and schema["minimum"] >= 0
    sign = "" if non_negative else "-?"
    if kind == "integer":
        return f"{sign}[0-9]+"
    if kind == "number":
        return f"{sign}[0-9]+(?:\\.[0-9]+)?"
    if kind == "boolean":
        return "(?:true|false)"
    if schema.get("format") == "uuid":
        return "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    return "[^/]+"


def _header_value(schema: dict[str, Any]) -> str:
    value = sample_from_schema(schema)
    return "" if value is None else str(value)
