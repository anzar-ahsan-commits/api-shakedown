"""Chaos scenarios and robustness probes.

*Scenarios* inject failures into the WireMock stubs to test how clients of the API cope:
latency, timeouts, connection faults, error bursts, rate limiting, malformed responses,
flaky sequences, and partial outages. They are described in a small constrained format
(`ScenarioProposal`) and compiled to WireMock mappings in code; `chaos.py` in the
generated harness switches them on and off.

*Probes* send malformed requests to test the server's robustness: each one must be
rejected with a 4xx, never a 5xx. They are derived from the spec's schemas alone.
"""

import copy
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from api_shakedown.contracts import SUPPORT_FILE, auth_requirements, happy_proposal, template
from api_shakedown.spec.edge_cases import edge_cases, properties_of, required_properties
from api_shakedown.spec.models import ApiSpec, Operation
from api_shakedown.spec.params import query_pairs
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import (
    CHAOS_TIER,
    DEFAULT_STUB_URL,
    STATUS_HEADER,
    ResponseContent,
    base_path,
    dataset_content,
    happy_response,
    request_matcher,
    response_definition,
    route_priority,
    sample_content,
)

SCENARIOS_FILE = "chaos_scenarios.json"
CHAOS_CLI_FILE = "chaos.py"
PROBES_FILE = "robustness_cases.json"
PROBE_RUNNER_FILE = "test_robustness.py"
MAX_DELAY_MS = 120_000
MAX_TIMES = 20
MAX_PROBES_PER_OPERATION = 15

Category = Literal[
    "latency", "timeout", "fault", "error_burst", "rate_limit",
    "malformed_response", "flaky", "partial_outage", "outage",
]  # fmt: skip
Fault = Literal[
    "CONNECTION_RESET_BY_PEER", "EMPTY_RESPONSE", "MALFORMED_RESPONSE_CHUNK",
    "RANDOM_DATA_THEN_CLOSE",
]  # fmt: skip
Malformation = Literal[
    "truncated_json", "wrong_types", "missing_required", "html_error_page", "empty_body"
]


class Effect(BaseModel):
    delay_ms: int | None = Field(default=None, description="Added latency, 1-120000 ms.")
    fault: Fault | None = Field(default=None, description="Drop or corrupt the connection.")
    status: int | None = Field(default=None, description="Replace the response status, 400-599.")
    malformation: Malformation | None = Field(
        default=None, description="Corrupt the success body while keeping its status."
    )
    retry_after_s: int | None = Field(default=None, description="Retry-After header value.")


class ScenarioProposal(BaseModel):
    name: str = Field(description="snake_case identifier, e.g. writes_down_reads_up")
    title: str
    category: Category
    operations: list[str] = Field(
        default_factory=list, description="operation ids affected; empty means all."
    )
    effect: Effect
    times: int | None = Field(
        default=None, description="Affect only the first N calls per operation, then recover."
    )
    expectation: str = Field(description="What a resilient client should do.")


class ChaosScenario(ScenarioProposal):
    id: str
    source: Literal["baseline", "llm"]
    mappings: list[dict[str, Any]]


class ProbeCase(BaseModel):
    id: str
    operation_id: str
    description: str
    method: str
    path: str
    path_params: dict[str, Any]
    query: dict[str, Any]
    query_pairs: list[list[str]] = Field(default_factory=list)  # as sent on the wire
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None
    raw_body: str | None = None
    body_content_type: str | None = None
    auth: list[Any] = Field(default_factory=list)
    stub_status: str | None = None  # documented 4xx to select when testing against stubs


class ChaosSuite(BaseModel):
    title: str
    version: str
    default_target: str
    scenarios: list[ChaosScenario]
    probes: list[ProbeCase]
    notes: list[str] = Field(default_factory=list)

    def files(self) -> dict[str, str]:
        scenarios = {
            "wiremock_url": DEFAULT_STUB_URL,
            "scenarios": [s.model_dump() for s in self.scenarios],
        }
        probes = {
            "default_target": self.default_target,
            "status_header": STATUS_HEADER,
            "cases": [p.model_dump() for p in self.probes],
        }
        return {
            CHAOS_CLI_FILE: template("chaos_cli.py"),
            SCENARIOS_FILE: json.dumps(scenarios, indent=2, default=str) + "\n",
            PROBE_RUNNER_FILE: template("robustness_runner.py"),
            PROBES_FILE: json.dumps(probes, indent=2, default=str) + "\n",
            SUPPORT_FILE: template(SUPPORT_FILE),
        }


# --- scenarios -----------------------------------------------------------------------


def baseline_scenarios(spec: ApiSpec) -> list[ScenarioProposal]:
    """Generic failures every client should survive; always included."""
    json_ops = [op.id for op in spec.operations if _json_happy(op)]
    scenarios = [
        ScenarioProposal(
            name="total_outage", title="Every endpoint returns 503", category="outage",
            effect=Effect(status=503, retry_after_s=30),
            expectation="Fail fast with a clear error, back off, and do not hammer the API.",
        ),
        ScenarioProposal(
            name="slow_responses", title="Every response takes 3 s", category="latency",
            effect=Effect(delay_ms=3000),
            expectation="Stay responsive: show progress, avoid blocking the UI or worker pool.",
        ),
        ScenarioProposal(
            name="timeouts", title="Every response takes 30 s", category="timeout",
            effect=Effect(delay_ms=30_000),
            expectation="The delay deliberately exceeds any sane client timeout: the client "
            "must give up on its own (e.g. after 5-10 s), surface an error, and free resources.",
        ),
        ScenarioProposal(
            name="connection_resets", title="Connections are reset", category="fault",
            effect=Effect(fault="CONNECTION_RESET_BY_PEER"),
            expectation="Treat it as a transient error; retry idempotent calls with backoff.",
        ),
        ScenarioProposal(
            name="flaky_start", title="First 2 calls fail with 503, then recover",
            category="flaky", effect=Effect(status=503), times=2,
            expectation="Retry with backoff and succeed on the third attempt.",
        ),
        ScenarioProposal(
            name="rate_limited", title="First 3 calls are rate limited (429)",
            category="rate_limit", effect=Effect(status=429, retry_after_s=1), times=3,
            expectation="Honour Retry-After, then succeed without user-visible failure.",
        ),
    ]  # fmt: skip
    if json_ops:
        scenarios.append(
            ScenarioProposal(
                name="truncated_json",
                title="Success bodies are cut off mid-JSON",
                category="malformed_response",
                operations=json_ops,
                effect=Effect(malformation="truncated_json"),
                expectation="Detect the parse failure and report it; never crash or show garbage.",
            )
        )
    return scenarios


def scenario_problems(
    spec: ApiSpec, proposal: ScenarioProposal, targeted: bool = True
) -> list[str]:
    """Problems with a proposal. `targeted` (LLM) scenarios must name their operations."""
    ops = {op.id: op for op in spec.operations}
    effect = proposal.effect
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9_]*", proposal.name):
        problems.append("name must be snake_case.")
    if unknown := sorted(set(proposal.operations) - ops.keys()):
        problems.append(f"unknown operations {unknown}.")
    if not any([effect.delay_ms, effect.fault, effect.status, effect.malformation]):
        problems.append("effect needs at least one of delay_ms, fault, status, malformation.")
    if effect.fault and (effect.status or effect.malformation):
        problems.append("fault cannot be combined with status or malformation.")
    if effect.status and effect.malformation:
        problems.append("use either status or malformation, not both.")
    if effect.status is not None and not 400 <= effect.status <= 599:
        problems.append("status must be between 400 and 599.")
    if effect.delay_ms is not None and not 1 <= effect.delay_ms <= MAX_DELAY_MS:
        problems.append(f"delay_ms must be between 1 and {MAX_DELAY_MS}.")
    if proposal.times is not None and not 1 <= proposal.times <= MAX_TIMES:
        problems.append(f"times must be between 1 and {MAX_TIMES}, or null.")
    if effect.malformation in ("truncated_json", "wrong_types", "missing_required"):
        targets = proposal.operations or list(ops)
        if not_json := [op_id for op_id in targets if op_id in ops and not _json_happy(ops[op_id])]:
            problems.append(f"{effect.malformation} needs JSON success bodies; not {not_json}.")
    return problems + _consistency_problems(spec, proposal, targeted)


def _consistency_problems(spec: ApiSpec, proposal: ScenarioProposal, targeted: bool) -> list[str]:
    """Does the effect actually do what the category (and a sensible engineer) means?"""
    effect, category = proposal.effect, proposal.category
    retryable = effect.status is not None and (effect.status >= 500 or effect.status == 429)
    problems = []
    if targeted and not proposal.operations:
        problems.append(
            "list the affected operation ids; the baseline already covers every operation."
        )
    if category == "partial_outage" and (
        not proposal.operations or set(proposal.operations) >= {op.id for op in spec.operations}
    ):
        problems.append("a partial_outage must leave some operations working.")
    if category == "flaky" and proposal.times is None:
        problems.append("a flaky scenario must recover: set times.")
    if category == "rate_limit" and effect.status != 429:
        problems.append("a rate_limit scenario must use status 429.")
    if category in ("flaky", "outage", "partial_outage") and not (retryable or effect.fault):
        problems.append(f"a {category} scenario needs a 5xx/429 status or a fault.")
    if category in ("latency", "timeout") and not effect.delay_ms:
        problems.append(f"a {category} scenario needs delay_ms.")
    if category == "malformed_response" and not effect.malformation:
        problems.append("a malformed_response scenario needs a malformation.")
    return problems


def compile_scenario(
    spec: ApiSpec,
    proposal: ScenarioProposal,
    source: Literal["baseline", "llm"],
    dataset: Dataset | None,
) -> ChaosScenario:
    prefix = base_path(spec)
    targets = [
        op for op in spec.operations if not proposal.operations or op.id in proposal.operations
    ]
    scenario_id = proposal.name.replace("_", "-")
    mappings = []
    for op in targets:
        response = _chaos_response(op, proposal.effect, dataset, spec.openapi_version)
        if response is None:
            continue
        base: dict[str, Any] = {
            "name": f"chaos {scenario_id} {op.id}",
            "priority": route_priority(op, CHAOS_TIER),
            "request": request_matcher(op, prefix),
            "response": response,
            "metadata": {"shakedown": {"scenario": scenario_id, "operationId": op.id}},
        }
        if proposal.times is None:
            mappings.append(base)
            continue
        for i in range(proposal.times):  # a WireMock state machine: fail N times, recover
            step = copy.deepcopy(base)
            step["name"] += f" call {i + 1}"
            step["scenarioName"] = f"chaos-{scenario_id}-{op.id}"
            step["requiredScenarioState"] = "Started" if i == 0 else f"call-{i}"
            step["newScenarioState"] = f"call-{i + 1}"
            mappings.append(step)
    return ChaosScenario(**proposal.model_dump(), id=scenario_id, source=source, mappings=mappings)


def _chaos_response(
    op: Operation, effect: Effect, dataset: Dataset | None, version: str
) -> dict[str, Any] | None:
    happy = happy_response(op)
    if happy is None:
        return None
    content = dataset_content(happy, dataset, creating=op.method == "post")
    content = content or sample_content(happy, version)
    if effect.fault:
        response: dict[str, Any] = {"fault": effect.fault}
    elif effect.status:
        response = _status_response(op, effect.status, version)
    elif effect.malformation:
        response = _malformed_response(op, happy.status, content, effect.malformation)
    else:
        response = response_definition(happy, content)
    if effect.retry_after_s is not None and "fault" not in response:
        response.setdefault("headers", {})["Retry-After"] = str(effect.retry_after_s)
    if effect.delay_ms:
        response["fixedDelayMilliseconds"] = effect.delay_ms
    return response


def _status_response(op: Operation, status: int, version: str) -> dict[str, Any]:
    documented = next((r for r in op.responses if r.status == str(status)), None)
    if documented is not None:
        return response_definition(documented, sample_content(documented, version))
    return {
        "status": status,
        "headers": {"Content-Type": "application/json"},
        "jsonBody": {"code": status, "message": f"Injected {status} by API Shakedown"},
    }


def _malformed_response(
    op: Operation, status: str, content: ResponseContent, malformation: Malformation
) -> dict[str, Any]:
    happy = happy_response(op)
    schema = happy.body.schema_ if happy and happy.body else None
    json_type = happy.body.content_type if happy and happy.body else "application/json"
    code = int(status) if status.isdigit() else 200
    if malformation == "html_error_page":
        html = "<html><body><h1>502 Bad Gateway</h1><p>nginx</p></body></html>"
        return {"status": code, "headers": {"Content-Type": "text/html"}, "body": html}
    if malformation == "empty_body":
        return {"status": code, "headers": {"Content-Type": json_type}, "body": ""}
    if malformation == "truncated_json":
        text = json.dumps(content.body)
        return {
            "status": code,
            "headers": {"Content-Type": json_type},
            "body": text[: len(text) // 2],
        }
    body = copy.deepcopy(content.body)
    target = body[0] if isinstance(body, list) and body else body
    item_schema = (schema or {}).get("items", schema) if isinstance(body, list) else schema
    if isinstance(target, dict) and isinstance(item_schema, dict):
        if malformation == "missing_required":
            for name in required_properties(item_schema):
                target.pop(name, None)
        else:  # wrong_types
            for name, value in list(target.items()):
                target[name] = _wrong_type(value)
    return {"status": code, "headers": {"Content-Type": json_type}, "jsonBody": body}


def _wrong_type(value: Any) -> Any:
    if isinstance(value, bool):
        return "yes"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return 12345
    if isinstance(value, list):
        return {"unexpected": "object"}
    if isinstance(value, dict):
        return ["unexpected", "array"]
    return "unexpected"


def _json_happy(op: Operation) -> bool:
    happy = happy_response(op)
    return bool(happy and happy.body and "json" in happy.body.content_type)


# --- probes ---------------------------------------------------------------------------


def build_probes(spec: ApiSpec, dataset: Dataset | None) -> list[ProbeCase]:
    probes: list[ProbeCase] = []
    for op in spec.operations:
        base = happy_proposal(op, dataset, spec.openapi_version)
        if base is None:
            continue
        stub_status = next(
            (r.status for r in op.responses if r.status in ("400", "422", "4XX")),
            next((r.status for r in op.responses if r.status.startswith("4")), None),
        )
        variants = _body_variants(spec, op, base.body) + _query_variants(spec, op)
        for i, (description, change) in enumerate(variants[:MAX_PROBES_PER_OPERATION]):
            fields: dict[str, Any] = {
                "body": base.body,
                "query": dict(base.query),
                "body_content_type": op.request_body.content_type if op.request_body else None,
            }
            fields.update(change)
            fields["query"] = dict(base.query) | change.get("query", {})
            fields["query_pairs"] = query_pairs(op, fields["query"])
            probes.append(
                ProbeCase(
                    id=f"{op.id}-{i + 1:02d}-{_slug(description)}",
                    operation_id=op.id,
                    description=description,
                    method=op.method,
                    path=op.path,
                    path_params=base.path_params,
                    auth=[a.model_dump() for a in auth_requirements(spec, op)],
                    stub_status=stub_status,
                    **fields,
                )
            )
    return probes


def _body_variants(spec: ApiSpec, op: Operation, base: Any) -> list[tuple[str, dict[str, Any]]]:
    if op.request_body is None or not isinstance(base, dict) or op.request_body.schema_ is None:
        return []
    schema = op.request_body.schema_
    variants: list[tuple[str, dict[str, Any]]] = [
        ("syntactically invalid JSON", {"body": None, "raw_body": '{"name": '}),
        (
            "wrong content type",
            {"body": None, "raw_body": "hello", "body_content_type": "text/plain"},
        ),
        ("empty JSON object", {"body": {}}),
        ("JSON array instead of object", {"body": [base]}),
    ]
    for name in sorted(required_properties(schema)):
        variants.append((f"missing required field {name}", {"body": _without(base, name)}))
    for name, prop in properties_of(schema).items():
        if not isinstance(prop, dict):
            continue
        for case in edge_cases(prop, spec.openapi_version):
            if not case.valid:
                variants.append(
                    (f"{name}: {case.description}", {"body": base | {name: case.value}})
                )
    return [
        v
        for v in variants
        if v[1].get("raw_body") is not None or _invalid(spec, schema, v[1]["body"])
    ]


def _query_variants(spec: ApiSpec, op: Operation) -> list[tuple[str, dict[str, Any]]]:
    variants = []
    for param in (p for p in op.parameters if p.location == "query" and p.schema_):
        for case in edge_cases(param.schema_ or {}, spec.openapi_version):
            # Everything in a query string is a string, so "1" and 1 are the same request.
            if not case.valid and case.value is not None and "as string" not in case.description:
                variants.append(
                    (f"query {param.name}: {case.description}", {"query": {param.name: case.value}})
                )
    return variants


def _invalid(spec: ApiSpec, schema: dict[str, Any], body: Any) -> bool:
    return bool(schema_errors(schema, body, spec.openapi_version, "write"))


def _without(body: dict[str, Any], name: str) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k != name}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]
