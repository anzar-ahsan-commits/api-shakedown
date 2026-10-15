import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from agentpanel import AgentContext
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.chaos import (
    CHAOS_CLI_FILE,
    PROBE_RUNNER_FILE,
    ChaosSuite,
    Effect,
    ScenarioProposal,
    baseline_scenarios,
    compile_scenario,
    scenario_problems,
)
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import CHAOS_TIER, StubSet, base_path, route_priority

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
BASELINE = [
    "total-outage", "slow-responses", "timeouts", "connection-resets",
    "flaky-start", "rate-limited", "truncated-json",
]  # fmt: skip


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


@pytest.fixture(scope="module")
def dataset(spec: ApiSpec) -> Dataset:
    ctx = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
    result: Dataset = asyncio.run(DatasetAgent().produce(ctx))
    return result


def context(spec: ApiSpec, dataset: Dataset | None = None) -> AgentContext:
    inputs: dict[str, Any] = {"spec": spec} | ({"dataset": dataset} if dataset else {})
    return AgentContext(inputs=inputs, artifacts={}, reviews=[], round=0)


def produce(agent: Any, ctx: AgentContext) -> Any:
    return asyncio.run(agent.produce(ctx))


def scenario(**fields: Any) -> ScenarioProposal:
    defaults: dict[str, Any] = {"name": "s", "title": "t", "category": "fault", "expectation": "e"}
    return ScenarioProposal(**(defaults | fields))


def compiled(spec: ApiSpec, dataset: Dataset | None = None, **fields: Any) -> list[dict[str, Any]]:
    return compile_scenario(spec, scenario(**fields), "llm", dataset).mappings


# --- compiling scenarios to WireMock ---------------------------------------------------


def test_baseline_scenarios(spec: ApiSpec) -> None:
    suite = produce(ChaosAgent(), context(spec))
    assert [s.id for s in suite.scenarios] == BASELINE
    assert all(s.source == "baseline" for s in suite.scenarios)
    outage = suite.scenarios[0]
    assert len(outage.mappings) == len(spec.operations)
    ops = {op.id: op for op in spec.operations}
    for m in outage.mappings:
        assert m["priority"] == route_priority(
            ops[m["metadata"]["shakedown"]["operationId"]], CHAOS_TIER
        )
    truncated = suite.scenarios[-1]
    assert {m["metadata"]["shakedown"]["operationId"] for m in truncated.mappings} == {
        "listPets", "createPet", "getPet",
    }  # fmt: skip


def test_fault_and_delay(spec: ApiSpec) -> None:
    (fault,) = compiled(spec, operations=["getPet"], effect=Effect(fault="EMPTY_RESPONSE"))
    assert fault["response"] == {"fault": "EMPTY_RESPONSE"}
    assert fault["request"] == {"method": "GET", "urlPathPattern": "/v1/pets/[0-9]+"}
    assert fault["metadata"]["shakedown"]["scenario"] == "s"

    (slow,) = compiled(spec, operations=["getPet"], effect=Effect(delay_ms=1500))
    assert slow["response"]["fixedDelayMilliseconds"] == 1500
    assert slow["response"]["status"] == 200 and "jsonBody" in slow["response"]


def test_status_uses_documented_response_when_available(spec: ApiSpec) -> None:
    (documented,) = compiled(spec, operations=["getPet"], effect=Effect(status=404))
    assert set(documented["response"]["jsonBody"]) == {"code", "message"}
    (injected,) = compiled(spec, operations=["getPet"], effect=Effect(status=429, retry_after_s=5))
    assert injected["response"]["status"] == 429
    assert injected["response"]["headers"]["Retry-After"] == "5"
    assert "Injected 429" in injected["response"]["jsonBody"]["message"]


def test_times_builds_a_state_machine(spec: ApiSpec) -> None:
    steps = compiled(spec, operations=["createPet"], effect=Effect(status=503), times=3)
    assert [(m["requiredScenarioState"], m["newScenarioState"]) for m in steps] == [
        ("Started", "call-1"), ("call-1", "call-2"), ("call-2", "call-3"),
    ]  # fmt: skip
    assert {m["scenarioName"] for m in steps} == {"chaos-s-createPet"}


def test_malformations(spec: ApiSpec, dataset: Dataset) -> None:
    def body(malformation: str) -> dict[str, Any]:
        (mapping,) = compiled(
            spec, dataset, operations=["getPet"], effect=Effect(malformation=malformation)
        )
        response: dict[str, Any] = mapping["response"]
        return response

    truncated = body("truncated_json")
    with pytest.raises(json.JSONDecodeError):
        json.loads(truncated["body"])
    assert truncated["headers"]["Content-Type"] == "application/json"

    pet_schema = spec.schemas["Pet"]
    wrong = body("wrong_types")["jsonBody"]
    assert schema_errors(pet_schema, wrong, spec.openapi_version)
    assert isinstance(wrong["id"], str)
    missing = body("missing_required")["jsonBody"]
    assert "id" not in missing and "name" not in missing
    assert body("html_error_page")["headers"]["Content-Type"] == "text/html"
    assert body("empty_body")["body"] == ""


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"name": "Bad Name", "effect": Effect(delay_ms=1)}, "snake_case"),
        ({"operations": ["nope"], "effect": Effect(delay_ms=1)}, "unknown operations"),
        ({"effect": Effect()}, "at least one of"),
        ({"effect": Effect(fault="EMPTY_RESPONSE", status=503)}, "fault cannot be combined"),
        ({"effect": Effect(status=200)}, "between 400 and 599"),
        ({"effect": Effect(delay_ms=10**6)}, "delay_ms must be"),
        ({"effect": Effect(status=503), "times": 0}, "times must be"),
        ({"operations": ["deletePet"], "effect": Effect(malformation="wrong_types")},
         "needs JSON success bodies"),
    ],
)  # fmt: skip
def test_scenario_problems(spec: ApiSpec, fields: dict[str, Any], problem: str) -> None:
    assert any(problem in p for p in scenario_problems(spec, scenario(**fields)))


ALL_OPS = ["listPets", "createPet", "getPet", "deletePet", "get_health"]


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"effect": Effect(status=503)}, "list the affected operation ids"),
        ({"category": "partial_outage", "operations": ALL_OPS, "effect": Effect(status=503)},
         "must leave some operations working"),
        ({"category": "flaky", "operations": ["createPet"], "effect": Effect(status=503)},
         "must recover: set times"),
        ({"category": "flaky", "operations": ["createPet"], "effect": Effect(status=400),
          "times": 2}, "needs a 5xx/429 status or a fault"),
        ({"category": "rate_limit", "operations": ["listPets"], "effect": Effect(status=503)},
         "must use status 429"),
        ({"category": "latency", "operations": ["listPets"], "effect": Effect(status=503)},
         "needs delay_ms"),
    ],
)  # fmt: skip
def test_scenarios_must_be_consistent(spec: ApiSpec, fields: dict[str, Any], problem: str) -> None:
    assert any(problem in p for p in scenario_problems(spec, scenario(**fields)))


def test_baseline_scenarios_pass_their_own_rules(spec: ApiSpec) -> None:
    for proposal in baseline_scenarios(spec):
        assert scenario_problems(spec, proposal, targeted=False) == [], proposal.name


def test_consistent_targeted_scenario_has_no_problems(spec: ApiSpec) -> None:
    flaky = scenario(category="flaky", operations=["createPet"], effect=Effect(status=503), times=2)
    assert scenario_problems(spec, flaky) == []


# --- the agent ----------------------------------------------------------------------------


PARTIAL_OUTAGE = {
    "name": "writes_down_reads_up", "title": "Writes fail, reads work",
    "category": "partial_outage", "operations": ["createPet", "deletePet"],
    "effect": {"status": 503}, "expectation": "Queue writes; keep showing cached reads.",
}  # fmt: skip


def test_agent_adds_valid_llm_scenarios(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[json.dumps({"scenarios": [PARTIAL_OUTAGE]})])
    suite = produce(ChaosAgent(model=model), context(spec))
    llm = [s for s in suite.scenarios if s.source == "llm"]
    assert [s.id for s in llm] == ["writes-down-reads-up"]
    assert {m["metadata"]["shakedown"]["operationId"] for m in llm[0].mappings} == {
        "createPet", "deletePet",
    }  # fmt: skip
    prompt = model.calls[0][-1].text
    assert "- total_outage: Every endpoint returns 503" in prompt
    assert "- getPet: GET /pets/{petId} (Get a pet by id) statuses ['200', '404']" in prompt


def test_agent_feeds_back_bad_scenarios_and_drops_them(spec: ApiSpec) -> None:
    bad = [
        PARTIAL_OUTAGE | {"name": "total_outage"},
        PARTIAL_OUTAGE | {"name": "ghost", "operations": ["nope"]},
    ]
    model = ScriptedChatModel(replies=[json.dumps({"scenarios": bad})] * 2)
    suite = produce(ChaosAgent(model=model, max_attempts=2), context(spec))
    retry = model.calls[1][-1].text
    assert "scenario total_outage: duplicate or baseline name." in retry
    assert "scenario ghost: unknown operations ['nope']." in retry
    assert [s.id for s in suite.scenarios] == BASELINE
    assert suite.notes


# --- robustness probes ------------------------------------------------------------------


def test_probes_are_invalid_requests(spec: ApiSpec, dataset: Dataset) -> None:
    suite = produce(ChaosAgent(), context(spec, dataset))
    create = [p for p in suite.probes if p.operation_id == "createPet"]
    assert create and {p.stub_status for p in create} == {"400"}
    request_schema = spec.operation("createPet").request_body.schema_  # type: ignore[union-attr]
    assert request_schema is not None
    for probe in create:
        if probe.raw_body is None:
            assert schema_errors(request_schema, probe.body, spec.openapi_version, "write")
    assert any(p.description == "missing required field name" for p in create)

    listing = [p for p in suite.probes if p.operation_id == "listPets"]
    assert listing and all(p.stub_status is None for p in listing)  # no documented 4xx
    assert not any("as string" in p.description for p in listing)
    assert all(p.auth for p in suite.probes if p.operation_id != "get_health")


def write_suite(suite: ChaosSuite, directory: Path) -> None:
    for name, content in suite.files().items():
        (directory / name).write_text(content, encoding="utf-8")
    (directory / "pytest.ini").write_text("[pytest]\n")


def test_robustness_suite_runs_against_stubs(
    spec: ApiSpec,
    dataset: Dataset,
    tmp_path: Path,
    serve_stubs: Callable[[StubSet], tuple[str, Any]],
) -> None:
    url, _ = serve_stubs(produce(StubAgent(), context(spec, dataset)))
    suite = produce(ChaosAgent(), context(spec, dataset))
    write_suite(suite, tmp_path)
    env = os.environ | {
        "SHAKEDOWN_TARGET_URL": url + base_path(spec),
        "SHAKEDOWN_SELECT_STATUS": "1",
    }
    runner = str(tmp_path / PROBE_RUNNER_FILE)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", runner],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    create = sum(p.operation_id == "createPet" for p in suite.probes)
    assert f"{create} passed" in result.stdout and "skipped" in result.stdout


def test_chaos_cli_lists_scenarios(spec: ApiSpec, tmp_path: Path) -> None:
    write_suite(produce(ChaosAgent(), context(spec)), tmp_path)
    result = subprocess.run(
        [sys.executable, str(tmp_path / CHAOS_CLI_FILE), "list"],
        capture_output=True, text=True, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0
    assert "flaky-start" in result.stdout and "[flaky]" in result.stdout
    unknown = subprocess.run(
        [sys.executable, str(tmp_path / CHAOS_CLI_FILE), "on", "nope"],
        capture_output=True, text=True, cwd=tmp_path,
    )  # fmt: skip
    assert unknown.returncode == 2 and "Unknown scenario 'nope'" in unknown.stderr
