import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from agentpanel import AgentContext, Artifact, Review
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.contracts import (
    CASES_FILE,
    RUNNER_FILE,
    CaseProposal,
    ContractSuite,
    build_suite,
    proposal_problems,
)
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec
from api_shakedown.wiremock import STATUS_HEADER, StubSet, base_path

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
NEW_PET = {"name": "Biscuit", "status": "available"}


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def context(spec: ApiSpec, **overrides: Any) -> AgentContext:
    base: dict[str, Any] = {"inputs": {"spec": spec}, "artifacts": {}, "reviews": [], "round": 0}
    return AgentContext(**(base | overrides))


def produce(agent: Any, ctx: AgentContext) -> Any:
    return asyncio.run(agent.produce(ctx))


def proposal(**fields: Any) -> CaseProposal:
    return CaseProposal(**({"name": "case", "expected_status": "200"} | fields))


GOOD_CASES = [
    {"operation_id": "listPets", "name": "lists pets", "expected_status": "200",
     "query": {"limit": 1}},
    {"operation_id": "listPets", "name": "max limit", "expected_status": "200",
     "query": {"limit": 100, "status": "sold"}},
    {"operation_id": "createPet", "name": "creates pet", "expected_status": "201",
     "body": NEW_PET},
    {"operation_id": "createPet", "name": "rejects missing name", "expected_status": "400",
     "body": {"status": "available"}},
    {"operation_id": "getPet", "name": "gets pet", "expected_status": "200",
     "path_params": {"petId": 7}},
    {"operation_id": "getPet", "name": "unknown id", "expected_status": "404",
     "path_params": {"petId": 999999}},
    {"operation_id": "deletePet", "name": "deletes pet", "expected_status": "204",
     "path_params": {"petId": 7}},
    {"operation_id": "get_health", "name": "is up", "expected_status": "200"},
]  # fmt: skip


def reply(cases: list[dict[str, Any]]) -> str:
    return json.dumps({"cases": cases})


def run_suite(
    suite: ContractSuite, directory: Path, target: str
) -> subprocess.CompletedProcess[str]:
    for name, content in suite.files().items():
        (directory / name).write_text(content, encoding="utf-8")
    (directory / "pytest.ini").write_text("[pytest]\n")  # isolate from this repo's config
    env = os.environ | {"SHAKEDOWN_TARGET_URL": target, "SHAKEDOWN_SELECT_STATUS": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(directory)],
        capture_output=True, text=True, env=env, cwd=directory,
    )  # fmt: skip


# --- proposal checks ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"operation_id": "getPet", "expected_status": "418", "path_params": {"petId": 1}},
         "expected_status must be one of ['200', '404']"),
        ({"operation_id": "getPet"}, "missing path_params ['petId']"),
        ({"operation_id": "getPet", "path_params": {"petId": 1, "x": 2}}, "unknown path_params"),
        ({"operation_id": "getPet", "expected_status": "404", "path_params": {"petId": "abc"}},
         "must be valid even in error cases"),
        ({"operation_id": "listPets", "query": {"limit": 500}}, "query limit=500"),
        ({"operation_id": "listPets", "query": {"page": 2}}, "unknown query params ['page']"),
        ({"operation_id": "createPet", "expected_status": "201"}, "needs a request body"),
        ({"operation_id": "listPets", "expected_status": "default", "body": {"x": 1}},
         "GET /pets takes no request body"),
        ({"operation_id": "createPet", "expected_status": "201", "body": {"status": "sold"}},
         "request body: (root): 'name' is a required property"),
    ],
)  # fmt: skip
def test_proposal_problems(spec: ApiSpec, fields: dict[str, Any], problem: str) -> None:
    op = spec.operation(fields["operation_id"])
    problems = proposal_problems(spec, op, proposal(**fields))
    assert any(problem in p for p in problems), problems


def test_error_cases_may_send_invalid_bodies_and_queries(spec: ApiSpec) -> None:
    bad = proposal(operation_id="createPet", expected_status="400", body={"name": 5})
    assert proposal_problems(spec, spec.operation("createPet"), bad) == []


def test_cases_carry_auth_and_response_contract(spec: ApiSpec) -> None:
    suite = build_suite(
        spec,
        [proposal(operation_id="getPet", path_params={"petId": 1},
                  headers={"X-API-Key": "leak", "Authorization": "x", "X-Trace": "t"})],
    )  # fmt: skip
    (case,) = suite.cases
    assert [(a.scheme, a.location, a.name) for a in case.auth] == [
        ("apiKey", "header", "X-API-Key")
    ]
    assert case.headers == {"X-Trace": "t"}  # auth-like headers are stripped
    assert case.other_statuses == ["404"]
    assert case.response_content_type == "application/json"
    assert case.response_schema is not None and "allOf" in case.response_schema
    assert suite.default_target == "http://localhost:8080/v1"


def test_case_ids_are_unique_and_readable(spec: ApiSpec) -> None:
    twice = [proposal(operation_id="get_health", name="Is it up?")] * 2
    ids = [c.id for c in build_suite(spec, twice).cases]
    assert ids == ["get_health-200-is_it_up", "get_health-200-is_it_up_2"]


# --- the agent ----------------------------------------------------------------------


def test_offline_agent_gives_one_happy_case_per_operation(spec: ApiSpec) -> None:
    suite = produce(ContractTestAgent(), context(spec))
    assert [c.id for c in suite.cases] == [
        "listPets-200-happy_path", "createPet-201-happy_path", "getPet-200-happy_path",
        "deletePet-204-happy_path", "get_health-200-happy_path",
    ]  # fmt: skip
    assert suite.notes == []


def test_agent_uses_valid_llm_cases(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[reply(GOOD_CASES)])
    suite = produce(ContractTestAgent(model=model), context(spec))
    assert len(suite.cases) == len(GOOD_CASES)
    assert suite.notes == []
    prompt = model.calls[0][-1].text
    assert "- getPet: documented statuses ['200', '404']" in prompt


def test_agent_feeds_back_problems_then_drops_bad_cases(spec: ApiSpec) -> None:
    bad = [*GOOD_CASES[:4], {"operation_id": "getPet", "name": "bad", "expected_status": "200",
                             "path_params": {"petId": "seven"}}]  # fmt: skip
    model = ScriptedChatModel(replies=[reply(bad), reply(bad)])
    suite = produce(ContractTestAgent(model=model, max_attempts=2), context(spec))

    retry = model.calls[1][-1].text
    assert "case getPet/bad: path param petId='seven'" in retry
    assert "No valid happy-path (2xx) case for deletePet." in retry
    ids = [c.id for c in suite.cases]
    assert "getPet-200-bad" not in ids
    assert "getPet-200-happy_path" in ids  # fallback filled the gap
    assert "deletePet: used a schema-generated happy-path case." in suite.notes


def test_cases_per_operation_are_capped(spec: ApiSpec) -> None:
    many = [GOOD_CASES[-1] | {"name": f"up_{i}"} for i in range(20)] + GOOD_CASES[:-1]
    suite = produce(
        ContractTestAgent(model=ScriptedChatModel(replies=[reply(many)])), context(spec)
    )
    assert sum(c.operation_id == "get_health" for c in suite.cases) == 8


def test_revision_includes_feedback(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[reply(GOOD_CASES)])
    rejection = Review(
        reviewer="critic", target="contracts", round=0, approved=False, comments=["Test limit=0"]
    )
    previous = Artifact(author="contracts", kind="tests", content="v0")
    produce(
        ContractTestAgent(model=model),
        context(spec, round=1, reviews=[rejection], artifacts={"contracts": previous}),
    )
    assert "- (critic) Test limit=0" in model.calls[0][-1].text


# --- end to end: generated tests against generated stubs ---------------------------------


def test_generated_suite_passes_against_generated_stubs(
    spec: ApiSpec,
    tmp_path: Path,
    serve_stubs: Callable[[StubSet], tuple[str, Any]],
) -> None:
    stubs = produce(StubAgent(), context(spec))
    url, handler = serve_stubs(stubs)
    suite = produce(ContractTestAgent(model=ScriptedChatModel(replies=[reply(GOOD_CASES)])),
                    context(spec))  # fmt: skip

    result = run_suite(suite, tmp_path, url + base_path(spec))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(GOOD_CASES)} passed" in result.stdout

    sent = {(method, path.split("?")[0]): headers for method, path, headers in handler.received}
    get_headers = sent[("GET", "/v1/pets/999999")]
    assert get_headers["X-API-Key"] == "test-key"
    assert get_headers[STATUS_HEADER] == "404"


def test_generated_suite_reports_contract_violations(
    spec: ApiSpec,
    tmp_path: Path,
    serve_stubs: Callable[[StubSet], tuple[str, Any]],
) -> None:
    stubs = produce(StubAgent(), context(spec))
    for mapping in stubs.mappings:
        if mapping["name"] == "getPet 200":
            mapping["response"]["jsonBody"] = {"name": "no id"}  # breaks the contract
    url, _ = serve_stubs(stubs)
    suite = produce(ContractTestAgent(), context(spec))

    result = run_suite(suite, tmp_path, url + base_path(spec))
    assert result.returncode == 1
    assert "1 failed, 4 passed" in result.stdout
    assert "'id' is a required property" in result.stdout


def test_suite_files(spec: ApiSpec) -> None:
    files = produce(ContractTestAgent(), context(spec)).files()
    assert set(files) == {RUNNER_FILE, CASES_FILE, "shakedown_support.py"}
    compile(files[RUNNER_FILE], RUNNER_FILE, "exec")
    assert json.loads(files[CASES_FILE])["status_header"] == STATUS_HEADER
