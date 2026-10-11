import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from agentpanel import AgentContext, Artifact, Verdict, Workflow
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.reviewer import ArtifactReviewer, describe
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.contracts import CaseProposal, build_suite
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec
from api_shakedown.wiremock import expected_responses

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def context(spec: ApiSpec) -> AgentContext:
    return AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)


def produce(agent: Any, spec: ApiSpec) -> Any:
    return asyncio.run(agent.produce(context(spec)))


def review(reviewer: ArtifactReviewer, content: Any, spec: ApiSpec) -> Verdict:
    artifact = Artifact(author="stubs", kind="WireMock stub mappings", content=content)
    return asyncio.run(reviewer.review(artifact, context(spec)))


def findings(*items: dict[str, str]) -> str:
    return json.dumps({"findings": list(items)})


def finding(severity: str, item: str, problem: str = "p", fix: str = "f") -> dict[str, str]:
    return {"severity": severity, "item": item, "problem": problem, "fix": fix}


# --- rendering ---------------------------------------------------------------------------


def test_describes_each_artifact_kind(spec: ApiSpec) -> None:
    stubs = describe(produce(StubAgent(), spec))
    assert stubs is not None
    kind, rendering, items, _ = stubs
    assert kind == "stubs"
    assert (
        '- getPet 404: GET [only when the client sends X-Shakedown-Status: 404] -> 404 {"code"'
        in rendering
    )
    assert "- getPet 200: GET [normal response] -> 200" in rendering
    assert {"getPet 404", "getPet"} <= items

    contracts = describe(produce(ContractTestAgent(), spec))
    assert contracts is not None and "getPet-200-happy_path" in contracts[2]
    assert "-> expect 200" in contracts[1]

    chaos = describe(produce(ChaosAgent(), spec))
    assert chaos is not None and "flaky-start" in chaos[2]
    assert "- createPet: " in chaos[1] and "probes, e.g." in chaos[1]

    dataset = describe(produce(DatasetAgent(), spec))
    assert dataset is not None and "Pet#2" in dataset[2]

    assert describe("plain text") is None


def test_dataset_stubs_say_when_they_answer(spec: ApiSpec) -> None:
    dataset = produce(DatasetAgent(), spec)
    ctx = AgentContext(inputs={"spec": spec, "dataset": dataset}, artifacts={}, reviews=[], round=0)
    described = describe(asyncio.run(StubAgent().produce(ctx)))
    assert described is not None
    assert (
        "- getPet 200 petId=2: GET [only for petId=2 (a record in the test data)]" in described[1]
    )
    assert "- getPet 404 (unknown id): GET [for ids not in the test data] -> 404" in described[1]


def test_dataset_bodies_are_not_blockable_for_the_stub_author(spec: ApiSpec) -> None:
    dataset = produce(DatasetAgent(), spec)
    ctx = AgentContext(inputs={"spec": spec, "dataset": dataset}, artifacts={}, reviews=[], round=0)
    stubs = asyncio.run(StubAgent().produce(ctx))
    described = describe(stubs)
    assert described is not None
    _, rendering, items, blockable = described
    assert "createPet 201" in items and "createPet 201" not in blockable
    assert "getPet 404" in blockable
    assert "X-Shakedown-Status: default (the spec's 'default' response)" in rendering


def test_prompt_lists_operation_inputs(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[findings()])
    review(ArtifactReviewer(model), produce(ContractTestAgent(), spec), spec)
    prompt = model.calls[0][-1].text
    assert "- listPets: GET /pets (List pets) params: limit (query, " in prompt
    assert "status (query, " in prompt and "no request body;" in prompt
    assert "- createPet: POST /pets (Add a pet) no params; body: application/json (required)" in (
        prompt
    )


def test_long_renderings_are_truncated(spec: ApiSpec) -> None:
    suite = produce(ChaosAgent(), spec)
    suite.scenarios = suite.scenarios * 200
    described = describe(suite)
    assert described is not None
    assert len(described[1]) < 17_000 and "more lines omitted" in described[1]


# --- verdicts ----------------------------------------------------------------------------


def test_no_findings_approves(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[findings()])
    verdict = review(ArtifactReviewer(model), produce(StubAgent(), spec), spec)
    assert verdict == Verdict(approved=True, comments=[])
    system, human = model.calls[0]
    assert "Checklist for WireMock stubs" in system.text
    assert "<artifact>" in human.text and "getPet: GET /pets/{petId}" in human.text


def test_suggestions_approve_blockers_reject(spec: ApiSpec) -> None:
    stubs = produce(StubAgent(), spec)
    suggestion = ArtifactReviewer(
        ScriptedChatModel(replies=[findings(finding("suggestion", "general", "nicer names"))])
    )
    assert review(suggestion, stubs, spec).approved

    blocker = ArtifactReviewer(
        ScriptedChatModel(
            replies=[
                findings(
                    finding("suggestion", "listPets 200", "s"),
                    finding("blocker", "getPet 404", "says code 1", "use code 404"),
                )
            ]
        )
    )
    verdict = review(blocker, stubs, spec)
    assert not verdict.approved
    assert verdict.comments[0] == "[blocker] getPet 404: says code 1 Fix: use code 404"
    assert verdict.comments[1].startswith("[suggestion] listPets 200")


def test_hallucinated_items_are_fed_back_then_dropped(spec: ApiSpec) -> None:
    ghost = findings(
        finding("blocker", "getPet 418"),
        finding("blocker", "general"),
        finding("suggestion", "getPet"),
    )
    model = ScriptedChatModel(replies=[ghost, ghost])
    verdict = review(ArtifactReviewer(model, max_attempts=2), produce(StubAgent(), spec), spec)
    retry = model.calls[1][-1].text
    assert "finding about 'getPet 418': item is not in the artifact" in retry
    assert "a blocker must name a specific item" in retry
    assert verdict.approved  # only the valid suggestion survives
    assert len(verdict.comments) == 1 and verdict.comments[0].startswith("[suggestion] getPet:")


def test_blockers_on_code_generated_items_become_suggestions(spec: ApiSpec) -> None:
    chaos = produce(ChaosAgent(), spec)
    described = describe(chaos)
    assert described is not None
    _, rendering, items, blockable = described
    assert "timeouts" in items and "timeouts" not in blockable
    assert "[timeout, baseline, fixed]" in rendering

    model = ScriptedChatModel(replies=[findings(finding("blocker", "timeouts", "too long"))])
    artifact = Artifact(author="chaos", kind="chaos suite", content=chaos)
    verdict = asyncio.run(ArtifactReviewer(model).review(artifact, context(spec)))
    assert verdict.approved
    assert verdict.comments == ["[suggestion] timeouts: too long Fix: f"]


def test_contract_rendering_explains_openapi_status_keys(spec: ApiSpec) -> None:
    suite = build_suite(
        spec, [CaseProposal(operation_id="listPets", name="x", expected_status="default")]
    )
    described = describe(suite)
    assert described is not None
    assert "expect default (any status the spec does not list explicitly)" in described[1]


def test_review_failure_does_not_block(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=["not json"])
    verdict = review(ArtifactReviewer(model, max_attempts=1), produce(StubAgent(), spec), spec)
    assert verdict.approved and verdict.comments[0].startswith("Review unavailable")


def test_unknown_artifacts_are_approved(spec: ApiSpec) -> None:
    verdict = review(ArtifactReviewer(ScriptedChatModel(replies=[])), "free text", spec)
    assert verdict.approved and "unknown artifact type" in verdict.comments[0]


# --- inside the workflow -------------------------------------------------------------------


def stub_reply(spec: ApiSpec, error_message: str) -> str:
    """A schema-valid stub reply whose error bodies carry `error_message`."""
    pet = {"id": 1, "name": "Rex"}

    def body(op_id: str, status: str, has_body: bool) -> Any:
        if not has_body:
            return None
        if op_id == "get_health":
            return "ok"
        if status.startswith("2"):
            return [pet] if op_id == "listPets" else pet
        return {"code": int(status) if status.isdigit() else 500, "message": error_message}

    entries = [
        {"operation_id": op.id, "status": r.status, "body": body(op.id, r.status, bool(r.body))}
        for op, r in expected_responses(spec.operations)
    ]
    return json.dumps({"responses": entries})


def test_blocker_triggers_a_revision_with_the_fix(spec: ApiSpec) -> None:
    writer = ScriptedChatModel(
        replies=[stub_reply(spec, "error"), stub_reply(spec, "Pet not found")]
    )
    critic = ScriptedChatModel(
        replies=[
            findings(
                finding("blocker", "getPet 404", "message is vague", "say the pet was not found")
            ),
            findings(),
        ]
    )

    class Collect:
        async def assemble(self, ctx: AgentContext) -> Any:
            return ctx.artifacts["stubs"].content

    workflow = Workflow(
        specialists=[StubAgent(model=writer)],
        reviewers=[ArtifactReviewer(critic)],
        coordinator=Collect(),
    )
    result = asyncio.run(workflow.run({"spec": spec}))
    assert result.rounds == 1 and result.unresolved == []
    revision_prompt = writer.calls[1][-1].text
    assert (
        "[blocker] getPet 404: message is vague Fix: say the pet was not found" in revision_prompt
    )
    not_found = next(m for m in result.result.mappings if m["name"] == "getPet 404")
    assert not_found["response"]["jsonBody"]["message"] == "Pet not found"
