"""The reviewer and the full review/revision workflow on a real local Ollama.
Opt in with `pytest -m live`."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from agentpanel import AgentContext, Artifact, Workflow
from agentpanel.llm import check_provider, create_chat_model

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.reviewer import ArtifactReviewer
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.config import ShakedownSettings
from api_shakedown.spec import load_spec

pytestmark = pytest.mark.live

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


@pytest.fixture(scope="module")
def settings() -> ShakedownSettings:
    settings = ShakedownSettings()
    status = check_provider(settings)
    if not status.ok:
        pytest.skip(status.detail)
    return settings


def test_reviewer_flags_placeholder_stubs(settings: ShakedownSettings) -> None:
    spec = load_spec(PETSTORE)
    ctx = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
    placeholder_stubs = asyncio.run(StubAgent().produce(ctx))  # bodies like "message": "string"
    artifact = Artifact(author="stubs", kind="WireMock stub mappings", content=placeholder_stubs)
    reviewer = ArtifactReviewer(create_chat_model(settings.for_agent("reviewer")))

    started = time.monotonic()
    verdict = asyncio.run(reviewer.review(artifact, ctx))
    print(f"\nreview took {time.monotonic() - started:.0f}s; approved={verdict.approved}")
    print(*verdict.comments, sep="\n  ")
    assert verdict.comments, "placeholder bodies should draw at least one finding"


def test_full_workflow_with_review(settings: ShakedownSettings) -> None:
    spec = load_spec(PETSTORE)

    def model(agent: str) -> Any:
        return create_chat_model(settings.for_agent(agent))

    class Collect:
        async def assemble(self, ctx: AgentContext) -> Any:
            return dict(ctx.artifacts)

    async def run() -> Any:
        base = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
        dataset = await DatasetAgent(model=model("testdata")).produce(base)
        workflow = Workflow(
            specialists=[
                StubAgent(model=model("stubs")),
                ContractTestAgent(model=model("contracts")),
                ChaosAgent(model=model("chaos")),
            ],
            reviewers=[ArtifactReviewer(model("reviewer"))],
            coordinator=Collect(),
            max_revisions=1,
        )
        return await workflow.run({"spec": spec, "dataset": dataset})

    started = time.monotonic()
    result = asyncio.run(run())
    print(f"\nworkflow took {time.monotonic() - started:.0f}s; rounds={result.rounds} "
          f"unresolved={result.unresolved}")  # fmt: skip
    for review in result.reviews:
        print(f"- round {review.round} {review.target}: approved={review.approved}")
        for comment in review.comments:
            print(f"    {comment}")
    assert set(result.result) == {"stubs", "contracts", "chaos"}
    assert {r.target for r in result.reviews} >= {"stubs", "contracts", "chaos"}
