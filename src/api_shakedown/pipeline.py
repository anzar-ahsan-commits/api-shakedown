"""Wires the shakedown agents into an agentpanel workflow.

spec -> test data -> [stubs, contracts, chaos] (parallel) -> review/revise -> harness
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentpanel import AgentContext, Review, Workflow
from agentpanel.llm import create_chat_model
from langchain_core.language_models import BaseChatModel

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.coordinator import Harness, HarnessCoordinator
from api_shakedown.agents.reviewer import ArtifactReviewer
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.config import ShakedownSettings
from api_shakedown.spec import ApiSpec
from api_shakedown.testdata import Dataset

Progress = Callable[[str], None]
ModelFactory = Callable[[str], BaseChatModel | None]


@dataclass
class PipelineOptions:
    offline: bool = False  # no LLM at all: deterministic output
    review: bool = True
    max_revisions: int = 1
    records_per_entity: int = 3
    generated_at: str = ""


async def generate_harness(
    spec: ApiSpec,
    settings: ShakedownSettings | None = None,
    options: PipelineOptions | None = None,
    progress: Progress | None = None,
    model_for: ModelFactory | None = None,
) -> Harness:
    """Run the whole pipeline. `model_for(agent)` overrides how each agent's model is made
    (None = no LLM for that agent); by default models come from `settings`."""
    settings = settings or ShakedownSettings()
    options = options or PipelineOptions()
    say = progress or (lambda message: None)

    def model(agent: str) -> BaseChatModel | None:
        if model_for is not None:
            return model_for(agent)
        return None if options.offline else create_chat_model(settings.for_agent(agent))

    say("Generating test data...")
    base = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
    dataset = await DatasetAgent(
        model=model("testdata"), records_per_entity=options.records_per_entity
    ).produce(base)
    say(f"Test data ready ({_dataset_summary(dataset)}).")

    reviewer = model("reviewer") if options.review else None
    workflow = Workflow(
        specialists=[
            StubAgent(model=model("stubs")),
            ContractTestAgent(model=model("contracts")),
            ChaosAgent(model=model("chaos")),
        ],
        reviewers=[ArtifactReviewer(reviewer)] if reviewer is not None else [],
        coordinator=HarnessCoordinator(generated_at=options.generated_at),
        max_revisions=options.max_revisions,
    )
    say("Generating stubs, contract tests, and chaos scenarios...")
    result = await workflow.run(
        {"spec": spec, "dataset": dataset},
        on_update=lambda node, update: _report(say, node, update),
    )
    harness: Harness = result.result
    return harness


def _dataset_summary(dataset: Dataset) -> str:
    """`3 Pet` for small APIs; a count for big ones, where a list would be a wall of text."""
    entities = list(dataset.entities.values())
    if not entities:
        return "no entities found"
    if len(entities) <= 4:
        return ", ".join(f"{len(e.records)} {e.name}" for e in entities)
    records = sum(len(e.records) for e in entities)
    keyed = sum(1 for e in entities if e.id_field)
    return f"{records} records for {len(entities)} schemas, {keyed} of them keyed resources"


def _report(say: Progress, node: str, update: dict[str, Any]) -> None:
    if node in ("stubs", "contracts", "chaos"):
        artifact = update["artifacts"][node]
        say(f"  {node}: done" + (f" (revision {artifact.round})" if artifact.round else ""))
    elif node == "review":
        review: Review = update["reviews"][0]
        blockers = sum(c.startswith("[blocker]") for c in review.comments)
        verdict = "approved" if review.approved else f"rejected ({blockers} blocker(s))"
        say(f"  review {review.target}: {verdict}")
    elif node == "decide" and update.get("revise"):
        say(f"Revising: {', '.join(update['revise'])}")
