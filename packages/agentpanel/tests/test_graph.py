import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from agentpanel.graph import Workflow
from agentpanel.state import AgentContext, Artifact, Verdict


@dataclass
class FakeSpecialist:
    name: str
    kind: str = "text"
    calls: list[AgentContext] = field(default_factory=list)

    async def produce(self, ctx: AgentContext) -> Any:
        self.calls.append(ctx)
        return f"{self.name}-v{ctx.round}"


@dataclass
class FakeReviewer:
    name: str
    approve: Callable[[Artifact], bool] = lambda artifact: True
    seen: list[tuple[str, int]] = field(default_factory=list)

    async def review(self, artifact: Artifact, ctx: AgentContext) -> Verdict:
        self.seen.append((artifact.author, artifact.round))
        ok = self.approve(artifact)
        return Verdict(approved=ok, comments=[] if ok else [f"fix {artifact.content}"])


@dataclass
class PeerSpecialist(FakeSpecialist):
    reviewed: list[str] = field(default_factory=list)

    async def review(self, artifact: Artifact, ctx: AgentContext) -> Verdict:
        self.reviewed.append(artifact.author)
        return Verdict(approved=True)


class JoinCoordinator:
    def __init__(self) -> None:
        self.context: AgentContext | None = None

    async def assemble(self, ctx: AgentContext) -> Any:
        self.context = ctx
        return {name: a.content for name, a in sorted(ctx.artifacts.items())}


def run(workflow: Workflow, inputs: dict[str, Any] | None = None) -> Any:
    return asyncio.run(workflow.run(inputs or {"spec": "petstore"}))


def test_happy_path_reviews_each_artifact_once() -> None:
    a, b = FakeSpecialist("a"), FakeSpecialist("b")
    reviewer = FakeReviewer("critic")
    result = run(Workflow(specialists=[a, b], reviewers=[reviewer], coordinator=JoinCoordinator()))

    assert result.result == {"a": "a-v0", "b": "b-v0"}
    assert sorted(reviewer.seen) == [("a", 0), ("b", 0)]
    assert result.rounds == 0
    assert result.unresolved == []
    assert a.calls[0].inputs == {"spec": "petstore"}


def test_rejected_artifact_is_revised_with_feedback() -> None:
    a, b = FakeSpecialist("a"), FakeSpecialist("b")
    reviewer = FakeReviewer(
        "critic", approve=lambda art: not (art.author == "a" and art.round == 0)
    )
    result = run(Workflow(specialists=[a, b], reviewers=[reviewer], coordinator=JoinCoordinator()))

    assert len(a.calls) == 2 and len(b.calls) == 1  # only the rejected one revises
    revision_ctx = a.calls[1]
    assert revision_ctx.artifacts["a"].content == "a-v0"
    assert [r.comments for r in revision_ctx.feedback_for("a")] == [["fix a-v0"]]
    assert result.result == {"a": "a-v1", "b": "b-v0"}
    assert result.rounds == 1
    assert result.unresolved == []
    assert sorted(reviewer.seen) == [("a", 0), ("a", 1), ("b", 0)]


def test_revision_budget_is_bounded_and_unresolved_reported() -> None:
    a = FakeSpecialist("a")
    coordinator = JoinCoordinator()
    reviewer = FakeReviewer("critic", approve=lambda art: False)
    result = run(
        Workflow(specialists=[a], reviewers=[reviewer], coordinator=coordinator, max_revisions=2)
    )

    assert len(a.calls) == 3
    assert result.rounds == 2
    assert result.unresolved == ["a"]
    assert coordinator.context is not None and coordinator.context.unresolved == ["a"]


def test_zero_revisions_goes_straight_to_coordinator() -> None:
    a = FakeSpecialist("a")
    reviewer = FakeReviewer("critic", approve=lambda art: False)
    result = run(
        Workflow(
            specialists=[a], reviewers=[reviewer], coordinator=JoinCoordinator(), max_revisions=0
        )
    )
    assert len(a.calls) == 1
    assert result.unresolved == ["a"]


def test_peer_review_never_reviews_own_work() -> None:
    a, b, c = PeerSpecialist("a"), PeerSpecialist("b"), PeerSpecialist("c")
    run(Workflow(specialists=[a, b, c], reviewers=[a, b, c], coordinator=JoinCoordinator()))
    assert sorted(a.reviewed) == ["b", "c"]
    assert sorted(b.reviewed) == ["a", "c"]
    assert sorted(c.reviewed) == ["a", "b"]


def test_no_reviewers() -> None:
    result = run(Workflow(specialists=[FakeSpecialist("a")], coordinator=JoinCoordinator()))
    assert result.result == {"a": "a-v0"}
    assert result.reviews == []


def test_specialists_run_concurrently() -> None:
    """Each specialist waits for the other to start, so this deadlocks if run sequentially."""
    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    @dataclass
    class Waiting:
        name: str
        other: str
        kind: str = "text"

        async def produce(self, ctx: AgentContext) -> Any:
            started[self.name].set()
            await asyncio.wait_for(started[self.other].wait(), timeout=2)
            return self.name

    async def go() -> Any:
        for event in started.values():  # bind events to this loop
            event.clear()
        return await Workflow(
            specialists=[Waiting("a", "b"), Waiting("b", "a")], coordinator=JoinCoordinator()
        ).run({})

    assert asyncio.run(go()).result == {"a": "a", "b": "b"}


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["a", "a"], "Duplicate"),
        (["review"], "reserved"),
        (["Bad-Name"], "Invalid"),
    ],
)
def test_invalid_specialist_names(names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Workflow(specialists=[FakeSpecialist(n) for n in names], coordinator=JoinCoordinator())


def test_requires_specialists_and_nonnegative_budget() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Workflow(specialists=[], coordinator=JoinCoordinator())
    with pytest.raises(ValueError, match="max_revisions"):
        Workflow(specialists=[FakeSpecialist("a")], coordinator=JoinCoordinator(), max_revisions=-1)


def test_on_update_reports_each_node() -> None:
    reviewer = FakeReviewer("critic", approve=lambda art: art.round > 0)
    events: list[tuple[str, dict[str, Any]]] = []
    workflow = Workflow(
        specialists=[FakeSpecialist("a")], reviewers=[reviewer], coordinator=JoinCoordinator()
    )
    result = asyncio.run(
        workflow.run({}, on_update=lambda node, update: events.append((node, update)))
    )
    nodes = [node for node, _ in events]
    assert nodes == [
        "a",
        "gather",
        "review",
        "decide",
        "a",
        "gather",
        "review",
        "decide",
        "coordinator",
    ]
    assert events[2][1]["reviews"][0].approved is False
    assert events[3][1] == {"round": 1, "revise": ["a"]}
    assert result.result == {"a": "a-v1"}
