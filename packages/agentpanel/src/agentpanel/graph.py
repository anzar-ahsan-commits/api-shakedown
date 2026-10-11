"""Builds the LangGraph workflow: parallel specialists, cross-review, bounded revision
rounds, then coordinator assembly.

    START -> specialists (parallel) -> gather -> review (one task per reviewer x artifact)
          -> decide -> [rejected specialists revise | coordinator] -> END
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentpanel.agents import Coordinator, Reviewer, Specialist
from agentpanel.state import AgentContext, Artifact, Review, WorkflowState

RESERVED_NAMES = frozenset({"gather", "review", "decide", "coordinator"})
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_STEPS_PER_ROUND = 4  # specialists, gather, review, decide

UpdateCallback = Callable[[str, dict[str, Any]], None]


class _Node(Protocol):
    async def __call__(self, state: WorkflowState) -> dict[str, Any]: ...


class _ReviewTask(TypedDict):
    reviewer: str
    target: str
    context: AgentContext


@dataclass(frozen=True)
class WorkflowResult:
    result: Any
    artifacts: dict[str, Artifact]
    reviews: list[Review]
    rounds: int  # revision rounds used
    unresolved: list[str]


class Workflow:
    def __init__(
        self,
        *,
        specialists: Sequence[Specialist],
        coordinator: Coordinator,
        reviewers: Sequence[Reviewer] = (),
        max_revisions: int = 1,
    ) -> None:
        if not specialists:
            raise ValueError("A workflow needs at least one specialist.")
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0.")
        _validate_names([s.name for s in specialists], "specialist")
        _validate_names([r.name for r in reviewers], "reviewer")

        self._specialists = {s.name: s for s in specialists}
        self._reviewers = {r.name: r for r in reviewers}
        self._coordinator = coordinator
        self.max_revisions = max_revisions
        self.graph = self._build().compile()

    async def run(
        self, inputs: Mapping[str, Any], on_update: UpdateCallback | None = None
    ) -> WorkflowResult:
        """Run to completion. `on_update(node, update)` is called as each node finishes,
        e.g. ("stubs", {"artifacts": ...}) or ("review", {"reviews": [Review]})."""
        initial: WorkflowState = {
            "inputs": dict(inputs),
            "artifacts": {},
            "reviews": [],
            "round": 0,
            "revise": [],
            "result": None,
        }
        config: RunnableConfig = {
            "recursion_limit": _STEPS_PER_ROUND * (self.max_revisions + 1) + 5
        }
        final: Any = None
        async for mode, chunk in self.graph.astream(
            initial, config, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                final = chunk
            elif on_update is not None:
                for node, update in cast(dict[str, Any], chunk).items():
                    on_update(node, update or {})
        final = cast(WorkflowState, final)
        return WorkflowResult(
            result=final["result"],
            artifacts=final["artifacts"],
            reviews=final["reviews"],
            rounds=final["round"],
            unresolved=AgentContext.from_state(final).unresolved,
        )

    def _build(self) -> StateGraph[WorkflowState]:
        graph = StateGraph(WorkflowState)
        for name, specialist in self._specialists.items():
            graph.add_node(name, self._specialist_node(specialist))
            graph.add_edge(START, name)
            graph.add_edge(name, "gather")
        graph.add_node("gather", _noop)
        graph.add_node("review", self._review, input_schema=_ReviewTask)
        graph.add_node("decide", self._decide)
        graph.add_node("coordinator", self._assemble)

        graph.add_conditional_edges("gather", self._route_reviews, ["review", "decide"])
        graph.add_edge("review", "decide")
        graph.add_conditional_edges(
            "decide", _route_after_decide, [*self._specialists, "coordinator"]
        )
        graph.add_edge("coordinator", END)
        return graph

    def _specialist_node(self, specialist: Specialist) -> _Node:
        async def node(state: WorkflowState) -> dict[str, Any]:
            content = await specialist.produce(AgentContext.from_state(state))
            artifact = Artifact(
                author=specialist.name, kind=specialist.kind, content=content, round=state["round"]
            )
            return {"artifacts": {specialist.name: artifact}}

        return node

    def _route_reviews(self, state: WorkflowState) -> list[Send] | str:
        context = AgentContext.from_state(state)
        fresh = [a for a in state["artifacts"].values() if a.round == state["round"]]
        tasks = [
            Send("review", _ReviewTask(reviewer=name, target=artifact.author, context=context))
            for artifact in fresh
            for name in self._reviewers
            if name != artifact.author
        ]
        return tasks or "decide"

    async def _review(self, state: _ReviewTask) -> dict[str, Any]:
        artifact = state["context"].artifacts[state["target"]]
        verdict = await self._reviewers[state["reviewer"]].review(artifact, state["context"])
        review = Review(
            reviewer=state["reviewer"],
            target=artifact.author,
            round=artifact.round,
            approved=verdict.approved,
            comments=verdict.comments,
        )
        return {"reviews": [review]}

    async def _decide(self, state: WorkflowState) -> dict[str, Any]:
        rejected = AgentContext.from_state(state).unresolved
        if rejected and state["round"] < self.max_revisions:
            return {"round": state["round"] + 1, "revise": rejected}
        return {"revise": []}

    async def _assemble(self, state: WorkflowState) -> dict[str, Any]:
        return {"result": await self._coordinator.assemble(AgentContext.from_state(state))}


async def _noop(state: WorkflowState) -> dict[str, Any]:
    return {}


def _route_after_decide(state: WorkflowState) -> list[str] | str:
    return state["revise"] or "coordinator"


def _validate_names(names: list[str], role: str) -> None:
    for name in names:
        if not _NAME_PATTERN.match(name):
            raise ValueError(f"Invalid {role} name {name!r}: use lowercase letters, digits, _.")
        if name in RESERVED_NAMES:
            raise ValueError(f"{role.capitalize()} name {name!r} is reserved.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {role} names: {duplicates}")
