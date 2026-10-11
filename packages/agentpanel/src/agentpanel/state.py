"""Shared graph state: artifacts, reviews, and the read-only context agents receive."""

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    """One specialist's output for one round."""

    model_config = ConfigDict(frozen=True)

    author: str
    kind: str
    content: Any
    round: int = 0


class Verdict(BaseModel):
    """A reviewer's decision about one artifact. Rejections trigger a revision."""

    model_config = ConfigDict(frozen=True)

    approved: bool = Field(description="True if the artifact is acceptable as-is.")
    comments: list[str] = Field(
        default_factory=list, description="Specific, actionable problems or suggestions."
    )


class Review(Verdict):
    reviewer: str
    target: str  # author of the reviewed artifact
    round: int  # round of the reviewed artifact


def _merge(left: dict[str, Artifact], right: dict[str, Artifact]) -> dict[str, Artifact]:
    return {**left, **right}


class WorkflowState(TypedDict):
    inputs: dict[str, Any]
    artifacts: Annotated[dict[str, Artifact], _merge]  # latest artifact per author
    reviews: Annotated[list[Review], operator.add]  # every review from every round
    round: int
    revise: list[str]  # authors scheduled to revise in the current round
    result: Any


@dataclass(frozen=True)
class AgentContext:
    """What an agent can see when it runs."""

    inputs: Mapping[str, Any]
    artifacts: Mapping[str, Artifact]
    reviews: Sequence[Review]
    round: int

    @classmethod
    def from_state(cls, state: WorkflowState) -> "AgentContext":
        return cls(
            inputs=state["inputs"],
            artifacts=state["artifacts"],
            reviews=state["reviews"],
            round=state["round"],
        )

    def reviews_of(self, author: str) -> list[Review]:
        """Reviews of `author`'s latest artifact."""
        artifact = self.artifacts.get(author)
        if artifact is None:
            return []
        return [r for r in self.reviews if r.target == author and r.round == artifact.round]

    def feedback_for(self, author: str) -> list[Review]:
        """Rejections of `author`'s latest artifact, i.e. what a revision must address."""
        return [r for r in self.reviews_of(author) if not r.approved]

    @property
    def unresolved(self) -> list[str]:
        """Authors whose latest artifact still has at least one rejection."""
        return sorted(author for author in self.artifacts if self.feedback_for(author))
