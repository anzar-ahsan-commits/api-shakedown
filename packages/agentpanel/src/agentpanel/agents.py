"""Agent contracts: specialists, reviewers, and coordinators.

Agents return plain values; the workflow stamps authorship and round numbers.
One object may implement several roles, e.g. a specialist that also peer-reviews.
"""

from typing import Any, Protocol

from agentpanel.state import AgentContext, Artifact, Verdict


class Specialist(Protocol):
    """Produces one artifact per round. On revision rounds, `ctx.feedback_for(self.name)`
    holds the rejections to address and `ctx.artifacts[self.name]` the previous attempt."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    async def produce(self, ctx: AgentContext) -> Any: ...


class Reviewer(Protocol):
    """Judges another agent's artifact. Never asked to review its own."""

    @property
    def name(self) -> str: ...

    async def review(self, artifact: Artifact, ctx: AgentContext) -> Verdict: ...


class Coordinator(Protocol):
    """Assembles the final result once reviews are settled. `ctx.unresolved` lists
    artifacts that still had rejections when the revision budget ran out."""

    async def assemble(self, ctx: AgentContext) -> Any: ...
