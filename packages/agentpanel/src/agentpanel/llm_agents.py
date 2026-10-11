"""Reusable LLM-backed agents. Applications supply instructions and prompt renderers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentpanel.state import AgentContext, Artifact, Review, Verdict
from agentpanel.structured import ainvoke_structured, format_content, message_text

REVIEW_GUIDANCE = (
    "Approve only if the artifact is correct and fit for its purpose. When rejecting, give "
    "specific, actionable comments: the author revises using only your comments."
)


def render_artifact(artifact: Artifact, ctx: AgentContext) -> str:
    """Default reviewer prompt: just the artifact."""
    return (
        f"Review this {artifact.kind} produced by '{artifact.author}':\n\n"
        f"<artifact>\n{format_content(artifact.content)}\n</artifact>"
    )


def render_feedback(feedback: list[Review]) -> str:
    """Reviewer comments as a bullet list, attributed to each reviewer."""
    return "\n".join(
        f"- ({review.reviewer}) {comment}"
        for review in feedback
        for comment in (review.comments or ["Rejected without comments."])
    )


def render_revision(previous: Artifact, feedback: list[Review]) -> str:
    return (
        "Your previous attempt was rejected.\n\n"
        f"<previous_attempt>\n{format_content(previous.content)}\n</previous_attempt>\n\n"
        f"Reviewer feedback to address:\n{render_feedback(feedback)}\n\n"
        "Produce a complete, corrected replacement, not a diff."
    )


@dataclass
class LLMSpecialist:
    """Produces text, or an `output_schema` instance when one is given.

    On revision rounds the previous attempt and reviewer feedback are appended to the
    prompt automatically, so `render` only needs to describe the task.
    """

    name: str
    kind: str
    model: BaseChatModel
    instructions: str
    render: Callable[[AgentContext], str]
    output_schema: type[BaseModel] | None = None
    check: Callable[[Any], list[str]] | None = None  # domain checks on structured output
    max_attempts: int = 3

    async def produce(self, ctx: AgentContext) -> Any:
        messages = [SystemMessage(self.instructions), HumanMessage(self._prompt(ctx))]
        if self.output_schema is None:
            return message_text(await self.model.ainvoke(messages))
        return await ainvoke_structured(
            self.model,
            messages,
            self.output_schema,
            max_attempts=self.max_attempts,
            check=self.check,
        )

    def _prompt(self, ctx: AgentContext) -> str:
        prompt = self.render(ctx)
        feedback = ctx.feedback_for(self.name)
        if not feedback:
            return prompt
        return f"{prompt}\n\n{render_revision(ctx.artifacts[self.name], feedback)}"


@dataclass
class LLMReviewer:
    name: str
    model: BaseChatModel
    instructions: str
    render: Callable[[Artifact, AgentContext], str] = render_artifact
    max_attempts: int = 3

    async def review(self, artifact: Artifact, ctx: AgentContext) -> Verdict:
        messages = [
            SystemMessage(f"{self.instructions}\n\n{REVIEW_GUIDANCE}"),
            HumanMessage(self.render(artifact, ctx)),
        ]
        return await ainvoke_structured(
            self.model, messages, Verdict, max_attempts=self.max_attempts
        )
