"""Shared plumbing for the shakedown specialists."""

from collections.abc import Callable
from typing import TypeVar

from agentpanel.structured import StructuredOutputError, ainvoke_structured
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


async def generate_checked(
    model: BaseChatModel,
    instructions: str,
    prompt: str,
    schema: type[T],
    *,
    check: Callable[[T], list[str]],
    max_attempts: int,
    notes: list[str],
) -> T | None:
    """Structured generation that never raises on bad output.

    Returns the passing reply, or the last reply that parsed (so callers can keep its
    valid parts), or None. Problems are recorded in `notes`.
    """
    messages = [SystemMessage(instructions), HumanMessage(prompt)]
    try:
        return await ainvoke_structured(
            model, messages, schema, max_attempts=max_attempts, check=check
        )
    except StructuredOutputError as exc:
        notes.append(f"Some LLM output was unusable: {exc}")
        return exc.value if isinstance(exc.value, schema) else None
