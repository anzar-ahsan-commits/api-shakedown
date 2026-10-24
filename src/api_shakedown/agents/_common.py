"""Shared plumbing for the shakedown specialists."""

import time
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

from agentpanel.structured import StructuredOutputError, ainvoke_structured
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
B = TypeVar("B")
Progress = Callable[[str], None]


def batches(items: Sequence[B], agent: str, progress: Progress | None) -> Iterator[B]:
    """Yield each LLM batch, reporting `agent: batch 3/12 done (4m 10s)` after each one.

    Quiet for a single batch: per-batch progress only matters on large specs, where one
    agent can take an hour.
    """
    started = time.monotonic()
    for number, item in enumerate(items, 1):
        yield item
        if progress is not None and len(items) > 1:
            progress(f"  {agent}: batch {number}/{len(items)} done ({_since(started)})")


def _since(started: float) -> str:
    seconds = int(time.monotonic() - started)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


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
