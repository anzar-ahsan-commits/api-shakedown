"""Structured output that works across providers, including small local models.

The JSON Schema goes in the prompt, the reply is parsed and validated with pydantic,
and validation errors are fed back to the model for another attempt. On Ollama,
decoding is additionally constrained to the schema.
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_MAX_ERROR_CHARS = 1500

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model never produced output that both parsed and passed the checks.

    `value` holds the last reply that parsed but failed `check`, if any, so callers
    can salvage the valid parts.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        attempts: int,
        problem: str,
        reply: str,
        value: BaseModel | None = None,
    ):
        super().__init__(
            f"No valid {schema.__name__} after {attempts} attempt(s). Last problem: {problem}"
        )
        self.reply = reply
        self.value = value


def strip_reasoning(text: str) -> str:
    """Drop `<think>...</think>` blocks that reasoning models put in their content."""
    return _THINK.sub("", text).strip()


def message_text(message: BaseMessage) -> str:
    return strip_reasoning(message.text)


def format_content(content: Any) -> str:
    """Render artifact content for a prompt."""
    if isinstance(content, str):
        return content
    if isinstance(content, BaseModel):
        return content.model_dump_json(indent=2)
    return json.dumps(content, indent=2, default=str)


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with only a JSON object (no prose) that matches this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), indent=2)
    )


def parse_reply(text: str, schema: type[T]) -> T:
    """Validate a reply, tolerating reasoning tags, code fences, and surrounding prose."""
    text = strip_reasoning(text)
    candidates = [text]
    if fenced := _FENCE.search(text):
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    error: ValidationError | None = None
    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            error = error or exc  # the error for the full reply is the most informative
    assert error is not None
    raise error


async def ainvoke_structured(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    schema: type[T],
    *,
    max_attempts: int = 3,
    check: Callable[[T], list[str]] | None = None,
) -> T:
    """Call `model` until its reply validates against `schema` and passes `check`.

    `check` returns domain-level problems (empty list = fine). Schema errors and check
    problems are both fed back to the model for the next attempt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    runnable = _constrained(model, schema)
    conversation = _with_instructions(list(messages), schema_instructions(schema))
    problem = ""
    text = ""
    last_value: T | None = None
    for _ in range(max_attempts):
        reply = await runnable.ainvoke(conversation)
        text = message_text(reply)
        try:
            value = parse_reply(text, schema)
        except ValidationError as exc:
            problem = str(exc)
        else:
            problems = check(value) if check else []
            if not problems:
                return value
            last_value = value
            problem = "\n".join(f"- {p}" for p in problems)
        conversation += [
            AIMessage(text),
            HumanMessage(
                f"That reply was invalid:\n{problem[:_MAX_ERROR_CHARS]}\n\n"
                "Respond again with the complete, corrected JSON object."
            ),
        ]
    raise StructuredOutputError(schema, max_attempts, problem, text, last_value)


def _constrained(model: BaseChatModel, schema: type[BaseModel]) -> Runnable[Any, BaseMessage]:
    if isinstance(model, ChatOllama):
        return model.bind(format=schema.model_json_schema())
    return model


def _with_instructions(messages: list[BaseMessage], instructions: str) -> list[BaseMessage]:
    """Append to the final user message rather than adding a second one in a row."""
    last = messages[-1] if messages else None
    if isinstance(last, HumanMessage) and isinstance(last.content, str):
        return [*messages[:-1], HumanMessage(f"{last.content}\n\n{instructions}")]
    return [*messages, HumanMessage(instructions)]
