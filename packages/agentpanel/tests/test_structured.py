import asyncio

import pytest
from agentpanel.structured import (
    StructuredOutputError,
    _constrained,
    ainvoke_structured,
    parse_reply,
)
from agentpanel.testing import ScriptedChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


@pytest.mark.parametrize(
    "reply",
    [
        '{"x": 1, "y": 2}',
        '<think>hmm, {"x": 9}</think>\n{"x": 1, "y": 2}',
        'Sure!\n```json\n{"x": 1, "y": 2}\n```',
        'Here you go: {"x": 1, "y": 2} Hope that helps.',
    ],
)
def test_parse_reply_tolerates_wrapping(reply: str) -> None:
    assert parse_reply(reply, Point) == Point(x=1, y=2)


def test_parse_reply_keeps_fences_inside_json_strings() -> None:
    class Code(BaseModel):
        source: str

    reply = '{"source": "```python\\nprint(1)\\n```"}'
    assert parse_reply(reply, Code).source == "```python\nprint(1)\n```"


def test_retries_with_error_feedback() -> None:
    model = ScriptedChatModel(replies=['{"x": "one"}', '{"x": 1, "y": 2}'])
    messages = [SystemMessage("sys"), HumanMessage("give me a point")]
    result = asyncio.run(ainvoke_structured(model, messages, Point))

    assert result == Point(x=1, y=2)
    first, second = model.calls
    assert "JSON Schema" in first[-1].text and "give me a point" in first[-1].text
    assert len(first) == 2  # instructions merged into the user message
    assert second[-2].text == '{"x": "one"}'
    assert "invalid" in second[-1].text and "y" in second[-1].text


def test_gives_up_after_max_attempts() -> None:
    model = ScriptedChatModel(replies=["nope", "still nope"])
    with pytest.raises(StructuredOutputError, match="after 2 attempt") as info:
        asyncio.run(ainvoke_structured(model, [HumanMessage("hi")], Point, max_attempts=2))
    assert info.value.reply == "still nope"


def test_ollama_decoding_is_constrained_to_schema() -> None:
    bound = _constrained(ChatOllama(model="qwen3:8b"), Point)
    assert isinstance(bound, RunnableBinding)
    assert bound.kwargs["format"] == Point.model_json_schema()


def test_other_models_are_not_bound() -> None:
    model = ScriptedChatModel(replies=[])
    assert _constrained(model, Point) is model
