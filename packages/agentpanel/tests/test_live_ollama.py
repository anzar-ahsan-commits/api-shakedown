"""Talks to a real local Ollama. Opt in with `pytest -m live`."""

import asyncio

import pytest
from agentpanel import AgentContext, Artifact
from agentpanel.config import LLMSettings
from agentpanel.llm import check_provider, create_chat_model
from agentpanel.llm_agents import LLMReviewer

pytestmark = pytest.mark.live


def test_reviewer_rejects_obvious_bug() -> None:
    settings = LLMSettings()
    status = check_provider(settings)
    if not status.ok:
        pytest.skip(status.detail)

    reviewer = LLMReviewer("critic", create_chat_model(settings), "You review Python code.")
    artifact = Artifact(
        author="coder",
        kind="Python function that adds two numbers",
        content="def add(a, b):\n    return a - b",
    )
    context = AgentContext(inputs={}, artifacts={"coder": artifact}, reviews=[], round=0)
    verdict = asyncio.run(reviewer.review(artifact, context))

    assert not verdict.approved
    assert verdict.comments
