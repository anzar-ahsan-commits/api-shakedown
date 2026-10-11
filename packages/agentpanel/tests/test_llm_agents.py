import asyncio
from typing import Any

from agentpanel import AgentContext, Artifact, Review, Verdict, Workflow
from agentpanel.llm_agents import LLMReviewer, LLMSpecialist
from agentpanel.testing import ScriptedChatModel
from pydantic import BaseModel


class Plan(BaseModel):
    steps: list[str]


def ctx(**overrides: Any) -> AgentContext:
    base: dict[str, Any] = {"inputs": {"topic": "x"}, "artifacts": {}, "reviews": [], "round": 0}
    return AgentContext(**(base | overrides))


def render_topic(c: AgentContext) -> str:
    return f"Write about {c.inputs['topic']}."


def test_text_specialist_strips_reasoning() -> None:
    model = ScriptedChatModel(replies=["<think>planning</think>Final text."])
    agent = LLMSpecialist("writer", "text", model, "You write.", render_topic)
    assert asyncio.run(agent.produce(ctx())) == "Final text."
    system, human = model.calls[0]
    assert system.text == "You write." and human.text == "Write about x."


def test_structured_specialist_returns_schema_instance() -> None:
    model = ScriptedChatModel(replies=['{"steps": ["a", "b"]}'])
    agent = LLMSpecialist("planner", "plan", model, "You plan.", render_topic, output_schema=Plan)
    assert asyncio.run(agent.produce(ctx())) == Plan(steps=["a", "b"])


def test_revision_prompt_includes_previous_attempt_and_feedback() -> None:
    model = ScriptedChatModel(replies=["v2"])
    agent = LLMSpecialist("writer", "text", model, "You write.", render_topic)
    revision = ctx(
        round=1,
        artifacts={"writer": Artifact(author="writer", kind="text", content="v1")},
        reviews=[
            Review(
                reviewer="critic", target="writer", round=0, approved=False, comments=["Too short"]
            ),
            Review(reviewer="fan", target="writer", round=0, approved=True, comments=["Nice"]),
        ],
    )
    asyncio.run(agent.produce(revision))
    prompt = model.calls[0][-1].text
    assert prompt.startswith("Write about x.")
    assert "<previous_attempt>\nv1\n</previous_attempt>" in prompt
    assert "- (critic) Too short" in prompt
    assert "Nice" not in prompt  # only rejections are fed back


def test_reviewer_returns_verdict_and_sees_artifact() -> None:
    model = ScriptedChatModel(replies=['{"approved": false, "comments": ["Missing y"]}'])
    reviewer = LLMReviewer("critic", model, "You review plans.")
    artifact = Artifact(author="planner", kind="plan", content=Plan(steps=["a"]))
    verdict = asyncio.run(reviewer.review(artifact, ctx()))

    assert verdict == Verdict(approved=False, comments=["Missing y"])
    system, human = model.calls[0]
    assert "You review plans." in system.text and "actionable" in system.text
    assert "plan produced by 'planner'" in human.text and '"steps"' in human.text


def test_workflow_with_llm_agents_revises_until_approved() -> None:
    writer_model = ScriptedChatModel(replies=["draft", "better draft"])
    critic_model = ScriptedChatModel(
        replies=[
            '{"approved": false, "comments": ["Be more specific"]}',
            '{"approved": true, "comments": []}',
        ]
    )

    class Join:
        async def assemble(self, c: AgentContext) -> Any:
            return c.artifacts["writer"].content

    workflow = Workflow(
        specialists=[LLMSpecialist("writer", "text", writer_model, "You write.", render_topic)],
        reviewers=[LLMReviewer("critic", critic_model, "You review.")],
        coordinator=Join(),
    )
    result = asyncio.run(workflow.run({"topic": "x"}))

    assert result.result == "better draft"
    assert result.rounds == 1 and result.unresolved == []
    assert "Be more specific" in writer_model.calls[1][-1].text
