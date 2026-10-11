"""Test helpers for code built on agentpanel."""

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns canned replies in order (or from `respond`) and records every call.

    `respond(messages)` computes the reply from the prompt instead, which suits models
    shared by several agents, e.g. a reviewer that sees different artifacts.
    """

    replies: list[str] = Field(default_factory=list)
    respond: Callable[[list[BaseMessage]], str] | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.call_kwargs.append(kwargs)
        if self.respond is not None:
            text = self.respond(list(messages))
        elif self.replies:
            text = self.replies.pop(0)
        else:
            raise AssertionError("ScriptedChatModel ran out of replies.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])
