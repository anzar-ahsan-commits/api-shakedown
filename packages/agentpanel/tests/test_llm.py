import sys
from typing import Any

import httpx
import pytest
from agentpanel.config import LLMSettings, Provider
from agentpanel.llm import (
    MissingAPIKeyError,
    ProviderNotInstalledError,
    check_provider,
    create_chat_model,
)
from langchain_ollama import ChatOllama
from pydantic import SecretStr


def _ollama_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _tags(*names: str) -> Any:
    return lambda request: httpx.Response(200, json={"models": [{"name": n} for n in names]})


def test_creates_ollama_model_by_default() -> None:
    model = create_chat_model(LLMSettings(temperature=0.1))
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:8b"
    assert model.temperature == 0.1


def test_ollama_replies_are_capped_so_a_looping_model_cannot_run_forever() -> None:
    model = create_chat_model(LLMSettings())
    assert isinstance(model, ChatOllama)
    assert model.num_predict == 8192 and model.num_ctx == 16384
    custom = create_chat_model(LLMSettings(max_output_tokens=2000))
    assert isinstance(custom, ChatOllama) and custom.num_predict == 2000


@pytest.mark.parametrize(
    ("provider", "class_name"), [("openai", "ChatOpenAI"), ("anthropic", "ChatAnthropic")]
)
def test_creates_cloud_model_with_byo_key(provider: Provider, class_name: str) -> None:
    settings = LLMSettings(
        provider=provider,
        openai_api_key=SecretStr("sk-openai"),
        anthropic_api_key=SecretStr("sk-anthropic"),
    )
    model = create_chat_model(settings)
    assert type(model).__name__ == class_name


def test_openai_compatible_base_url() -> None:
    settings = LLMSettings(
        provider="openai", base_url="http://localhost:1234/v1/", openai_api_key=SecretStr("x")
    )
    model: Any = create_chat_model(settings)
    assert model.openai_api_base == "http://localhost:1234/v1"


def test_cloud_provider_without_key_raises() -> None:
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        create_chat_model(LLMSettings(provider="anthropic"))


def test_missing_provider_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_openai", None)  # makes the import fail
    with pytest.raises(ProviderNotInstalledError, match="`openai` extra"):
        create_chat_model(LLMSettings(provider="openai", openai_api_key=SecretStr("sk-test")))


def test_ollama_healthy() -> None:
    status = check_provider(LLMSettings(), client=_ollama_client(_tags("qwen3:8b")))
    assert status.ok


def test_ollama_latest_tag_is_implicit() -> None:
    settings = LLMSettings(model="llama3.2")
    assert check_provider(settings, client=_ollama_client(_tags("llama3.2:latest"))).ok


def test_ollama_model_not_pulled() -> None:
    status = check_provider(LLMSettings(), client=_ollama_client(_tags("other:1b")))
    assert not status.ok
    assert "ollama pull qwen3:8b" in status.detail


def test_ollama_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    status = check_provider(LLMSettings(), client=_ollama_client(refuse))
    assert not status.ok
    assert "ollama serve" in status.detail


def test_cloud_check_reports_missing_key() -> None:
    status = check_provider(LLMSettings(provider="anthropic"))
    assert not status.ok
    assert "ANTHROPIC_API_KEY" in status.detail
