from pathlib import Path

import pytest
from agentpanel.config import DEFAULT_OLLAMA_URL, LLMSettings
from pydantic_settings import SettingsConfigDict


def test_defaults_to_local_ollama() -> None:
    settings = LLMSettings()
    assert settings.provider == "ollama"
    assert settings.resolved_model == "qwen3:8b"
    assert settings.resolved_base_url == DEFAULT_OLLAMA_URL
    assert settings.api_key is None
    assert settings.resolved_timeout == 600.0


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTPANEL_PROVIDER", "openai")
    monkeypatch.setenv("AGENTPANEL_MODEL", "my-model")
    settings = LLMSettings()
    assert settings.provider == "openai"
    assert settings.resolved_model == "my-model"
    assert settings.resolved_base_url is None
    assert settings.resolved_timeout == 120.0


def test_api_key_uses_standard_vendor_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = LLMSettings(provider="anthropic")
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)


def test_reads_dotenv_file() -> None:
    Path(".env").write_text("AGENTPANEL_PROVIDER=openai\nOPENAI_API_KEY=sk-dotenv\n")
    settings = LLMSettings()
    assert settings.provider == "openai"
    assert settings.api_key is not None


def test_for_agent_overrides_model_only() -> None:
    settings = LLMSettings(agent_models={"stubs": "qwen3:14b"})
    assert settings.for_agent("stubs").resolved_model == "qwen3:14b"  # ollama tag, not provider
    assert settings.for_agent("stubs").provider == "ollama"
    assert settings.for_agent("other") is settings


def test_for_agent_can_switch_provider_without_leaking_base_url() -> None:
    settings = LLMSettings(
        base_url="http://gpu-box:11434",
        agent_models={"reviewer": "anthropic:claude-sonnet-5-5"},
    )
    reviewer = settings.for_agent("reviewer")
    assert reviewer.provider == "anthropic"
    assert reviewer.resolved_model == "claude-sonnet-5-5"
    assert reviewer.base_url is None


def test_ollama_url_only_applies_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTPANEL_OLLAMA_URL", "http://host.docker.internal:11434/")
    assert LLMSettings().resolved_base_url == "http://host.docker.internal:11434"
    assert LLMSettings(provider="anthropic").resolved_base_url is None
    assert LLMSettings(base_url="http://gpu:11434").resolved_base_url == "http://gpu:11434"


def test_for_agent_reasoning_override() -> None:
    settings = LLMSettings(reasoning=True, agent_reasoning={"stubs": False})
    assert settings.for_agent("stubs").reasoning is False
    assert settings.for_agent("reviewer").reasoning is True


def test_agent_models_from_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTPANEL_AGENT_MODELS", '{"reviewer": "openai:gpt-5-mini"}')
    assert LLMSettings().for_agent("reviewer").provider == "openai"


def test_subclass_can_change_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    class AppSettings(LLMSettings):
        model_config = SettingsConfigDict(env_prefix="MYAPP_")

    monkeypatch.setenv("MYAPP_MODEL", "llama3.2")
    assert AppSettings().resolved_model == "llama3.2"
