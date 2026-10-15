"""Settings loaded from environment variables and an optional `.env` file.

Applications subclass `LLMSettings` to use their own env prefix, e.g.::

    class MyAppSettings(LLMSettings):
        model_config = SettingsConfigDict(env_prefix="MY_APP_")
"""

from typing import Literal, Self, get_args

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "openai", "anthropic"]

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:8b",
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-5-5",
}
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Cloud keys use each vendor's standard variable name, not the app prefix,
# so keys users already have in their environment just work.
API_KEY_ENV_VARS: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class LLMSettings(BaseSettings):
    """Which LLM to talk to and how."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTPANEL_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Provider = "ollama"
    model: str | None = Field(default=None, description="Defaults per provider if unset.")
    base_url: str | None = Field(
        default=None,
        description="Endpoint override for the selected provider, e.g. an "
        "OpenAI-compatible server. Wins over ollama_url.",
    )
    ollama_url: str = Field(
        default=DEFAULT_OLLAMA_URL,
        description="Where Ollama runs. Only used when the provider is ollama, so it is "
        "safe to set globally (e.g. in a container).",
    )
    temperature: float | None = Field(
        default=None, description="Unset means the provider's default."
    )
    timeout: float | None = Field(
        default=None,
        description="Per-request timeout in seconds. Unset: 600 for Ollama (local models are "
        "slow and serve one request at a time), 120 for cloud providers.",
    )
    context_window: int = Field(
        default=16_384,
        description="Ollama only: tokens of context. Ollama's own default is small and "
        "silently truncates long prompts.",
    )
    max_output_tokens: int = Field(
        default=8_192,
        description="Ollama only: cap on tokens per reply. A small model can fall into a "
        "repetition loop, and Ollama then shifts its context and keeps generating forever; "
        "the cap turns that into a bounded, retryable failure.",
    )

    reasoning: bool | None = Field(
        default=None,
        description="Ollama only: True/False turns a reasoning model's thinking on/off; "
        "unset uses the model's default. Off is much faster for structured generation.",
    )

    agent_models: dict[str, str] = Field(
        default_factory=dict,
        description='Per-agent overrides as JSON, e.g. {"reviewer": "anthropic:claude-sonnet-5-5"} '
        'or {"stubs": "qwen3:14b"}. A known provider prefix switches provider.',
    )
    agent_reasoning: dict[str, bool] = Field(
        default_factory=dict,
        description='Per-agent `reasoning` overrides as JSON, e.g. {"reviewer": true}.',
    )

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    def for_agent(self, name: str) -> Self:
        """Settings for one agent, applying any `agent_models` / `agent_reasoning` override."""
        update: dict[str, object] = {}
        if name in self.agent_reasoning:
            update["reasoning"] = self.agent_reasoning[name]
        spec = self.agent_models.get(name)
        if spec is not None:
            provider, sep, model = spec.partition(":")
            if sep and provider in get_args(Provider):
                update["model"] = model
                if provider != self.provider:
                    # base_url belongs to the original provider; don't leak it to another one
                    update |= {"provider": provider, "base_url": None}
            else:
                update["model"] = spec
        return self.model_copy(update=update) if update else self

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    @property
    def resolved_timeout(self) -> float:
        if self.timeout is not None:
            return self.timeout
        return 600.0 if self.provider == "ollama" else 120.0

    @property
    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url.rstrip("/")
        return self.ollama_url.rstrip("/") if self.provider == "ollama" else None

    @property
    def api_key(self) -> SecretStr | None:
        """The key for the selected provider (None for Ollama)."""
        if self.provider == "openai":
            return self.openai_api_key
        if self.provider == "anthropic":
            return self.anthropic_api_key
        return None
