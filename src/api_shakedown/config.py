"""API Shakedown settings. Env vars use the `SHAKEDOWN_` prefix, e.g. SHAKEDOWN_PROVIDER."""

from agentpanel.config import LLMSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

# Generators produce large, schema-checked JSON: thinking makes them 3-9x slower on
# qwen3 without improving validity. Reviewers keep the model's default.
GENERATOR_AGENTS = ("testdata", "stubs", "contracts", "chaos")


class ShakedownSettings(LLMSettings):
    model_config = SettingsConfigDict(env_prefix="SHAKEDOWN_")

    agent_reasoning: dict[str, bool] = Field(
        default_factory=lambda: dict.fromkeys(GENERATOR_AGENTS, False),
        description="Per-agent thinking on/off (Ollama). Setting this env var replaces "
        "the whole default mapping.",
    )
