"""LLM provider abstraction: Ollama by default, bring-your-own-key for cloud providers.

Provider packages are imported lazily so cloud SDKs stay optional extras.
"""

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, cast

import httpx
from langchain_core.language_models import BaseChatModel

from agentpanel.config import API_KEY_ENV_VARS, LLMSettings, Provider

# provider -> (module, chat class, pip extra)
_PROVIDERS: dict[Provider, tuple[str, str, str | None]] = {
    "ollama": ("langchain_ollama", "ChatOllama", None),
    "openai": ("langchain_openai", "ChatOpenAI", "openai"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "anthropic"),
}


class LLMConfigError(Exception):
    """The configured provider cannot be used as-is. The message says how to fix it."""


class ProviderNotInstalledError(LLMConfigError):
    pass


class MissingAPIKeyError(LLMConfigError):
    pass


def create_chat_model(settings: LLMSettings | None = None) -> BaseChatModel:
    """Build a LangChain chat model for the configured provider. Makes no network calls."""
    settings = settings or LLMSettings()
    if settings.provider != "ollama" and settings.api_key is None:
        env_var = API_KEY_ENV_VARS[settings.provider]
        raise MissingAPIKeyError(
            f"Provider '{settings.provider}' needs an API key. Set {env_var} "
            "in your environment or .env file."
        )
    _, class_name, _ = _PROVIDERS[settings.provider]
    chat_class = getattr(_import_provider(settings.provider), class_name)

    kwargs: dict[str, Any] = {"model": settings.resolved_model}
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.resolved_base_url:
        kwargs["base_url"] = settings.resolved_base_url
    if settings.provider == "ollama":
        kwargs["client_kwargs"] = {"timeout": settings.resolved_timeout}
        kwargs["num_ctx"] = settings.context_window
        if settings.reasoning is not None:
            kwargs["reasoning"] = settings.reasoning
    else:
        kwargs["api_key"] = settings.api_key
        kwargs["timeout"] = settings.resolved_timeout

    return cast(BaseChatModel, chat_class(**kwargs))


def _import_provider(provider: Provider) -> Any:
    module_name, _, extra = _PROVIDERS[provider]
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = module_name.replace("_", "-")
        hint = f"Install `{package}`" + (f" (the `{extra}` extra)." if extra else ".")
        raise ProviderNotInstalledError(f"Provider '{provider}' is not installed. {hint}") from exc


@dataclass(frozen=True)
class ProviderStatus:
    ok: bool
    provider: Provider
    model: str
    detail: str


def check_provider(
    settings: LLMSettings | None = None, *, client: httpx.Client | None = None
) -> ProviderStatus:
    """Check the configured provider is usable, with an actionable message if not.

    For Ollama this contacts the server and checks the model is pulled. For cloud
    providers it only checks the package and key are present; it does not spend
    tokens validating the key.
    """
    settings = settings or LLMSettings()
    model = settings.resolved_model

    def status(ok: bool, detail: str) -> ProviderStatus:
        return ProviderStatus(ok=ok, provider=settings.provider, model=model, detail=detail)

    module_name, _, extra = _PROVIDERS[settings.provider]
    if importlib.util.find_spec(module_name) is None:
        return status(
            False,
            f"Package `{module_name.replace('_', '-')}` is not installed (the `{extra}` extra).",
        )

    if settings.provider != "ollama":
        env_var = API_KEY_ENV_VARS[settings.provider]
        if settings.api_key is None:
            return status(False, f"No API key. Set {env_var} in your environment or .env file.")
        return status(True, f"{env_var} is set (not validated against the API).")

    return status(*_check_ollama(settings, model, client))


def _check_ollama(
    settings: LLMSettings, model: str, client: httpx.Client | None
) -> tuple[bool, str]:
    url = settings.resolved_base_url
    http = client or httpx.Client(timeout=5.0)
    try:
        response = http.get(f"{url}/api/tags")
        response.raise_for_status()
        pulled = {m["name"] for m in response.json().get("models", [])}
    except httpx.HTTPError as exc:
        return False, (
            f"Cannot reach Ollama at {url} ({exc.__class__.__name__}). "
            "Is it running? Start it with `ollama serve`."
        )
    finally:
        if client is None:
            http.close()

    if _ollama_tag(model) not in {_ollama_tag(name) for name in pulled}:
        return False, f"Model '{model}' is not pulled. Run `ollama pull {model}`."
    return True, f"Ollama is running at {url} and '{model}' is available."


def _ollama_tag(name: str) -> str:
    """`qwen3` and `qwen3:latest` refer to the same Ollama model."""
    return name if ":" in name else f"{name}:latest"
