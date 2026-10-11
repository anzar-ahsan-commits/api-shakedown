import json
from pathlib import Path

import pytest
from agentpanel.llm import ProviderStatus
from typer.testing import CliRunner

from api_shakedown import __version__, cli
from api_shakedown.cli import app
from api_shakedown.config import ShakedownSettings

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


PETSTORE = str(Path(__file__).parents[1] / "examples" / "petstore.yaml")


def test_inspect_summarizes_spec() -> None:
    result = runner.invoke(app, ["inspect", PETSTORE])
    assert result.exit_code == 0
    assert "Petstore 1.0.0 (OpenAPI 3.0.3): 5 operations in 1 chunk(s)" in result.stdout
    assert "GET     /pets/{petId}" in result.stdout


def test_inspect_json() -> None:
    result = runner.invoke(app, ["inspect", PETSTORE, "--json"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["operations"]) == 5


def test_inspect_reports_unusable_spec() -> None:
    result = runner.invoke(app, ["inspect", "does-not-exist.yaml"])
    assert result.exit_code == 2
    assert "not found" in result.stderr


def test_settings_use_shakedown_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAKEDOWN_PROVIDER", "anthropic")
    assert ShakedownSettings().provider == "anthropic"


def test_doctor_fails_with_hint_when_key_missing() -> None:
    result = runner.invoke(app, ["doctor", "--provider", "openai"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.stdout


def test_doctor_passes_options_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ShakedownSettings] = []

    def fake_check(settings: ShakedownSettings) -> ProviderStatus:
        seen.append(settings)
        return ProviderStatus(True, settings.provider, settings.resolved_model, "fine")

    monkeypatch.setattr(cli, "check_provider", fake_check)
    result = runner.invoke(app, ["doctor", "--model", "llama3.2"])
    assert result.exit_code == 0
    assert seen[0].resolved_model == "llama3.2"
    assert "[OK] ollama / llama3.2" in result.stdout
