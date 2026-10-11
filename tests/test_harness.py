import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from agentpanel import Review
from agentpanel.testing import ScriptedChatModel
from typer.testing import CliRunner

from api_shakedown import cli
from api_shakedown.agents.coordinator import Harness, _final_reviews
from api_shakedown.harness import HarnessExistsError, write_harness
from api_shakedown.pipeline import PipelineOptions, generate_harness
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec
from api_shakedown.wiremock import StubSet

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
EXPECTED_FILES = {
    "README.md", "shakedown.json", "docker-compose.yml", "requirements.txt",
    "wiremock/mappings/stubs.json", "data/dataset.json",
    "chaos/chaos.py", "chaos/chaos_scenarios.json",
    "tests/pytest.ini", "tests/shakedown_support.py",
    "tests/test_contract.py", "tests/contract_cases.json",
    "tests/test_robustness.py", "tests/robustness_cases.json",
}  # fmt: skip


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def offline(spec: ApiSpec, **kwargs: Any) -> Harness:
    options = PipelineOptions(offline=True, generated_at="2026-01-01 00:00 UTC")
    return asyncio.run(generate_harness(spec, options=options, **kwargs))


def test_offline_harness_files(spec: ApiSpec) -> None:
    harness = offline(spec)
    assert set(harness.files) == EXPECTED_FILES
    assert harness.summary == {
        "operations": 5, "stub_mappings": 15, "contract_cases": 5,
        "chaos_scenarios": 7, "robustness_probes": 30, "records": 3,
    }  # fmt: skip
    compose = harness.files["docker-compose.yml"]
    assert "wiremock/wiremock:3.13.1" in compose and "${SHAKEDOWN_WIREMOCK_PORT:-8080}" in compose
    stubs = json.loads(harness.files["wiremock/mappings/stubs.json"])
    assert len(stubs["mappings"]) == 15
    manifest = json.loads(harness.files["shakedown.json"])
    assert manifest["summary"]["contract_cases"] == 5
    assert manifest["generated_at"] == "2026-01-01 00:00 UTC"
    for name in ("chaos/chaos.py", "tests/test_contract.py", "tests/test_robustness.py"):
        compile(harness.files[name], name, "exec")


def test_readme_describes_the_harness(spec: ApiSpec) -> None:
    readme = offline(spec).files["README.md"]
    assert readme.startswith("# Chaos test harness: Petstore 1.0.0")
    assert "on 2026-01-01 00:00 UTC" in readme
    assert "SHAKEDOWN_SELECT_STATUS=1 pytest tests/test_contract.py" in readme
    assert "| `flaky-start` | flaky | First 2 calls fail with 503, then recover |" in readme
    assert "http://localhost:8080/v1" in readme
    assert "## Review" not in readme  # offline: nothing was reviewed


def test_progress_and_review_outcome_in_readme(spec: ApiSpec) -> None:
    rejection = json.dumps(
        {"findings": [{"severity": "blocker", "item": "getPet 404", "problem": "vague",
                       "fix": "say not found"}]}
    )  # fmt: skip

    def respond(messages: list[Any]) -> str:  # never satisfied with the stubs
        return rejection if "produced by 'stubs'" in messages[-1].text else '{"findings": []}'

    reviewer = ScriptedChatModel(respond=respond)
    messages: list[str] = []
    options = PipelineOptions(review=True, max_revisions=1)
    harness = asyncio.run(
        generate_harness(
            spec,
            options=options,
            progress=messages.append,
            model_for=lambda agent: reviewer if agent == "reviewer" else None,
        )
    )
    assert "  review stubs: rejected (1 blocker(s))" in messages
    assert "Revising: stubs" in messages
    assert "  stubs: done (revision 1)" in messages
    assert harness.unresolved == ["stubs"]
    readme = harness.files["README.md"]
    assert "- stubs: REJECTED\n  - [blocker] getPet 404: vague Fix: say not found" in readme
    assert "Still rejected when the revision budget ran out: stubs." in readme


def test_final_reviews_merge_reviewers_and_keep_latest_round() -> None:
    reviews = [
        Review(reviewer="a", target="stubs", round=0, approved=False, comments=["old"]),
        Review(reviewer="a", target="stubs", round=1, approved=True, comments=["fine"]),
        Review(reviewer="b", target="stubs", round=1, approved=False, comments=["no"]),
    ]
    (final,) = _final_reviews(reviews)
    assert (final.round, final.approved, final.comments) == (1, False, ["fine", "no"])


def test_write_harness_refuses_non_empty_dirs(spec: ApiSpec, tmp_path: Path) -> None:
    harness = offline(spec)
    keep = tmp_path / "mine.txt"
    keep.write_text("keep me")
    with pytest.raises(HarnessExistsError):
        write_harness(harness, tmp_path)
    written = write_harness(harness, tmp_path, force=True)
    assert len(written) == len(EXPECTED_FILES)
    assert keep.read_text() == "keep me"  # --force never deletes other files


def test_generated_harness_runs_end_to_end(
    spec: ApiSpec, tmp_path: Path, serve_stubs: Callable[[StubSet], tuple[str, Any]]
) -> None:
    harness = offline(spec)
    write_harness(harness, tmp_path)
    stubs = StubSet(**json.loads((tmp_path / "wiremock/mappings/stubs.json").read_text()))
    url, _ = serve_stubs(stubs)
    env = os.environ | {"SHAKEDOWN_TARGET_URL": url + "/v1", "SHAKEDOWN_SELECT_STATUS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert "20 passed, 15 skipped" in result.stdout  # 5 contract + 15 createPet probes


# --- the CLI ------------------------------------------------------------------------------


runner = CliRunner()


def test_cli_run_offline(tmp_path: Path) -> None:
    out = tmp_path / "harness"
    result = runner.invoke(cli.app, ["run", str(PETSTORE), "--out", str(out), "--offline"])
    assert result.exit_code == 0, result.stdout
    assert "Wrote 14 files" in result.stdout and "15 stubs, 5 contract cases" in result.stdout
    assert (out / "README.md").exists()

    again = runner.invoke(cli.app, ["run", str(PETSTORE), "--out", str(out), "--offline"])
    assert again.exit_code == 2 and "not empty" in again.stderr
    forced = runner.invoke(
        cli.app, ["run", str(PETSTORE), "--out", str(out), "--offline", "--force"]
    )
    assert forced.exit_code == 0


def test_cli_run_explains_provider_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agentpanel.llm import ProviderStatus

    monkeypatch.setattr(
        cli, "check_provider", lambda s: ProviderStatus(False, "ollama", "qwen3:8b", "down")
    )
    result = runner.invoke(cli.app, ["run", str(PETSTORE), "--out", str(tmp_path / "x")])
    assert result.exit_code == 1
    assert "use --offline" in result.stderr


def test_cli_run_rejects_bad_spec(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", str(tmp_path / "nope.yaml"), "--offline"])
    assert result.exit_code == 2 and "not found" in result.stderr
