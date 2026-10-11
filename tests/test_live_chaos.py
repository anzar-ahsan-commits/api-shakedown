"""Chaos scenarios switched on in a real WireMock (Docker) with the generated chaos.py,
plus LLM-proposed scenarios from a real local Ollama. Opt in with `pytest -m live`."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentpanel import AgentContext
from agentpanel.llm import check_provider, create_chat_model

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.chaos import CHAOS_CLI_FILE, PROBE_RUNNER_FILE, ChaosSuite
from api_shakedown.config import ShakedownSettings
from api_shakedown.spec import load_spec
from api_shakedown.wiremock import base_path

pytestmark = pytest.mark.live

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


def build(model_agents: bool = False) -> tuple[Any, Any, ChaosSuite]:
    spec = load_spec(PETSTORE)
    ctx = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
    dataset = asyncio.run(DatasetAgent().produce(ctx))
    with_data = AgentContext(
        inputs={"spec": spec, "dataset": dataset}, artifacts={}, reviews=[], round=0
    )
    stubs = asyncio.run(StubAgent().produce(with_data))
    model = None
    if model_agents:
        settings = ShakedownSettings()
        status = check_provider(settings)
        if not status.ok:
            pytest.skip(status.detail)
        model = create_chat_model(settings.for_agent("chaos"))
    suite: ChaosSuite = asyncio.run(ChaosAgent(model=model).produce(with_data))
    return spec, stubs, suite


def install(wiremock: httpx.Client, stubs: Any, suite: ChaosSuite, directory: Path) -> Any:
    wiremock.post("/__admin/reset").raise_for_status()
    wiremock.post("/__admin/mappings/import", json=stubs.to_wiremock()).raise_for_status()
    for name, content in suite.files().items():
        (directory / name).write_text(content, encoding="utf-8")
    (directory / "pytest.ini").write_text("[pytest]\n")

    def chaos(*args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(directory / CHAOS_CLI_FILE), "--wiremock",
             str(wiremock.base_url), *args],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return result.stdout

    return chaos


def test_baseline_scenarios_behave_in_wiremock(wiremock: httpx.Client, tmp_path: Path) -> None:
    _, stubs, suite = build()
    chaos = install(wiremock, stubs, suite, tmp_path)
    api = httpx.Client(base_url=str(wiremock.base_url).rstrip("/") + "/v1", timeout=10)

    assert api.get("/pets/1").status_code == 200

    chaos("on", "slow-responses")
    started = time.monotonic()
    assert api.get("/pets/1").status_code == 200
    assert time.monotonic() - started >= 3

    chaos("on", "connection-resets")
    with pytest.raises(httpx.TransportError):
        api.get("/pets")

    chaos("on", "flaky-start")
    assert [api.get("/pets").status_code for _ in range(3)] == [503, 503, 200]
    assert api.get("/pets/2").status_code == 503  # each operation fails twice on its own

    chaos("on", "rate-limited")
    limited = api.get("/pets")
    assert limited.status_code == 429 and limited.headers["Retry-After"] == "1"

    chaos("on", "truncated-json")
    broken = api.get("/pets/1")
    assert broken.status_code == 200
    with pytest.raises(json.JSONDecodeError):
        broken.json()

    chaos("on", "total-outage")
    assert api.delete("/pets/1").status_code == 503

    assert "Chaos off" in chaos("off")
    assert api.get("/pets/1").status_code == 200
    assert api.get("/pets/999").status_code == 404  # dataset-aware stubs are back
    stubs_left = wiremock.get("/__admin/mappings").json()["mappings"]
    assert len(stubs_left) == len(stubs.mappings)


def test_robustness_probes_against_wiremock(wiremock: httpx.Client, tmp_path: Path) -> None:
    spec, stubs, suite = build()
    install(wiremock, stubs, suite, tmp_path)
    env = os.environ | {
        "SHAKEDOWN_TARGET_URL": str(wiremock.base_url).rstrip("/") + base_path(spec),
        "SHAKEDOWN_SELECT_STATUS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(tmp_path / PROBE_RUNNER_FILE)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    print(result.stdout[-600:])
    assert result.returncode == 0


def test_llm_scenarios_activate_in_wiremock(wiremock: httpx.Client, tmp_path: Path) -> None:
    started = time.monotonic()
    _, stubs, suite = build(model_agents=True)
    print(f"\ngeneration took {time.monotonic() - started:.0f}s; notes: {suite.notes}")
    llm = [s for s in suite.scenarios if s.source == "llm"]
    for item in llm:
        print(f"  {item.id} [{item.category}] ops={item.operations or 'all'} "
              f"effect={item.effect.model_dump(exclude_none=True)} times={item.times}\n"
              f"    expect: {item.expectation}")  # fmt: skip
    assert llm, "the LLM should add at least one targeted scenario"

    chaos = install(wiremock, stubs, suite, tmp_path)
    for item in llm:
        assert f"Activated {item.id}" in chaos("on", item.id)
        active = wiremock.get("/__admin/mappings").json()["mappings"]
        assert len(active) == len(stubs.mappings) + len(item.mappings)
    chaos("off")
