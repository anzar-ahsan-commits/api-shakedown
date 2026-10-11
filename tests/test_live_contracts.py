"""LLM-written stubs and contract tests, run against a real WireMock (Docker) with a real
local Ollama. Opt in with `pytest -m live`."""

import asyncio
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

from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.config import ShakedownSettings
from api_shakedown.spec import load_spec
from api_shakedown.wiremock import base_path

pytestmark = pytest.mark.live

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


def test_llm_pipeline_dataset_stubs_contracts(wiremock: httpx.Client, tmp_path: Path) -> None:
    settings = ShakedownSettings()
    status = check_provider(settings)
    if not status.ok:
        pytest.skip(status.detail)

    spec = load_spec(PETSTORE)
    ctx = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)

    async def generate() -> Any:
        data = DatasetAgent(model=create_chat_model(settings.for_agent("testdata")))
        dataset = await data.produce(ctx)
        with_data = AgentContext(
            inputs={"spec": spec, "dataset": dataset}, artifacts={}, reviews=[], round=0
        )
        stubs = StubAgent(model=create_chat_model(settings.for_agent("stubs")))
        contracts = ContractTestAgent(model=create_chat_model(settings.for_agent("contracts")))
        return dataset, *await asyncio.gather(
            stubs.produce(with_data), contracts.produce(with_data)
        )

    started = time.monotonic()
    dataset, stub_set, suite = asyncio.run(generate())
    print(f"\ngeneration took {time.monotonic() - started:.0f}s")
    print("pets:", *dataset.entities["Pet"].records, sep="\n  ")
    assert dataset.notes == [], dataset.notes
    print("stub notes:", *stub_set.notes, "\ncontract notes:", *suite.notes, sep="\n  ")
    print("cases:", *(c.id for c in suite.cases), sep="\n  ")
    fallbacks = [n for n in suite.notes if "schema-generated happy-path" in n]
    assert len(suite.cases) > len(spec.operations), "the LLM should add cases beyond happy paths"
    assert len(fallbacks) <= len(spec.operations) // 2

    wiremock.post("/__admin/reset").raise_for_status()
    wiremock.post("/__admin/mappings/import", json=stub_set.to_wiremock()).raise_for_status()

    for name, content in suite.files().items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = os.environ | {
        "SHAKEDOWN_TARGET_URL": str(wiremock.base_url).rstrip("/") + base_path(spec),
        "SHAKEDOWN_SELECT_STATUS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    print(result.stdout[-3000:])
    assert result.returncode == 0

    known = dataset.entities["Pet"].ids()
    served = wiremock.get(f"/v1/pets/{known[-1]}").json()
    assert served == dataset.entities["Pet"].record(known[-1])
