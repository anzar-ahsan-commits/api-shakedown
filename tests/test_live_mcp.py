"""The MCP server with a real local Ollama and a real WireMock, and inside the Docker
image. Opt in with `pytest -m live`."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentpanel.llm import check_provider
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from api_shakedown.config import ShakedownSettings
from api_shakedown.mcp_server import server

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[1]
PETSTORE = str(ROOT / "examples" / "petstore.yaml")
IMAGE = "api-shakedown:test"


def test_mcp_generates_with_llm_and_drives_chaos(wiremock: httpx.Client, tmp_path: Path) -> None:
    status = check_provider(ShakedownSettings())
    if not status.ok:
        pytest.skip(status.detail)
    out = tmp_path / "harness"
    progress: list[str] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append(message or "")

    async def go() -> tuple[Any, Any, Any]:
        async with Client(server) as client:
            made = await client.call_tool(
                "generate_harness",
                {"spec": PETSTORE, "out_dir": str(out), "review": False},
                progress_callback=on_progress,
            )
            stubs = json.loads((out / "wiremock/mappings/stubs.json").read_text("utf-8"))
            wiremock.post("/__admin/reset").raise_for_status()
            wiremock.post("/__admin/mappings/import", json=stubs).raise_for_status()
            url = str(wiremock.base_url).rstrip("/")
            on = await client.call_tool(
                "activate_chaos",
                {"harness_dir": str(out), "scenario_id": "total-outage", "wiremock_url": url},
            )
            down = wiremock.get("/v1/pets").status_code
            await client.call_tool("deactivate_chaos", {"wiremock_url": url})
            return made, on, down

    made, on, down = asyncio.run(go())
    print("\nprogress:", *progress, sep="\n  ")
    assert not made.is_error, made.content
    print("summary:", made.structured_content["summary"])
    assert made.structured_content["summary"]["contract_cases"] >= 5
    assert not on.is_error and down == 503
    assert wiremock.get("/v1/pets").status_code == 200


def test_mcp_server_runs_in_docker_image() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed.")
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(ROOT)], check=True,
                   capture_output=True)  # fmt: skip
    params = StdioServerParameters(
        command="docker", args=["run", "-i", "--rm", "--entrypoint", "shakedown-mcp", IMAGE]
    )

    async def go() -> tuple[list[str], Any, Any]:
        async with Client(params) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            spec = await client.call_tool(
                "inspect_spec", {"spec": "/opt/shakedown/examples/petstore.yaml"}
            )
            doctor = await client.call_tool("doctor", {})
            return names, spec, doctor

    names, spec, doctor = asyncio.run(go())
    assert "generate_harness" in names
    assert spec.structured_content["title"] == "Petstore"
    print("\ndoctor in container:", doctor.structured_content)
    assert doctor.structured_content["ok"], "the container should reach Ollama on the host"
