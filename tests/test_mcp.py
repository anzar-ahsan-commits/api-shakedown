import asyncio
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from agentpanel.llm import ProviderStatus
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from api_shakedown import mcp_server
from api_shakedown.mcp_server import server

PETSTORE = str(Path(__file__).parents[1] / "examples" / "petstore.yaml")


def call(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments, **kwargs)

    return asyncio.run(go())


def test_tools_and_annotations() -> None:
    async def go() -> Any:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {
        "doctor", "inspect_spec", "generate_harness",
        "list_chaos_scenarios", "activate_chaos", "deactivate_chaos",
    }  # fmt: skip
    assert tools["inspect_spec"].annotations.read_only_hint is True
    assert tools["generate_harness"].annotations.destructive_hint is False
    assert "ctx" not in tools["generate_harness"].input_schema["properties"]
    assert tools["generate_harness"].input_schema["required"] == ["spec", "out_dir"]


def test_inspect_spec() -> None:
    result = call("inspect_spec", {"spec": PETSTORE})
    assert not result.is_error
    report = result.structured_content
    assert report["title"] == "Petstore"
    assert [op["id"] for op in report["operations"]][:2] == ["listPets", "createPet"]


def test_errors_are_reported_to_the_client(tmp_path: Path) -> None:
    missing = call("inspect_spec", {"spec": str(tmp_path / "nope.yaml")})
    assert missing.is_error and "not found" in missing.content[0].text

    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file.txt").write_text("x")
    busy = call(
        "generate_harness", {"spec": PETSTORE, "out_dir": str(tmp_path / "busy"), "offline": True}
    )
    assert busy.is_error and "force=true" in busy.content[0].text


def test_generate_harness_offline_with_progress(tmp_path: Path) -> None:
    progress: list[str] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append(message or "")

    out = tmp_path / "harness"
    result = call(
        "generate_harness",
        {"spec": PETSTORE, "out_dir": str(out), "offline": True},
        progress_callback=on_progress,
    )
    assert not result.is_error, result.content
    report = result.structured_content
    assert report["summary"]["contract_cases"] == 5
    assert "docker-compose.yml" in report["files"] and (out / "README.md").exists()
    assert report["next_steps"][0].endswith("docker compose up -d")
    assert "Generating test data..." in progress and "  stubs: done" in progress

    scenarios = call("list_chaos_scenarios", {"harness_dir": str(out)}).structured_content
    assert "flaky-start" in [s["id"] for s in scenarios["result"]]


def test_generate_harness_explains_provider_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mcp_server, "check_provider", lambda s: ProviderStatus(False, "ollama", "qwen3:8b", "down")
    )
    result = call("generate_harness", {"spec": PETSTORE, "out_dir": str(tmp_path / "h")})
    assert result.is_error and "offline=true" in result.content[0].text


# --- chaos control against a recording WireMock admin API --------------------------------


class _Admin(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else None
        self.calls.append((self.path, body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def admin() -> Iterator[str]:
    _Admin.calls = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Admin)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_activate_and_deactivate_chaos(tmp_path: Path, admin: str) -> None:
    out = tmp_path / "harness"
    call("generate_harness", {"spec": PETSTORE, "out_dir": str(out), "offline": True})

    on = call(
        "activate_chaos",
        {"harness_dir": str(out), "scenario_id": "flaky-start", "wiremock_url": admin},
    )
    assert on.structured_content == {"wiremock_url": admin, "active": "flaky-start",
                                     "mappings_added": 10}  # fmt: skip
    paths = [path for path, _ in _Admin.calls]
    assert paths[:2] == ["/__admin/mappings/remove-by-metadata", "/__admin/scenarios/reset"]
    assert paths[2:] == ["/__admin/mappings"] * 10
    assert _Admin.calls[2][1]["metadata"]["shakedown"]["scenario"] == "flaky-start"

    _Admin.calls.clear()
    off = call("deactivate_chaos", {"wiremock_url": admin})
    assert off.structured_content["active"] is None
    assert [path for path, _ in _Admin.calls] == [
        "/__admin/mappings/remove-by-metadata", "/__admin/scenarios/reset",
    ]  # fmt: skip

    unknown = call(
        "activate_chaos", {"harness_dir": str(out), "scenario_id": "nope", "wiremock_url": admin}
    )
    assert unknown.is_error and "Unknown scenario 'nope'" in unknown.content[0].text


def test_chaos_tools_need_a_harness(tmp_path: Path) -> None:
    result = call("list_chaos_scenarios", {"harness_dir": str(tmp_path)})
    assert result.is_error and "is this a generated harness" in result.content[0].text


# --- the real entry point over stdio --------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args"),
    [("shakedown-mcp", []), ("api-shakedown", ["mcp"]), ("shakedown", ["mcp"])],
)
def test_stdio_entry_points(tmp_path: Path, command: str, args: list[str]) -> None:
    """Every documented way to start the server, incl. what `uvx api-shakedown mcp` runs."""
    exe = Path(sys.executable).with_name(command)
    params = StdioServerParameters(command=str(exe), args=args, cwd=tmp_path)

    async def go() -> Any:
        async with Client(params) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            report = await client.call_tool("inspect_spec", {"spec": PETSTORE})
            return names, report

    names, report = asyncio.run(go())
    assert "generate_harness" in names
    assert report.structured_content["version"] == "1.0.0"
