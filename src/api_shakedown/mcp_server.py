"""MCP server exposing API Shakedown as tools (`shakedown-mcp`, stdio transport).

LLM settings come from the server's environment (SHAKEDOWN_*, OPENAI_API_KEY, ...), as
for the CLI. Relative paths resolve against the server's working directory, so clients
should pass absolute paths.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from agentpanel.llm import check_provider
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from api_shakedown import __version__, chaos_control
from api_shakedown.config import ShakedownSettings
from api_shakedown.harness import HarnessExistsError, write_harness
from api_shakedown.pipeline import PipelineOptions
from api_shakedown.pipeline import generate_harness as run_pipeline
from api_shakedown.spec import SpecError, load_spec
from api_shakedown.wiremock import DEFAULT_STUB_URL

INSTRUCTIONS = """\
API Shakedown turns an OpenAPI spec into a ready-to-run test harness: WireMock stubs, \
contract tests, robustness probes, and chaos scenarios. Typical flow: inspect_spec, then \
generate_harness (slow with an LLM: minutes; offline=true takes seconds), then have the \
user run `docker compose up -d` in the harness and use activate_chaos / deactivate_chaos \
while their client talks to the stubs. Use absolute paths."""

server = MCPServer(name="api-shakedown", version=__version__, instructions=INSTRUCTIONS)


class ProviderReport(BaseModel):
    ok: bool
    provider: str
    model: str
    detail: str


class OperationInfo(BaseModel):
    id: str
    method: str
    path: str
    summary: str | None = None


class SpecReport(BaseModel):
    title: str
    version: str
    openapi_version: str
    operations: list[OperationInfo]
    warnings: list[str]


class HarnessReport(BaseModel):
    out_dir: str
    files: list[str]
    summary: dict[str, int]
    unresolved: list[str]
    reviews: list[str]
    notes: list[str]
    next_steps: list[str]


class ScenarioInfo(BaseModel):
    id: str
    category: str
    title: str
    expectation: str


class ChaosReport(BaseModel):
    wiremock_url: str
    active: str | None
    mappings_added: int = 0


@server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
async def doctor() -> ProviderReport:
    """Check that the configured LLM provider is reachable and the model is available."""
    status = await asyncio.to_thread(check_provider, ShakedownSettings())
    return ProviderReport(
        ok=status.ok, provider=status.provider, model=status.model, detail=status.detail
    )


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
async def inspect_spec(spec: str) -> SpecReport:
    """Load and validate an OpenAPI spec (file path or http(s) URL) and list its operations."""
    api = await asyncio.to_thread(_load, spec)
    return SpecReport(
        title=api.title,
        version=api.version,
        openapi_version=api.openapi_version,
        operations=[
            OperationInfo(id=op.id, method=op.method.upper(), path=op.path, summary=op.summary)
            for op in api.operations
        ],
        warnings=api.warnings,
    )


@server.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False))
async def generate_harness(
    spec: str,
    out_dir: str,
    ctx: Context,
    offline: bool = False,
    review: bool = True,
    max_revisions: int = 1,
    force: bool = False,
) -> HarnessReport:
    """Generate a ready-to-run chaos test harness for an OpenAPI spec into out_dir.

    offline=true skips the LLM (deterministic, seconds); otherwise this takes minutes on a
    local model, and review=false roughly halves that. Refuses a non-empty out_dir unless
    force=true (which overwrites only the harness's own files).
    """
    api = await asyncio.to_thread(_load, spec)
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()) and not force:
        raise ToolError(f"{out} is not empty; pass force=true to overwrite.")
    settings = ShakedownSettings()
    if not offline:
        status = await asyncio.to_thread(check_provider, settings)
        if not status.ok:
            raise ToolError(f"{status.provider}/{status.model}: {status.detail} (or offline=true)")

    loop = asyncio.get_running_loop()
    reports: list[asyncio.Task[None]] = []

    def progress(message: str) -> None:
        reports.append(loop.create_task(ctx.report_progress(len(reports) + 1, None, message)))

    options = PipelineOptions(
        offline=offline, review=review and not offline, max_revisions=max_revisions
    )
    harness = await run_pipeline(api, settings, options, progress=progress)
    await asyncio.gather(*reports)
    try:
        written = await asyncio.to_thread(write_harness, harness, out, force=force)
    except HarnessExistsError as exc:
        raise ToolError(str(exc)) from exc
    return HarnessReport(
        out_dir=str(out.resolve()),
        files=sorted(str(p.relative_to(out)).replace("\\", "/") for p in written),
        summary=harness.summary,
        unresolved=harness.unresolved,
        reviews=harness.reviews,
        notes=harness.notes,
        next_steps=[
            f"cd {out.resolve()} && docker compose up -d",
            "pip install -r requirements.txt",
            "SHAKEDOWN_SELECT_STATUS=1 pytest tests/test_contract.py",
            "Use activate_chaos / deactivate_chaos while a client talks to the stubs.",
        ],
    )


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_chaos_scenarios(harness_dir: str) -> list[ScenarioInfo]:
    """List the chaos scenarios of a generated harness."""
    data = _scenarios(harness_dir)
    return [
        ScenarioInfo(id=s["id"], category=s["category"], title=s["title"],
                     expectation=s["expectation"])
        for s in data["scenarios"]
    ]  # fmt: skip


@server.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=True))
async def activate_chaos(
    harness_dir: str, scenario_id: str, wiremock_url: str = DEFAULT_STUB_URL
) -> ChaosReport:
    """Turn one chaos scenario on in the harness's running WireMock (others go off)."""
    data = _scenarios(harness_dir)
    try:
        count = await asyncio.to_thread(chaos_control.activate, wiremock_url, data, scenario_id)
    except chaos_control.ChaosControlError as exc:
        raise ToolError(str(exc)) from exc
    return ChaosReport(wiremock_url=wiremock_url, active=scenario_id, mappings_added=count)


@server.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=True))
async def deactivate_chaos(wiremock_url: str = DEFAULT_STUB_URL) -> ChaosReport:
    """Turn all chaos scenarios off: the stubs behave normally again."""
    try:
        await asyncio.to_thread(chaos_control.deactivate, wiremock_url)
    except chaos_control.ChaosControlError as exc:
        raise ToolError(str(exc)) from exc
    return ChaosReport(wiremock_url=wiremock_url, active=None)


def _load(spec: str) -> Any:
    try:
        return load_spec(spec)
    except SpecError as exc:
        raise ToolError(str(exc)) from exc


def _scenarios(harness_dir: str) -> dict[str, Any]:
    try:
        return chaos_control.load_scenarios(Path(harness_dir).expanduser())
    except chaos_control.ChaosControlError as exc:
        raise ToolError(str(exc)) from exc


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per LLM call
    server.run()  # stdio


if __name__ == "__main__":
    main()
