"""Command-line entry point (`shakedown`)."""

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from agentpanel.config import Provider
from agentpanel.llm import check_provider

from api_shakedown import __version__
from api_shakedown.config import ShakedownSettings
from api_shakedown.harness import write_harness
from api_shakedown.pipeline import PipelineOptions, generate_harness
from api_shakedown.spec import SpecError, chunk_operations, load_spec

app = typer.Typer(help="Generate a chaos test harness from an OpenAPI spec.", no_args_is_help=True)

ProviderOption = Annotated[
    Provider | None, typer.Option(help="LLM provider. Overrides SHAKEDOWN_PROVIDER.")
]
ModelOption = Annotated[str | None, typer.Option(help="Model name. Overrides SHAKEDOWN_MODEL.")]


def _settings(provider: Provider | None, model: str | None) -> ShakedownSettings:
    candidates: dict[str, Any] = {"provider": provider, "model": model}
    overrides = {k: v for k, v in candidates.items() if v is not None}
    return ShakedownSettings(**overrides)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def doctor(provider: ProviderOption = None, model: ModelOption = None) -> None:
    """Check that the configured LLM provider is reachable and ready."""
    status = check_provider(_settings(provider, model))
    mark = "OK" if status.ok else "FAIL"
    typer.echo(f"[{mark}] {status.provider} / {status.model}: {status.detail}")
    if not status.ok:
        raise typer.Exit(code=1)


@app.command()
def inspect(
    spec: Annotated[str, typer.Argument(help="Path or URL to an OpenAPI spec.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the normalized spec.")] = False,
) -> None:
    """Show what the agents will see: operations, chunks, and spec warnings."""
    try:
        api = load_spec(spec)
    except SpecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        typer.echo(api.model_dump_json(indent=2, by_alias=True, exclude_none=True))
        return
    chunks = chunk_operations(api.operations)
    typer.echo(
        f"{api.title} {api.version} (OpenAPI {api.openapi_version}): "
        f"{len(api.operations)} operations in {len(chunks)} chunk(s)"
    )
    for op in api.operations:
        tags = f"  [{', '.join(op.tags)}]" if op.tags else ""
        typer.echo(f"  {op.method.upper():7} {op.path:30} {op.id}{tags}")
    for warning in api.warnings:
        typer.echo(f"Warning: {warning}", err=True)


@app.command()
def run(
    spec: Annotated[str, typer.Argument(help="Path or URL to an OpenAPI spec.")],
    out: Annotated[Path, typer.Option(help="Directory to write the harness to.")] = Path(
        "shakedown-output"
    ),
    offline: Annotated[
        bool, typer.Option(help="No LLM: deterministic stubs, tests, and scenarios.")
    ] = False,
    review: Annotated[bool, typer.Option(help="Have the reviewer agent check artifacts.")] = True,
    max_revisions: Annotated[int, typer.Option(min=0, help="Revision rounds after review.")] = 1,
    force: Annotated[bool, typer.Option(help="Write into a non-empty directory.")] = False,
    provider: ProviderOption = None,
    model: ModelOption = None,
) -> None:
    """Generate a ready-to-run chaos test harness for SPEC."""
    try:
        api = load_spec(spec)
    except SpecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if out.exists() and any(out.iterdir()) and not force:
        typer.echo(f"Error: {out} is not empty; use --force to overwrite.", err=True)
        raise typer.Exit(code=2)

    settings = _settings(provider, model)
    if not offline:
        status = check_provider(settings)
        if not status.ok:
            typer.echo(f"[FAIL] {status.provider} / {status.model}: {status.detail}", err=True)
            typer.echo("Fix that, or use --offline for a deterministic harness.", err=True)
            raise typer.Exit(code=1)
    for warning in api.warnings:
        typer.echo(f"Warning: {warning}", err=True)

    mode = "offline" if offline else f"{settings.provider} / {settings.resolved_model}"
    typer.echo(f"{api.title} {api.version}: {len(api.operations)} operations ({mode})")
    started = time.monotonic()
    options = PipelineOptions(
        offline=offline,
        review=review and not offline,
        max_revisions=max_revisions,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    harness = asyncio.run(generate_harness(api, settings, options, progress=typer.echo))
    write_harness(harness, out, force=force)

    s = harness.summary
    typer.echo(
        f"\nWrote {len(harness.files)} files to {out} in {time.monotonic() - started:.0f}s: "
        f"{s['stub_mappings']} stubs, {s['contract_cases']} contract cases, "
        f"{s['chaos_scenarios']} chaos scenarios, {s['robustness_probes']} probes."
    )
    if harness.unresolved:
        typer.echo(f"Still rejected by review: {', '.join(harness.unresolved)} (see README).")
    typer.echo(f"Next: cd {out} && docker compose up -d   (then see README.md)")
