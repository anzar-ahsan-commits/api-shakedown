"""Coordinator agent that assembles reviewed artifacts into the final harness.

Assembly is deterministic: every file comes from an artifact or a fixed template, and the
README is generated from what is actually in the harness. Review outcomes are reported,
not hidden: anything still rejected when the revision budget ran out is listed.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentpanel import AgentContext, Review
from pydantic import BaseModel, Field

from api_shakedown.chaos import CHAOS_CLI_FILE, SCENARIOS_FILE, ChaosSuite
from api_shakedown.contracts import ContractSuite
from api_shakedown.spec import ApiSpec
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import DEFAULT_STUB_URL, STATUS_HEADER, StubSet, base_path

WIREMOCK_IMAGE = "wiremock/wiremock:3.13.1"
TEST_REQUIREMENTS = ["pytest>=8", "httpx>=0.27", "openapi-schema-validator>=0.6"]


class Harness(BaseModel):
    title: str
    version: str
    files: dict[str, str]  # relative path -> content
    summary: dict[str, int]
    notes: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)  # final-round outcome per artifact
    unresolved: list[str] = Field(default_factory=list)


@dataclass
class HarnessCoordinator:
    generated_at: str = ""  # shown in the README; empty keeps output reproducible

    async def assemble(self, ctx: AgentContext) -> Harness:
        spec: ApiSpec = ctx.inputs["spec"]
        dataset: Dataset | None = ctx.inputs.get("dataset")
        stubs: StubSet = ctx.artifacts["stubs"].content
        contracts: ContractSuite = ctx.artifacts["contracts"].content
        chaos: ChaosSuite = ctx.artifacts["chaos"].content
        return build_harness(
            spec, dataset, stubs, contracts, chaos, ctx.reviews, ctx.unresolved, self.generated_at
        )


def build_harness(
    spec: ApiSpec,
    dataset: Dataset | None,
    stubs: StubSet,
    contracts: ContractSuite,
    chaos: ChaosSuite,
    reviews: Sequence[Review] = (),
    unresolved: list[str] | None = None,
    generated_at: str = "",
) -> Harness:
    files: dict[str, str] = {
        "docker-compose.yml": _compose(),
        "wiremock/mappings/stubs.json": _json(stubs.to_wiremock()),
        "requirements.txt": "\n".join(TEST_REQUIREMENTS) + "\n",
        "tests/pytest.ini": "[pytest]\n",
    }
    files |= {f"tests/{name}": content for name, content in contracts.files().items()}
    for name, content in chaos.files().items():
        folder = "chaos" if name in (CHAOS_CLI_FILE, SCENARIOS_FILE) else "tests"
        files[f"{folder}/{name}"] = content
    if dataset is not None:
        files["data/dataset.json"] = _json(dataset.model_dump(by_alias=True))

    final_reviews = _final_reviews(reviews)
    summary = {
        "operations": len(spec.operations),
        "stub_mappings": len(stubs.mappings),
        "contract_cases": len(contracts.cases),
        "seed_records": len(contracts.setup),
        "chaos_scenarios": len(chaos.scenarios),
        "robustness_probes": len(chaos.probes),
        "records": sum(len(e.records) for e in dataset.entities.values()) if dataset else 0,
    }
    notes = [
        *(f"testdata: {n}" for n in (dataset.notes if dataset else [])),
        *(f"stubs: {n}" for n in stubs.notes),
        *(f"contracts: {n}" for n in contracts.notes),
        *(f"chaos: {n}" for n in chaos.notes),
    ]
    review_lines = [
        f"{r.target}: {'approved' if r.approved else 'REJECTED'}"
        + "".join(f"\n  - {c}" for c in r.comments)
        for r in final_reviews
    ]
    harness = Harness(
        title=spec.title,
        version=spec.version,
        files=files,
        summary=summary,
        notes=notes,
        reviews=review_lines,
        unresolved=list(unresolved or []),
    )
    harness.files["README.md"] = _readme(spec, dataset, chaos, harness, generated_at)
    harness.files["shakedown.json"] = _json(
        harness.model_dump(exclude={"files"}) | {"generated_at": generated_at or None}
    )
    return harness


def _final_reviews(reviews: Sequence[Review]) -> list[Review]:
    """Each artifact's latest review (merging comments from all its reviewers)."""
    latest: dict[str, Review] = {}
    for review in reviews:
        current = latest.get(review.target)
        if current is None or review.round > current.round:
            latest[review.target] = review
        elif review.round == current.round:
            latest[review.target] = current.model_copy(
                update={
                    "approved": current.approved and review.approved,
                    "comments": [*current.comments, *review.comments],
                }
            )
    return [latest[k] for k in sorted(latest)]


def _compose() -> str:
    return f"""\
# The virtual service: WireMock serving the generated stubs.
# Port: SHAKEDOWN_WIREMOCK_PORT (default 8080).
services:
  virtual-service:
    image: {WIREMOCK_IMAGE}
    ports:
      - "${{SHAKEDOWN_WIREMOCK_PORT:-8080}}:8080"
    volumes:
      - ./wiremock:/home/wiremock
"""


def _readme(
    spec: ApiSpec,
    dataset: Dataset | None,
    chaos: ChaosSuite,
    harness: Harness,
    generated_at: str,
) -> str:
    target = DEFAULT_STUB_URL + base_path(spec)
    s = harness.summary
    when = f" on {generated_at}" if generated_at else ""
    seeding = (
        f", plus {s['seed_records']} test records to create first with `SHAKEDOWN_SEED=1`"
        if s.get("seed_records")
        else ""
    )
    scenario_rows = "\n".join(
        f"| `{sc.id}` | {sc.category} | {sc.title} | {sc.expectation} |" for sc in chaos.scenarios
    )
    sections = [
        f"# Chaos test harness: {spec.title} {spec.version}",
        f"Generated by API Shakedown{when} from the OpenAPI spec.",
        "## Quick start",
        "```bash\n"
        "docker compose up -d                 # WireMock with the stubs on :8080\n"
        "pip install -r requirements.txt\n\n"
        "# Contract tests against the stubs (the header asks the stubs for each status)\n"
        "SHAKEDOWN_SELECT_STATUS=1 pytest tests/test_contract.py\n\n"
        "# Switch failure scenarios on and off while your client talks to the stubs\n"
        "python chaos/chaos.py list\n"
        "python chaos/chaos.py on flaky-start\n"
        "python chaos/chaos.py off\n\n"
        "# Against a real deployment: contract tests and robustness probes\n"
        "SHAKEDOWN_TARGET_URL=https://api.example.com/v1 pytest tests\n\n"
        "# Same, creating the test records first (writes to the server: use a test instance)\n"
        "SHAKEDOWN_SEED=1 SHAKEDOWN_TARGET_URL=https://api.example.com/v1 pytest tests\n"
        "```",
        'PowerShell: `$env:SHAKEDOWN_SELECT_STATUS = "1"; pytest tests/test_contract.py`.',
        f"The stubs serve `{target}`. Use another port with `SHAKEDOWN_WIREMOCK_PORT`, then "
        "point `SHAKEDOWN_TARGET_URL` and `chaos.py --wiremock` at it.",
        "## What's inside",
        "| | |\n|---|---|\n"
        f"| `wiremock/mappings/` | {s['stub_mappings']} stub mappings for {s['operations']} "
        "operations |\n"
        f"| `tests/test_contract.py` | {s['contract_cases']} contract test cases{seeding} |\n"
        f"| `tests/test_robustness.py` | {s['robustness_probes']} malformed-request probes "
        "(expect 4xx, never 5xx) |\n"
        f"| `chaos/chaos.py` | {s['chaos_scenarios']} failure scenarios to inject into the "
        "stubs |\n"
        + (
            f"| `data/dataset.json` | {s['records']} test records and edge-case values |\n"
            if dataset
            else ""
        ),
        "## How the stubs behave",
        "- Each operation returns its success response; with test data, known ids return "
        "their record and unknown ids return 404.\n"
        f"- Any other documented response is returned when the request carries "
        f"`{STATUS_HEADER}: <status>`, e.g. `{STATUS_HEADER}: 404`.\n"
        "- An active chaos scenario overrides everything until `chaos.py off`.",
        "## Chaos scenarios",
        "| Scenario | Category | What happens | A resilient client should |\n"
        "|---|---|---|---|\n" + scenario_rows,
    ]
    if harness.reviews or harness.unresolved:
        review = "\n".join(f"- {line}" for line in harness.reviews)
        if harness.unresolved:
            review += (
                "\n\nStill rejected when the revision budget ran out: "
                + ", ".join(harness.unresolved)
                + ". Check those parts before relying on them."
            )
        sections += ["## Review", review]
    if harness.notes:
        sections += ["## Generation notes", "\n".join(f"- {n}" for n in harness.notes)]
    return "\n\n".join(sections) + "\n"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
