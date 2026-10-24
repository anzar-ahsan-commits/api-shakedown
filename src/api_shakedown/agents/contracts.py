"""Agent that writes contract tests against the spec.

The LLM proposes test cases as data; code validates each against the spec, feeds problems
back, and guarantees a happy-path case per operation (falling back to a schema-derived
case). The cases run under a fixed pytest runner, so no generated code is executed.
Without a model, the agent emits only the deterministic happy-path cases.
"""

from dataclasses import dataclass

from agentpanel import AgentContext
from agentpanel.llm_agents import render_feedback
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from api_shakedown.agents._common import Progress, batches, generate_checked
from api_shakedown.contracts import (
    CaseProposal,
    ContractSuite,
    build_suite,
    happy_proposal,
    proposal_problems,
)
from api_shakedown.seeding import seed_plan
from api_shakedown.spec import ApiSpec, Operation, chunk_operations
from api_shakedown.spec.chunking import DEFAULT_MAX_CHARS
from api_shakedown.testdata import Dataset

MAX_CASES_PER_OPERATION = 8

INSTRUCTIONS = """\
You design HTTP contract test cases from an OpenAPI description.

For each operation, propose:
- one happy-path case with realistic, valid inputs;
- one case per documented error status where the inputs would plausibly cause it, \
e.g. a well-formed but unknown id for 404, or a body missing a required field for 400;
- boundary cases for constrained inputs (minimum/maximum values, enum members).

Rules:
- expected_status must be one of the operation's documented statuses.
- Path parameters must always be valid values for their schema, even in error cases, \
so the URL still routes.
- Always include required query parameters.
- Success cases must send inputs that satisfy every schema.
- Never add authentication headers; they are added automatically.
- Give each case a short snake_case name describing what it checks."""


class CaseProposals(BaseModel):
    cases: list[CaseProposal]


@dataclass
class ContractTestAgent:
    model: BaseChatModel | None = None
    name: str = "contracts"
    kind: str = "contract test suite"
    max_chars: int = DEFAULT_MAX_CHARS
    max_attempts: int = 3
    progress: Progress | None = None

    async def produce(self, ctx: AgentContext) -> ContractSuite:
        spec: ApiSpec = ctx.inputs["spec"]
        dataset: Dataset | None = ctx.inputs.get("dataset")
        notes: list[str] = []
        proposals: list[CaseProposal] = []
        if self.model is not None:
            feedback = render_feedback(ctx.feedback_for(self.name))
            chunks = chunk_operations(spec.operations, self.max_chars)
            for chunk in batches(chunks, self.name, self.progress):
                proposals += await self._generate(self.model, spec, dataset, chunk, feedback, notes)
        cases = _with_happy_paths(spec, dataset, proposals, notes, self.model)
        suite = build_suite(spec, cases, notes)
        if dataset is not None:
            suite.setup, seed_notes = seed_plan(spec, dataset)
            suite.notes += seed_notes
        return suite

    async def _generate(
        self,
        model: BaseChatModel,
        spec: ApiSpec,
        dataset: Dataset | None,
        chunk: list[Operation],
        feedback: str,
        notes: list[str],
    ) -> list[CaseProposal]:
        ops = {op.id: op for op in chunk}
        output = await generate_checked(
            model,
            INSTRUCTIONS,
            _prompt(spec, dataset, chunk, feedback),
            CaseProposals,
            check=lambda out: _problems(spec, dataset, ops, out),
            max_attempts=self.max_attempts,
            notes=notes,
        )
        usable: list[CaseProposal] = []
        per_op: dict[str, int] = {}
        for proposal in output.cases if output else []:
            op = ops.get(proposal.operation_id)
            if op is None or proposal_problems(spec, op, proposal, dataset):
                continue
            if per_op.get(op.id, 0) < MAX_CASES_PER_OPERATION:
                per_op[op.id] = per_op.get(op.id, 0) + 1
                usable.append(proposal)
        return usable


def _with_happy_paths(
    spec: ApiSpec,
    dataset: Dataset | None,
    proposals: list[CaseProposal],
    notes: list[str],
    model: BaseChatModel | None,
) -> list[CaseProposal]:
    covered = {p.operation_id for p in proposals if p.expected_status.startswith("2")}
    result = list(proposals)
    for op in spec.operations:
        if op.id in covered:
            continue
        fallback = happy_proposal(op, dataset, spec.openapi_version)
        if fallback is not None:
            result.append(fallback)
            if model is not None:
                notes.append(f"{op.id}: used a schema-generated happy-path case.")
    order = {op.id: i for i, op in enumerate(spec.operations)}
    return sorted(result, key=lambda p: order[p.operation_id])  # stable within an operation


def _prompt(spec: ApiSpec, dataset: Dataset | None, chunk: list[Operation], feedback: str) -> str:
    statuses = "\n".join(
        f"- {op.id}: documented statuses {[r.status for r in op.responses]}" for op in chunk
    )
    prompt = (
        f"API: {spec.title} {spec.version}\n\n"
        "<operations>\n" + "\n".join(op.to_prompt() for op in chunk) + "\n</operations>\n\n"
        f"Propose test cases for these operations:\n{statuses}"
    )
    known = _known_ids(dataset, chunk)
    if known:
        prompt += (
            "\n\nThese ids exist in the test data. Success cases must use one of them; "
            "404 cases must use a well-formed id that is NOT listed:\n" + "\n".join(known)
        )
    if feedback:
        prompt += f"\n\nReviewers rejected the previous tests. Address this feedback:\n{feedback}"
    return prompt


def _known_ids(dataset: Dataset | None, chunk: list[Operation]) -> list[str]:
    lines = []
    for op in chunk:
        for param in (p for p in op.parameters if p.location == "path"):
            ids = dataset.known_ids(op.path, param.name) if dataset else None
            if ids:
                lines.append(f"- {op.id} {param.name}: {ids}")
    return lines


def _problems(
    spec: ApiSpec, dataset: Dataset | None, ops: dict[str, Operation], output: CaseProposals
) -> list[str]:
    problems = []
    for proposal in output.cases:
        label = f"case {proposal.operation_id}/{proposal.name}"
        op = ops.get(proposal.operation_id)
        if op is None:
            problems.append(f"{label}: unknown operation_id; use one of {sorted(ops)}.")
            continue
        problems += [f"{label}: {p}" for p in proposal_problems(spec, op, proposal, dataset)]
    happy = {
        p.operation_id
        for p in output.cases
        if p.expected_status.startswith("2")
        and p.operation_id in ops
        and not proposal_problems(spec, ops[p.operation_id], p, dataset)
    }
    problems += [
        f"No valid happy-path (2xx) case for {op_id}." for op_id in ops if op_id not in happy
    ]
    return problems
