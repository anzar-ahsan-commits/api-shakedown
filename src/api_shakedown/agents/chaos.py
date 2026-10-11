"""Agent that invents failure scenarios: timeouts, malformed payloads, partial outages.

Baseline scenarios and robustness probes come from code. The LLM adds scenarios aimed at
this particular API (which calls are critical, which can be retried, where partial
failure hurts most), expressed in the constrained `ScenarioProposal` format and checked
before being compiled to WireMock mappings.
"""

from dataclasses import dataclass

from agentpanel import AgentContext
from agentpanel.llm_agents import render_feedback
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from api_shakedown.agents._common import generate_checked
from api_shakedown.chaos import (
    ChaosSuite,
    ScenarioProposal,
    baseline_scenarios,
    build_probes,
    compile_scenario,
    scenario_problems,
)
from api_shakedown.spec import ApiSpec
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import DEFAULT_STUB_URL, base_path

MAX_LLM_SCENARIOS = 10

INSTRUCTIONS = """\
You are a chaos engineer. You design failure scenarios, injected into a mock of an API, \
that test how CLIENTS of that API cope.

Propose scenarios specific to this API, for example:
- partial outages: writes fail while reads still work, or one critical endpoint is down;
- a slow dependency on the most-used read;
- a create call that fails twice then succeeds (tests retries and idempotency);
- schema drift (wrong_types, missing_required) in an important response;
- rate limiting on list endpoints.

Rules:
- Do not repeat the baseline scenarios listed in the prompt.
- Each effect uses: delay_ms, fault, status (400-599), or malformation. fault cannot be \
combined with status or malformation; status and malformation are exclusive.
- operations must list the affected operation ids (never empty: the baseline already \
covers every operation). A scenario about one endpoint lists only that endpoint.
- Use "times" for failures that recover after N calls; null means always while active. \
A flaky scenario must set times.
- Match effect to category: latency/timeout need delay_ms; rate_limit uses status 429; \
flaky, outage and partial_outage use a 5xx/429 status or a fault (a 4xx is not a \
transient failure); a partial_outage leaves some operations working; malformed_response \
needs a malformation.
- expectation says concretely what a resilient client should do."""


class ScenarioProposals(BaseModel):
    scenarios: list[ScenarioProposal]


@dataclass
class ChaosAgent:
    model: BaseChatModel | None = None
    name: str = "chaos"
    kind: str = "chaos scenarios and robustness probes"
    max_attempts: int = 3

    async def produce(self, ctx: AgentContext) -> ChaosSuite:
        spec: ApiSpec = ctx.inputs["spec"]
        dataset: Dataset | None = ctx.inputs.get("dataset")
        notes: list[str] = []
        baseline = baseline_scenarios(spec)
        proposals: list[ScenarioProposal] = []
        if self.model is not None:
            feedback = render_feedback(ctx.feedback_for(self.name))
            proposals = await self._generate(self.model, spec, baseline, feedback, notes)

        scenarios = [compile_scenario(spec, p, "baseline", dataset) for p in baseline]
        scenarios += [compile_scenario(spec, p, "llm", dataset) for p in proposals]
        return ChaosSuite(
            title=spec.title,
            version=spec.version,
            default_target=DEFAULT_STUB_URL + base_path(spec),
            scenarios=scenarios,
            probes=build_probes(spec, dataset),
            notes=notes,
        )

    async def _generate(
        self,
        model: BaseChatModel,
        spec: ApiSpec,
        baseline: list[ScenarioProposal],
        feedback: str,
        notes: list[str],
    ) -> list[ScenarioProposal]:
        taken = {p.name for p in baseline}
        output = await generate_checked(
            model,
            INSTRUCTIONS,
            _prompt(spec, baseline, feedback),
            ScenarioProposals,
            check=lambda out: _problems(spec, taken, out),
            max_attempts=self.max_attempts,
            notes=notes,
        )
        usable: list[ScenarioProposal] = []
        for proposal in output.scenarios if output else []:
            if proposal.name in taken or scenario_problems(spec, proposal):
                continue
            taken.add(proposal.name)
            usable.append(proposal)
        return usable[:MAX_LLM_SCENARIOS]


def _prompt(spec: ApiSpec, baseline: list[ScenarioProposal], feedback: str) -> str:
    operations = "\n".join(
        f"- {op.id}: {op.method.upper()} {op.path}"
        + (f" ({op.summary})" if op.summary else "")
        + f" statuses {[r.status for r in op.responses]}"
        for op in spec.operations
    )
    existing = "\n".join(f"- {p.name}: {p.title}" for p in baseline)
    prompt = (
        f"API: {spec.title} {spec.version}\n\nOperations:\n{operations}\n\n"
        f"Baseline scenarios (already included, do not repeat):\n{existing}\n\n"
        "Propose 3 to 6 additional scenarios targeted at this API."
    )
    if feedback:
        prompt += (
            f"\n\nReviewers rejected the previous scenarios. Address this feedback:\n{feedback}"
        )
    return prompt


def _problems(spec: ApiSpec, taken: set[str], output: ScenarioProposals) -> list[str]:
    problems = []
    seen = set(taken)
    if not output.scenarios:
        problems.append("Propose at least one scenario.")
    for proposal in output.scenarios:
        if proposal.name in seen:
            problems.append(f"scenario {proposal.name}: duplicate or baseline name.")
        seen.add(proposal.name)
        problems += [f"scenario {proposal.name}: {p}" for p in scenario_problems(spec, proposal)]
    return problems
