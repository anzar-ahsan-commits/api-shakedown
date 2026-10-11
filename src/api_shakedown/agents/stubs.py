"""Agent that generates WireMock stub mappings from the spec.

Request matching comes from code (`api_shakedown.wiremock`). The LLM writes realistic
response bodies, which are validated against the spec's response schemas; problems are
fed back for another attempt, and anything still invalid falls back to a schema sample.
Without a model, the agent is fully deterministic.
"""

from dataclasses import dataclass
from typing import Any

from agentpanel import AgentContext
from agentpanel.llm_agents import render_feedback
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from api_shakedown.agents._common import generate_checked
from api_shakedown.spec import ApiSpec, Operation, chunk_operations
from api_shakedown.spec.chunking import DEFAULT_MAX_CHARS
from api_shakedown.spec.models import Response
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import (
    ResponseContent,
    ResponseKey,
    StubSet,
    build_stub_set,
    dataset_content,
    expected_responses,
)

INSTRUCTIONS = """\
You generate realistic mock HTTP response bodies for WireMock stubs, from an OpenAPI \
description of each operation.

Rules:
- Every body must strictly conform to its response schema: required fields, types, \
enums, formats, and min/max constraints.
- Use realistic, internally consistent data: reuse the same ids and names across \
related operations, and avoid placeholders like "string" or "foo".
- Error bodies should read like real errors for their status code.
- Give plausible values for documented response headers.
- Use null as the body when a response has no content."""


class GeneratedResponse(BaseModel):
    operation_id: str
    status: str
    body: Any = Field(default=None, description="JSON value matching the response schema.")
    headers: dict[str, str] = Field(default_factory=dict)


class GeneratedResponses(BaseModel):
    responses: list[GeneratedResponse]


@dataclass
class StubAgent:
    model: BaseChatModel | None = None
    name: str = "stubs"
    kind: str = "WireMock stub mappings"
    max_chars: int = DEFAULT_MAX_CHARS
    max_attempts: int = 3

    async def produce(self, ctx: AgentContext) -> StubSet:
        spec: ApiSpec = ctx.inputs["spec"]
        dataset: Dataset | None = ctx.inputs.get("dataset")
        contents: dict[ResponseKey, ResponseContent] = {}
        notes: list[str] = []
        if self.model is not None:
            feedback = ctx.feedback_for(self.name)
            for chunk in chunk_operations(spec.operations, self.max_chars):
                expected = _needs_llm(chunk, dataset)
                if expected:
                    contents |= await self._generate(
                        self.model, spec, chunk, expected, render_feedback(feedback), notes
                    )
            notes += [
                f"{op.id} {response.status}: used a schema-generated sample body."
                for op, response in expected_responses(spec.operations)
                if (op.id, response.status) not in contents
                and dataset_content(response, dataset) is None
            ]
        return build_stub_set(spec, contents, notes, dataset)

    async def _generate(
        self,
        model: BaseChatModel,
        spec: ApiSpec,
        chunk: list[Operation],
        expected: dict[ResponseKey, Response],
        feedback: str,
        notes: list[str],
    ) -> dict[ResponseKey, ResponseContent]:
        output = await generate_checked(
            model,
            INSTRUCTIONS,
            _prompt(spec, chunk, expected, feedback),
            GeneratedResponses,
            check=lambda out: _problems(spec, expected, out),
            max_attempts=self.max_attempts,
            notes=notes,
        )
        usable: dict[ResponseKey, ResponseContent] = {}
        for item in output.responses if output else []:
            key = (item.operation_id, item.status)
            response = expected.get(key)
            if response is None or (response.body is not None and item.body is None):
                continue  # unknown entry, or a null body where the sample is better
            if not _item_problems(spec, response, item):
                usable[key] = ResponseContent(body=item.body, headers=item.headers, source="llm")
        return usable


def _needs_llm(chunk: list[Operation], dataset: Dataset | None) -> dict[ResponseKey, Response]:
    """Responses the LLM should write: everything the dataset does not already cover."""
    return {
        (op.id, r.status): r
        for op, r in expected_responses(chunk)
        if dataset_content(r, dataset) is None
    }


def _prompt(
    spec: ApiSpec, chunk: list[Operation], expected: dict[ResponseKey, Response], feedback: str
) -> str:
    entries = []
    for (op_id, status), response in expected.items():
        body = response.body
        what = f"{body.content_type}" if body else "no body: use null"
        entries.append(f"- operation_id={op_id} status={status} ({what})")
    prompt = (
        f"API: {spec.title} {spec.version}\n\n"
        "<operations>\n" + "\n".join(op.to_prompt() for op in chunk) + "\n</operations>\n\n"
        "Produce exactly one entry for each of these responses:\n" + "\n".join(entries)
    )
    if feedback:
        prompt += f"\n\nReviewers rejected the previous stubs. Address this feedback:\n{feedback}"
    return prompt


def _problems(
    spec: ApiSpec, expected: dict[ResponseKey, Response], output: GeneratedResponses
) -> list[str]:
    produced = {(item.operation_id, item.status): item for item in output.responses}
    problems = [
        f"Missing entry for operation_id={op_id} status={status}."
        for op_id, status in expected
        if (op_id, status) not in produced
    ]
    for key, item in produced.items():
        response = expected.get(key)
        if response is not None:
            problems += [f"{key[0]} {key[1]}: {p}" for p in _item_problems(spec, response, item)]
    return problems


def _item_problems(spec: ApiSpec, response: Response, item: GeneratedResponse) -> list[str]:
    if response.body is None or response.body.schema_ is None:
        return []
    if item.body is None:
        return ["body is null but the response has a schema."]
    return schema_errors(response.body.schema_, item.body, spec.openapi_version)
