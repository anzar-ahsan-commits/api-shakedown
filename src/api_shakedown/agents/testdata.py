"""Agent that builds the shared test dataset.

The LLM writes realistic, varied records for each entity the API serves; code validates
every record against its schema, enforces unique ids, and falls back to deterministic
records where needed. Edge-case values come from code alone. The dataset is meant to run
first and be passed to the other agents as `inputs["dataset"]`.
"""

import json
from dataclasses import dataclass
from typing import Any

from agentpanel import AgentContext
from agentpanel.llm_agents import render_feedback
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from api_shakedown.agents._common import Progress, batches, generate_checked
from api_shakedown.spec import ApiSpec
from api_shakedown.spec.chunking import DEFAULT_MAX_CHARS
from api_shakedown.testdata import (
    DEFAULT_RECORDS,
    Dataset,
    Entity,
    build_dataset,
    find_entities,
    record_problems,
)

MAX_RECORDS = 10

INSTRUCTIONS = """\
You create realistic test data for an API: records for each entity, matching its JSON \
schema exactly.

Rules:
- Every record must satisfy the schema: required fields, types, enums, formats, and \
min/max constraints.
- Records must have unique ids.
- Make records realistic and varied: different enum values, plausible names, dates, \
and amounts. Avoid placeholders like "string", "foo", or "test".
- Keep data consistent across entities that refer to each other."""


class EntityRecords(BaseModel):
    entity: str
    records: list[dict[str, Any]]


class GeneratedDataset(BaseModel):
    entities: list[EntityRecords] = Field(description="One entry per requested entity.")


@dataclass
class DatasetAgent:
    model: BaseChatModel | None = None
    name: str = "testdata"
    kind: str = "test dataset"
    records_per_entity: int = DEFAULT_RECORDS
    max_chars: int = DEFAULT_MAX_CHARS
    max_attempts: int = 3
    progress: Progress | None = None

    async def produce(self, ctx: AgentContext) -> Dataset:
        spec: ApiSpec = ctx.inputs["spec"]
        entities = find_entities(spec)
        notes: list[str] = []
        if self.model is not None and entities:
            feedback = render_feedback(ctx.feedback_for(self.name))
            chosen = _worth_generating(list(entities.values()), self.max_chars, notes)
            for group in batches(_groups(chosen, self.max_chars), self.name, self.progress):
                await self._generate(self.model, spec, group, feedback, notes)
            notes += [
                f"{e.name}: used deterministic sample records." for e in chosen if not e.records
            ]
        return build_dataset(spec, entities, notes)

    async def _generate(
        self,
        model: BaseChatModel,
        spec: ApiSpec,
        group: list[Entity],
        feedback: str,
        notes: list[str],
    ) -> None:
        count = max(1, min(self.records_per_entity, MAX_RECORDS))
        by_name = {e.name: e for e in group}
        output = await generate_checked(
            model,
            INSTRUCTIONS,
            _prompt(spec, group, count, feedback),
            GeneratedDataset,
            check=lambda out: _problems(spec, by_name, out),
            max_attempts=self.max_attempts,
            notes=notes,
        )
        for item in output.entities if output else []:
            entity = by_name.get(item.entity)
            if entity is None:
                continue
            valid: list[dict[str, Any]] = []
            for record in item.records[:MAX_RECORDS]:
                if not record_problems(entity, [*valid, record], spec.openapi_version):
                    valid.append(record)
            entity.records = valid


def _worth_generating(entities: list[Entity], max_chars: int, notes: list[str]) -> list[Entity]:
    """Resources (schemas with a key) small enough for one prompt.

    Keyless schemas are response envelopes (search results, status objects) rather than
    data, and a single huge schema stalls a small model; both get valid schema samples.
    """
    envelopes = sorted(e.name for e in entities if e.id_field is None)
    oversized = [e for e in entities if e.id_field and len(json.dumps(e.schema_)) > max_chars]
    if envelopes:
        shown = ", ".join(envelopes[:4]) + (", ..." if len(envelopes) > 4 else "")
        notes.append(
            f"{len(envelopes)} response schemas without a key ({shown}) use schema samples "
            "rather than generated records."
        )
    notes += [
        f"{e.name}: schema too large for one prompt ({len(json.dumps(e.schema_))} chars); "
        "used schema samples."
        for e in oversized
    ]
    return [e for e in entities if e.id_field and e not in oversized]


def _groups(entities: list[Entity], max_chars: int) -> list[list[Entity]]:
    groups: list[list[Entity]] = []
    size = 0
    for entity in entities:
        entity_size = len(json.dumps(entity.schema_))
        if not groups or size + entity_size > max_chars:
            groups.append([])
            size = 0
        groups[-1].append(entity)
        size += entity_size
    return groups


def _prompt(spec: ApiSpec, group: list[Entity], count: int, feedback: str) -> str:
    schemas = "\n".join(
        f'<entity name="{e.name}" id_field="{e.id_field}">\n{json.dumps(e.schema_)}\n</entity>'
        for e in group
    )
    prompt = (
        f"API: {spec.title} {spec.version}\n\n{schemas}\n\n"
        f"Create {count} records for each of: {', '.join(e.name for e in group)}."
    )
    if feedback:
        prompt += f"\n\nReviewers rejected the previous dataset. Address this feedback:\n{feedback}"
    return prompt


def _problems(spec: ApiSpec, entities: dict[str, Entity], output: GeneratedDataset) -> list[str]:
    produced = {item.entity: item for item in output.entities}
    problems = [f"Missing records for entity {name}." for name in entities if name not in produced]
    for name, item in produced.items():
        entity = entities.get(name)
        if entity is None:
            problems.append(f"Unknown entity {name!r}; use one of {sorted(entities)}.")
        elif not item.records:
            problems.append(f"No records for {name}.")
        else:
            problems += record_problems(entity, item.records, spec.openapi_version)
    return problems
