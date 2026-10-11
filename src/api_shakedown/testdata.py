"""Test datasets: realistic records per entity, linked to path parameters, plus edge cases.

An *entity* is a named component schema that some operation returns on success (directly
or as array items), e.g. `Pet`. A *binding* links a path parameter to an entity's id
field, e.g. `/pets/{petId}` -> `Pet.id`, so stubs and tests can agree on which ids exist.
"""

import copy
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from api_shakedown.spec.edge_cases import EdgeCase, object_edge_cases
from api_shakedown.spec.edge_cases import properties_of as _properties
from api_shakedown.spec.models import ApiSpec, Operation
from api_shakedown.spec.samples import sample_from_schema
from api_shakedown.spec.schema_check import schema_errors

DEFAULT_RECORDS = 3
_ID_PATTERN = re.compile(r"(^id$|_id$|Id$)")
_UUID = re.compile(r"[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")


class Entity(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    id_field: str | None = None
    records: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    def ids(self) -> list[Any]:
        return [r[self.id_field] for r in self.records if self.id_field and self.id_field in r]

    def new_record(self) -> dict[str, Any] | None:
        """What a create call returns: like the first record, with an id not in the data."""
        if not self.records:
            return None
        record = copy.deepcopy(self.records[0])
        if self.id_field:
            ids = self.ids()
            if ids and all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
                record[self.id_field] = max(ids) + 1
            elif ids and all(_UUID.fullmatch(str(i)) for i in ids):
                record[self.id_field] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}/new"))
            else:
                record[self.id_field] = f"{self.name.lower()}-new"
        return record

    def record(self, record_id: Any) -> dict[str, Any] | None:
        return next(
            (r for r in self.records if str(r.get(self.id_field or "")) == str(record_id)), None
        )


class ParamBinding(BaseModel):
    path: str
    param: str
    entity: str
    field: str


class Dataset(BaseModel):
    entities: dict[str, Entity] = Field(default_factory=dict)
    bindings: list[ParamBinding] = Field(default_factory=list)
    edge_cases: dict[str, dict[str, list[EdgeCase]]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def binding(self, path: str, param: str) -> ParamBinding | None:
        return next((b for b in self.bindings if (b.path, b.param) == (path, param)), None)

    def known_ids(self, path: str, param: str) -> list[Any] | None:
        """Ids that exist for a bound path parameter, or None if it is not bound."""
        bound = self.binding(path, param)
        return self.entities[bound.entity].ids() if bound else None

    def entity_for(self, schema: dict[str, Any] | None) -> tuple[Entity, bool] | None:
        """(entity, is_array) if `schema` is an entity or an array of one."""
        for entity in self.entities.values():
            if schema == entity.schema_:
                return entity, False
            if isinstance(schema, dict) and schema.get("items") == entity.schema_:
                return entity, True
        return None


def find_entities(spec: ApiSpec) -> dict[str, Entity]:
    returned = []
    for op in spec.operations:
        for response in op.responses:
            schema = response.body.schema_ if response.body else None
            if response.status.startswith("2") and isinstance(schema, dict):
                returned += [schema, schema.get("items")]
    return {
        name: Entity(name=name, schema=schema, id_field=id_field(schema))
        for name, schema in spec.schemas.items()
        if _is_object(schema) and schema in returned
    }


def find_bindings(spec: ApiSpec, entities: dict[str, Entity]) -> list[ParamBinding]:
    """Bind the last path parameter of each path that returns a single entity."""
    bindings: dict[tuple[str, str], ParamBinding] = {}
    for op in spec.operations:
        entity = _single_entity(op, entities)
        params = re.findall(r"\{([^}]+)\}", op.path)
        if entity is None or entity.id_field is None or not params:
            continue
        bindings[(op.path, params[-1])] = ParamBinding(
            path=op.path, param=params[-1], entity=entity.name, field=entity.id_field
        )
    return list(bindings.values())


def id_field(schema: dict[str, Any]) -> str | None:
    names = list(_properties(schema))
    if "id" in names:
        return "id"
    return next((n for n in names if _ID_PATTERN.search(n)), None)


def record_problems(
    entity: Entity, records: list[dict[str, Any]], openapi_version: str
) -> list[str]:
    problems = []
    seen: set[str] = set()
    for i, record in enumerate(records):
        problems += [
            f"{entity.name} record {i}: {error}"
            for error in schema_errors(entity.schema_, record, openapi_version)
        ]
        if entity.id_field:
            value = record.get(entity.id_field)
            if value is None:
                problems.append(f"{entity.name} record {i}: missing id field {entity.id_field!r}.")
            elif str(value) in seen:
                problems.append(f"{entity.name} record {i}: duplicate id {value!r}.")
            seen.add(str(value))
    return problems


def sample_records(
    entity: Entity, openapi_version: str, count: int = DEFAULT_RECORDS
) -> list[dict[str, Any]]:
    """Deterministic records: schema samples with distinct ids and cycled enum values."""
    base = sample_from_schema(entity.schema_)
    if not isinstance(base, dict):
        return []
    properties = _properties(entity.schema_)
    records = []
    for i in range(count):
        record = copy.deepcopy(base)
        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            candidate = _variant(name, prop, i, entity)
            if candidate is not None and not schema_errors(prop, candidate, openapi_version):
                record[name] = candidate
        records.append(record)
    return records


def build_dataset(
    spec: ApiSpec, entities: dict[str, Entity], notes: list[str] | None = None
) -> Dataset:
    """Assemble a dataset; entities without records get deterministic samples."""
    for entity in entities.values():
        if not entity.records:
            entity.records = sample_records(entity, spec.openapi_version)
    edge_targets = dict(entities.items())
    request_schemas = [op.request_body.schema_ for op in spec.operations if op.request_body]
    for name, schema in spec.schemas.items():
        if name not in edge_targets and _is_object(schema) and schema in request_schemas:
            edge_targets[name] = Entity(name=name, schema=schema)
    return Dataset(
        entities=entities,
        bindings=find_bindings(spec, entities),
        edge_cases={
            name: object_edge_cases(e.schema_, spec.openapi_version)
            for name, e in edge_targets.items()
        },
        notes=list(notes or []),
    )


def _variant(name: str, prop: dict[str, Any], i: int, entity: Entity) -> Any:
    if name == entity.id_field:
        if prop.get("type") in ("integer", "number"):
            return i + 1
        if prop.get("format") == "uuid":
            return f"00000000-0000-4000-8000-{i + 1:012d}"
        return f"{entity.name.lower()}-{i + 1}"
    if prop.get("enum"):
        return prop["enum"][i % len(prop["enum"])]
    if prop.get("type") == "string" and "format" not in prop and "pattern" not in prop:
        return f"{name} {i + 1}"
    return None


def _single_entity(op: Operation, entities: dict[str, Entity]) -> Entity | None:
    for response in op.responses:
        schema = response.body.schema_ if response.body else None
        if response.status.startswith("2"):
            for entity in entities.values():
                if schema == entity.schema_:
                    return entity
    return None


def _is_object(schema: Any) -> bool:
    return isinstance(schema, dict) and (
        schema.get("type") == "object" or "properties" in schema or "allOf" in schema
    )
