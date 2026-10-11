import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentpanel import AgentContext
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.config import ShakedownSettings
from api_shakedown.contracts import CaseProposal, happy_proposal, proposal_problems
from api_shakedown.spec import load_spec
from api_shakedown.spec.edge_cases import edge_cases
from api_shakedown.spec.models import ApiSpec
from api_shakedown.testdata import Dataset, find_entities, sample_records
from api_shakedown.wiremock import StubSet, base_path

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
PETS = [
    {"id": 11, "name": "Biscuit", "status": "available", "tags": ["dog"]},
    {"id": 12, "name": "Mittens", "status": "sold", "tags": ["cat"]},
]


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def context(spec: ApiSpec, **inputs: Any) -> AgentContext:
    return AgentContext(inputs={"spec": spec, **inputs}, artifacts={}, reviews=[], round=0)


def produce(agent: Any, ctx: AgentContext) -> Any:
    return asyncio.run(agent.produce(ctx))


def dataset_reply(records: list[dict[str, Any]]) -> str:
    return json.dumps({"entities": [{"entity": "Pet", "records": records}]})


@pytest.fixture
def dataset(spec: ApiSpec) -> Dataset:
    model = ScriptedChatModel(replies=[dataset_reply(PETS)])
    result: Dataset = produce(DatasetAgent(model=model), context(spec))
    return result


# --- entities, bindings, records ------------------------------------------------------


def test_finds_entities_and_bindings(spec: ApiSpec) -> None:
    dataset = produce(DatasetAgent(), context(spec))
    assert list(dataset.entities) == ["Pet"]  # NewPet is request-only, Error is error-only
    assert dataset.entities["Pet"].id_field == "id"
    assert [(b.path, b.param, b.entity, b.field) for b in dataset.bindings] == [
        ("/pets/{petId}", "petId", "Pet", "id")
    ]
    assert dataset.known_ids("/pets/{petId}", "petId") == [1, 2, 3]
    assert dataset.notes == []


def test_sample_records_are_valid_distinct_and_varied(spec: ApiSpec) -> None:
    pet = find_entities(spec)["Pet"]
    records = sample_records(pet, spec.openapi_version, count=4)
    assert [r["id"] for r in records] == [1, 2, 3, 4]
    assert [r["status"] for r in records] == ["available", "pending", "sold", "available"]
    assert len({r["name"] for r in records}) == 4


def test_edge_cases_know_what_the_schema_accepts(spec: ApiSpec) -> None:
    name_schema = spec.schemas["NewPet"]["properties"]["name"]  # string, 1..64 chars
    cases = {c.description: c.valid for c in edge_cases(name_schema, spec.openapi_version)}
    assert cases["empty string"] is False
    assert cases["exactly minLength (1)"] is True
    assert cases["exactly maxLength (64)"] is True
    assert cases["maxLength+1 (65)"] is False
    assert cases["SQL injection probe"] is True  # valid input the server must handle safely
    assert cases["null"] is False

    id_schema = {"type": "integer", "format": "int64"}
    ids = {c.description: c.valid for c in edge_cases(id_schema, spec.openapi_version)}
    assert ids["2^31 (int32 overflow)"] is True
    assert ids["2^63 (int64 overflow)"] is False
    assert ids["fraction for an integer"] is False


def test_dataset_includes_edge_cases_for_request_schemas(dataset: Dataset) -> None:
    assert set(dataset.edge_cases) == {"Pet", "NewPet"}
    statuses = {c.value: c.valid for c in dataset.edge_cases["NewPet"]["status"]}
    assert statuses["sold"] is True and statuses["NOT_A_MEMBER"] is False


# --- the agent ----------------------------------------------------------------------------


def test_agent_uses_valid_llm_records(dataset: Dataset) -> None:
    assert dataset.entities["Pet"].records == PETS
    assert dataset.known_ids("/pets/{petId}", "petId") == [11, 12]
    assert dataset.notes == []


def test_agent_feeds_back_invalid_and_duplicate_records(spec: ApiSpec) -> None:
    bad = [PETS[0], {"id": 11, "name": "Dup", "status": "sold"}, {"id": 13, "status": "lost"}]
    model = ScriptedChatModel(replies=[dataset_reply(bad), dataset_reply(PETS)])
    dataset = produce(DatasetAgent(model=model), context(spec))
    retry = model.calls[1][-1].text
    assert "Pet record 1: duplicate id 11." in retry
    assert "Pet record 2: (root): 'name' is a required property" in retry
    assert dataset.entities["Pet"].records == PETS


def test_agent_salvages_valid_records_then_falls_back(spec: ApiSpec) -> None:
    bad = [PETS[0], {"id": 11, "name": "Dup", "status": "sold"}]
    model = ScriptedChatModel(replies=[dataset_reply(bad)])
    salvaged = produce(DatasetAgent(model=model, max_attempts=1), context(spec))
    assert salvaged.entities["Pet"].records == [PETS[0]]

    garbage = ScriptedChatModel(replies=["no"])
    fallback = produce(DatasetAgent(model=garbage, max_attempts=1), context(spec))
    assert fallback.known_ids("/pets/{petId}", "petId") == [1, 2, 3]
    assert "Pet: used deterministic sample records." in fallback.notes


def test_prompt_shows_schema_and_count(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[dataset_reply(PETS)])
    produce(DatasetAgent(model=model, records_per_entity=5), context(spec))
    prompt = model.calls[0][-1].text
    assert '<entity name="Pet" id_field="id">' in prompt
    assert "Create 5 records for each of: Pet." in prompt


# --- stubs and contracts use the dataset --------------------------------------------------


def test_stubs_serve_dataset_records(spec: ApiSpec, dataset: Dataset) -> None:
    stubs = produce(StubAgent(), context(spec, dataset=dataset))
    mappings = {m["name"]: m for m in stubs.mappings}

    assert mappings["listPets 200"]["response"]["jsonBody"] == PETS
    pet12 = mappings["getPet 200 petId=12"]
    assert pet12["request"]["urlPathPattern"] == "/v1/pets/12"
    assert pet12["priority"] == 4 and pet12["response"]["jsonBody"] == PETS[1]
    assert mappings["deletePet 204 petId=11"]["response"]["status"] == 204
    unknown = mappings["getPet 404 (unknown id)"]
    assert unknown["priority"] == 5 and unknown["response"]["status"] == 404
    assert "getPet 200" not in mappings  # no generic happy stub for a bound id


def test_create_returns_a_new_record(spec: ApiSpec, dataset: Dataset) -> None:
    stubs = produce(StubAgent(), context(spec, dataset=dataset))
    created = next(m for m in stubs.mappings if m["name"] == "createPet 201")
    assert created["response"]["jsonBody"]["id"] == 13  # existing ids are 11 and 12
    assert created["metadata"]["shakedown"]["source"] == "dataset"
    error = next(m for m in stubs.mappings if m["name"] == "getPet 404")
    assert error["metadata"]["shakedown"]["source"] == "sample"


def test_new_record_ids_for_string_and_uuid_keys(spec: ApiSpec) -> None:
    pet = find_entities(spec)["Pet"]
    pet.records = [{"id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "name": "a"}]
    new = pet.new_record()
    assert new is not None and new["id"] != pet.records[0]["id"] and len(new["id"]) == 36
    pet.records = [{"id": "pet-a", "name": "a"}]
    assert pet.new_record() == {"id": "pet-new", "name": "a"}


def test_stub_llm_is_not_asked_for_entity_bodies(spec: ApiSpec, dataset: Dataset) -> None:
    model = ScriptedChatModel(replies=[json.dumps({"responses": []})])
    produce(StubAgent(model=model, max_attempts=1), context(spec, dataset=dataset))
    prompt = model.calls[0][-1].text
    assert "operation_id=getPet status=404" in prompt
    assert "operation_id=getPet status=200" not in prompt
    assert "operation_id=createPet status=201" not in prompt  # returns a Pet


def test_contract_cases_must_respect_known_ids(spec: ApiSpec, dataset: Dataset) -> None:
    op = spec.operation("getPet")

    def problems(status: str, pet_id: int) -> list[str]:
        case = CaseProposal(
            operation_id="getPet", name="x", expected_status=status, path_params={"petId": pet_id}
        )
        return proposal_problems(spec, op, case, dataset)

    assert any("must be an existing id: [11, 12]" in p for p in problems("200", 999))
    assert any("a 404 case needs another id" in p for p in problems("404", 11))
    assert problems("200", 12) == [] and problems("404", 999) == []
    fallback = happy_proposal(op, dataset)
    assert fallback is not None and fallback.path_params == {"petId": 11}


def test_contract_prompt_lists_known_ids(spec: ApiSpec, dataset: Dataset) -> None:
    model = ScriptedChatModel(replies=[json.dumps({"cases": []})])
    produce(ContractTestAgent(model=model, max_attempts=1), context(spec, dataset=dataset))
    assert "- getPet petId: [11, 12]" in model.calls[0][-1].text


def test_dataset_aware_stubs_answer_by_id(
    spec: ApiSpec,
    dataset: Dataset,
    serve_stubs: Callable[[StubSet], tuple[str, Any]],
) -> None:
    url, _ = serve_stubs(produce(StubAgent(), context(spec, dataset=dataset)))
    with httpx.Client(base_url=url + base_path(spec)) as client:
        assert client.get("/pets/12").json() == PETS[1]
        assert client.get("/pets/11").json() == PETS[0]
        assert client.get("/pets/999").status_code == 404
        assert client.delete("/pets/12").status_code == 204
        assert client.delete("/pets/999").status_code == 404


# --- reasoning defaults -------------------------------------------------------------------


def test_generators_default_to_no_thinking() -> None:
    settings = ShakedownSettings()
    assert settings.for_agent("stubs").reasoning is False
    assert settings.for_agent("testdata").reasoning is False
    assert settings.for_agent("reviewer").reasoning is None  # model default
