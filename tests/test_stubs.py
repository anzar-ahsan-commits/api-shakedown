import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from agentpanel import AgentContext, Artifact, Review
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.stubs import StubAgent
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec, Operation, Parameter
from api_shakedown.spec.samples import sample_from_schema
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.wiremock import (
    STATUS_HEADER,
    StubSet,
    build_stub_set,
    expected_responses,
    path_regex,
    status_code,
)

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
PET = {"id": 7, "name": "Biscuit", "status": "pending", "tags": ["indoor"]}
ERROR = {"code": 404, "message": "Pet 7 not found"}


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def context(spec: ApiSpec, **overrides: Any) -> AgentContext:
    base: dict[str, Any] = {"inputs": {"spec": spec}, "artifacts": {}, "reviews": [], "round": 0}
    return AgentContext(**(base | overrides))


def produce(agent: StubAgent, ctx: AgentContext) -> StubSet:
    return asyncio.run(agent.produce(ctx))


def good_reply(spec: ApiSpec) -> str:
    bodies = {
        ("listPets", "200"): [PET],
        ("createPet", "201"): PET,
        ("getPet", "200"): PET,
        ("deletePet", "204"): None,
        ("get_health", "200"): "ok",
    }
    entries = [
        {
            "operation_id": op.id,
            "status": r.status,
            "body": bodies.get((op.id, r.status), ERROR),
            "headers": {"X-Next-Cursor": "c2"} if r.headers else {},
        }
        for op, r in expected_responses(spec.operations)
    ]
    return json.dumps({"responses": entries})


def by_name(stubs: StubSet) -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in stubs.mappings}


# --- samples and schema checks ------------------------------------------------


def test_samples_validate_against_every_petstore_schema(spec: ApiSpec) -> None:
    for _, response in expected_responses(spec.operations):
        if response.body and response.body.schema_:
            sample = sample_from_schema(response.body.schema_)
            assert schema_errors(response.body.schema_, sample, spec.openapi_version) == []


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string", "format": "uuid"}, "3fa85f64-5717-4562-b3fc-2c963f66afa6"),
        ({"type": "string", "minLength": 10}, "stringxxxx"),
        ({"type": "integer", "minimum": 5, "maximum": 9}, 5),
        ({"type": ["null", "number"]}, 1.0),
        ({"oneOf": [{"type": "boolean"}, {"type": "string"}]}, True),
        ({"type": "array", "items": {"type": "integer"}, "minItems": 2}, [1, 1]),
        ({"$ref": "#/x", "x-circular": True}, None),
    ],
)
def test_sample_from_schema(schema: dict[str, Any], expected: Any) -> None:
    assert sample_from_schema(schema) == expected


def test_schema_errors_reports_paths(spec: ApiSpec) -> None:
    schema = spec.operation("getPet").responses[0].body.schema_  # type: ignore[union-attr]
    assert schema is not None
    errors = schema_errors(schema, {"name": "x", "status": "lost"}, spec.openapi_version)
    assert any("'id' is a required property" in e for e in errors)
    assert any(e.startswith("status:") for e in errors)


# --- request matching ------------------------------------------------------------


def test_path_regex_constrains_params_by_type() -> None:
    def op(schema: dict[str, Any]) -> Operation:
        param = Parameter(name="id", location="path", required=True, schema=schema)
        return Operation(id="x", method="get", path="/items/{id}.json", parameters=[param])

    cases = {
        "/v1/items/42.json": True,
        "/v1/items/-3.json": False,
        "/v1/items/abc.json": False,
        "/v1/items/42xjson": False,  # the literal dot is escaped
    }
    regex = path_regex(op({"type": "integer", "minimum": 0}), "/v1")
    for path, matches in cases.items():
        assert bool(re.fullmatch(regex, path)) is matches, path
    assert re.fullmatch(path_regex(op({"enum": ["a.b", "c"]})), "/items/a.b.json")
    assert not re.fullmatch(path_regex(op({"type": "string"})), "/items/a/b.json")


@pytest.mark.parametrize(("label", "code"), [("404", 404), ("4XX", 400), ("default", 500)])
def test_status_code(label: str, code: int) -> None:
    assert status_code(label) == code


def test_offline_stubs_cover_every_response(spec: ApiSpec) -> None:
    stubs = produce(StubAgent(), context(spec))
    assert len(stubs.mappings) == 9
    assert stubs.notes == []
    mappings = by_name(stubs)

    happy = mappings["getPet 200"]
    assert happy["priority"] == 15 and "headers" not in happy["request"]  # 1 path param, normal
    assert happy["request"] == {"method": "GET", "urlPathPattern": "/v1/pets/[0-9]+"}

    not_found = mappings["getPet 404"]
    assert not_found["priority"] == 12  # 1 path param, header-selected
    assert not_found["request"]["headers"] == {STATUS_HEADER: {"equalTo": "404"}}
    assert not_found["response"]["status"] == 404

    assert mappings["listPets default"]["response"]["status"] == 500
    assert mappings["listPets 200"]["response"]["jsonBody"] == [
        {"id": 1, "name": "Rex", "status": "available"}  # the spec's own example
    ]
    assert mappings["listPets 200"]["response"]["headers"]["X-Next-Cursor"] == "string"
    assert "jsonBody" not in mappings["deletePet 204"]["response"]
    health = mappings["get_health 200"]["response"]
    assert (health["body"], health["headers"]["Content-Type"]) == ("ok", "text/plain")


def test_required_query_params_are_matched() -> None:
    param = Parameter(name="q", location="query", required=True, schema={"type": "string"})
    responses = load_spec(PETSTORE).operation("get_health").responses
    op = Operation(
        id="search", method="get", path="/search", parameters=[param], responses=responses
    )
    spec = ApiSpec(title="t", version="1", openapi_version="3.0.3", operations=[op])
    (mapping,) = build_stub_set(spec, {}).mappings
    assert mapping["request"]["queryParameters"] == {"q": {"matches": ".+"}}


# --- the LLM path ---------------------------------------------------------------------


def test_uses_valid_llm_bodies(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[good_reply(spec)])
    stubs = produce(StubAgent(model=model), context(spec))
    mappings = by_name(stubs)
    assert mappings["getPet 200"]["response"]["jsonBody"] == PET
    assert mappings["getPet 404"]["response"]["jsonBody"] == ERROR
    assert mappings["listPets 200"]["response"]["headers"]["X-Next-Cursor"] == "c2"
    assert stubs.notes == []
    prompt = model.calls[0][-1].text
    assert "operation_id=deletePet status=204 (no body: use null)" in prompt
    assert '"id":"getPet"' in prompt


def test_schema_violations_are_fed_back(spec: ApiSpec) -> None:
    bad = json.loads(good_reply(spec))
    bad["responses"][0]["body"] = [{"name": "No id"}]  # listPets 200 lacks required id
    model = ScriptedChatModel(replies=[json.dumps(bad), good_reply(spec)])
    stubs = produce(StubAgent(model=model), context(spec))
    retry_prompt = model.calls[1][-1].text
    assert "listPets 200: 0: 'id' is a required property" in retry_prompt
    assert stubs.notes == []


def test_unfixable_bodies_fall_back_to_samples_and_keep_good_ones(spec: ApiSpec) -> None:
    bad = json.loads(good_reply(spec))
    bad["responses"][0]["body"] = [{"name": "No id"}]
    model = ScriptedChatModel(replies=[json.dumps(bad)] * 2)
    stubs = produce(StubAgent(model=model, max_attempts=2), context(spec))
    mappings = by_name(stubs)
    assert mappings["listPets 200"]["response"]["jsonBody"][0]["name"] == "Rex"  # spec example
    assert mappings["getPet 200"]["response"]["jsonBody"] == PET  # valid LLM body kept
    assert "listPets 200: used a schema-generated sample body." in stubs.notes


def test_garbage_reply_falls_back_entirely(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=["I cannot help with that."])
    stubs = produce(StubAgent(model=model, max_attempts=1), context(spec))
    assert len(stubs.mappings) == 9
    assert len(stubs.notes) == 1 + 9


def test_revision_includes_reviewer_feedback(spec: ApiSpec) -> None:
    model = ScriptedChatModel(replies=[good_reply(spec)])
    rejection = Review(
        reviewer="critic", target="stubs", round=0, approved=False, comments=["Use real names"]
    )
    previous = produce(StubAgent(), context(spec))
    ctx = context(
        spec,
        round=1,
        reviews=[rejection],
        artifacts={"stubs": Artifact(author="stubs", kind="stubs", content=previous)},
    )
    produce(StubAgent(model=model), ctx)
    assert "- (critic) Use real names" in model.calls[0][-1].text


def test_large_specs_are_chunked(spec: ApiSpec) -> None:
    replies = [good_reply(spec)] * 5  # each chunk ignores entries for other chunks
    model = ScriptedChatModel(replies=replies)
    produce(StubAgent(model=model, max_chars=1), context(spec))
    assert len(model.calls) == len(spec.operations)
