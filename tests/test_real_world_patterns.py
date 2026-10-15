"""Regressions from running the Typesense spec (79 operations): patterns real specs use that
the small example spec did not."""

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentpanel import AgentContext
from agentpanel.testing import ScriptedChatModel

from api_shakedown.agents.chaos import ChaosAgent
from api_shakedown.agents.contracts import ContractTestAgent
from api_shakedown.agents.stubs import StubAgent
from api_shakedown.agents.testdata import DatasetAgent
from api_shakedown.chaos import Effect, ScenarioProposal, compile_scenario
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec, Parameter
from api_shakedown.spec.params import appears_by_name, query_pairs, serialize
from api_shakedown.spec.samples import example_problems, valid_sample
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Entity, find_bindings, find_entities, sample_records
from api_shakedown.wiremock import StubSet, build_stub_set, request_matcher

DOC = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}
SEARCH_PARAMS = {
    "type": "object",
    "required": ["q"],
    "properties": {"q": {"type": "string"}, "per_page": {"type": "integer"}},
}
SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Search", "version": "1"},
    "paths": {
        "/collections/{c}/documents/{documentId}": {
            "get": {
                "operationId": "getDocument",
                "parameters": [
                    {"name": "c", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "documentId", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "doc", "content": {
                    "application/json": {"schema": DOC}}}},
            }
        },
        "/collections/{c}/documents/export": {
            "get": {
                "operationId": "exportDocuments",
                "parameters": [
                    {"name": "c", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {"200": {"description": "jsonl", "content": {
                    "application/octet-stream": {"schema": {"type": "string"}}}}},
            }
        },
        "/collections/{c}/documents/search": {
            "get": {
                "operationId": "search",
                "parameters": [
                    {"name": "c", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "searchParameters", "in": "query", "required": True,
                     "schema": SEARCH_PARAMS},
                ],
                "responses": {
                    "200": {"description": "hits", "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {"found": {"type": "integer", "example": "many"}},
                    }}}},
                    "400": {"description": "bad", "content": {"application/json": {
                        "schema": {"type": "object"}}}},
                },
            }
        },
        "/stopwords": {
            "get": {
                "operationId": "listStopwords",
                "responses": {"200": {"description": "sets", "content": {"application/json": {
                    "schema": {"type": "object", "required": ["stopwords"],
                               "properties": {"stopwords": {"type": "array"}}},
                    "example": '{"stopwords": []}\n',
                }}}},
            }
        },
    },
}  # fmt: skip


@pytest.fixture
def spec(tmp_path: Path) -> ApiSpec:
    path = tmp_path / "search.json"
    path.write_text(json.dumps(SPEC))
    return load_spec(path)


def context(spec: ApiSpec) -> AgentContext:
    return AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)


def produce(agent: Any, spec: ApiSpec) -> Any:
    return asyncio.run(agent.produce(context(spec)))


# --- 1. overlapping routes ------------------------------------------------------------------


def test_literal_route_beats_parameterized_route(
    spec: ApiSpec, serve_stubs: Callable[[StubSet], tuple[str, Any]]
) -> None:
    url, _ = serve_stubs(build_stub_set(spec, {}))
    with httpx.Client(base_url=url) as client:
        export = client.get("/collections/books/documents/export")
        assert export.headers["content-type"] == "application/octet-stream"
        assert client.get("/collections/books/documents/abc").json() == {"id": "string"}


def test_chaos_on_a_parameterized_route_does_not_leak_onto_a_literal_one(spec: ApiSpec) -> None:
    stubs = build_stub_set(spec, {})
    chaos = compile_scenario(
        spec,
        ScenarioProposal(name="docs_down", title="t", category="outage",
                         operations=["getDocument"], effect=Effect(status=503), expectation="e"),
        "llm", None,
    ).mappings  # fmt: skip
    export_normal = next(m for m in stubs.mappings if m["name"] == "exportDocuments 200")
    (doc_chaos,) = chaos
    assert export_normal["priority"] < doc_chaos["priority"]  # export still wins


# --- 2. OpenAPI query serialization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "value", "expected"),
    [
        (Parameter(name="p", location="query", schema=SEARCH_PARAMS), {"q": "x", "per_page": 2},
         [("q", "x"), ("per_page", "2")]),
        (Parameter(name="p", location="query", style="deepObject"), {"q": "x"}, [("p[q]", "x")]),
        (Parameter(name="p", location="query", explode=False), {"q": "x", "n": 1},
         [("p", "q,x,n,1")]),
        (Parameter(name="tag", location="query"), ["a", "b"], [("tag", "a"), ("tag", "b")]),
        (Parameter(name="tag", location="query", explode=False), ["a", "b"], [("tag", "a,b")]),
        (Parameter(name="tag", location="query", style="pipeDelimited", explode=False),
         ["a", "b"], [("tag", "a|b")]),
        (Parameter(name="flag", location="query"), True, [("flag", "true")]),
        (Parameter(name="filter", location="query", content_type="application/json"),
         {"a": 1}, [("filter", '{"a": 1}')]),
        (Parameter(name="p", location="query"), None, []),
    ],
)  # fmt: skip
def test_serialize(param: Parameter, value: Any, expected: list[tuple[str, str]]) -> None:
    assert serialize(param, value) == expected


def test_exploded_object_params_are_not_required_by_name(spec: ApiSpec) -> None:
    search = spec.operation("search")
    (param,) = (p for p in search.parameters if p.location == "query")
    assert not appears_by_name(param)
    assert "queryParameters" not in request_matcher(search, "")
    pairs = query_pairs(search, {"searchParameters": {"q": "dune"}})
    assert pairs == [["q", "dune"]]


def test_generated_suites_send_exploded_queries(
    spec: ApiSpec, tmp_path: Path, serve_stubs: Callable[[StubSet], tuple[str, Any]]
) -> None:
    url, handler = serve_stubs(produce(StubAgent(), spec))
    contracts = produce(ContractTestAgent(), spec)
    chaos = produce(ChaosAgent(), spec)
    for name, content in (contracts.files() | chaos.files()).items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = os.environ | {"SHAKEDOWN_TARGET_URL": url, "SHAKEDOWN_SELECT_STATUS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    search_paths = [path for _, path, _ in handler.received if "/documents/search" in path]
    assert "/collections/string/documents/search?q=string&per_page=1" in search_paths
    # Only the deliberate "string instead of object" probe uses the parameter's own name;
    # nothing sends a serialized dict or list under it.
    named = [p for p in search_paths if "searchParameters=" in p]
    assert named == ["/collections/string/documents/search?searchParameters=a+string"]


# --- 3. spec examples that contradict their schemas -------------------------------------------


def test_valid_sample_repairs_or_replaces_bad_examples() -> None:
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    assert valid_sample(schema, '{"n": 3}', "3.0.3") == {"n": 3}  # JSON written as text
    assert valid_sample(schema, {"n": "three"}, "3.0.3") == {"n": 1}  # wrong type: regenerated
    nested = {"type": "object", "properties": {"n": {"type": "integer", "example": "x"}}}
    assert valid_sample(nested, None, "3.0.3") == {"n": 1}  # nested example ignored


def test_stub_bodies_are_valid_and_problems_are_reported(spec: ApiSpec) -> None:
    stubs = build_stub_set(spec, {})
    listing = next(m for m in stubs.mappings if m["name"] == "listStopwords 200")
    assert listing["response"]["jsonBody"] == {"stopwords": []}
    found = next(m for m in stubs.mappings if m["name"] == "search 200")["response"]["jsonBody"]
    search_schema = spec.operation("search").responses[0].body.schema_  # type: ignore[union-attr]
    assert search_schema is not None and not schema_errors(search_schema, found, "3.0.3")
    notes = example_problems(spec)
    assert "listStopwords 200: the spec's example is JSON written as text; parsed it." in notes
    assert "search 200: examples inside the spec's schema contradict it; ignored them." in notes
    assert all(n in stubs.notes for n in notes)


def test_dataset_records_ignore_contradicting_nested_examples() -> None:
    schema = {
        "type": "object",
        "required": ["id", "hits"],
        "properties": {
            "id": {"type": "string"},
            "hits": {"type": "array", "items": {"type": "object", "example": "not an object"}},
        },
    }
    records = sample_records(Entity(name="Result", schema=schema, id_field="id"), "3.0.3")
    assert records and all(not schema_errors(schema, r, "3.0.3") for r in records)


# --- 4. test data at scale: which schemas deserve generated records -----------------------

COLLECTION = {
    "type": "object",
    "required": ["name"],
    "properties": {"name": {"type": "string"}, "num_documents": {"type": "integer"}},
}
DELETED = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
RESULTS = {
    "type": "object",
    "properties": {
        "found": {"type": "integer"},
        "hits": {"type": "array", "items": {"type": "object"}},
    },
}
DATA_SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Search", "version": "1"},
    "paths": {
        "/collections/{collectionName}": {
            "parameters": [{"name": "collectionName", "in": "path", "required": True,
                            "schema": {"type": "string"}}],
            "delete": {"operationId": "deleteCollection", "responses": {"200": {
                "description": "gone", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/CollectionDeleted"}}}}}},
            "get": {"operationId": "getCollection", "responses": {"200": {
                "description": "ok", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/Collection"}}}}}},
        },
        "/search": {"get": {"operationId": "search", "responses": {"200": {
            "description": "hits", "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/SearchResult"}}}}}}},
    },
    "components": {"schemas": {
        "Collection": COLLECTION, "CollectionDeleted": DELETED, "SearchResult": RESULTS,
    }},
}  # fmt: skip


@pytest.fixture
def data_spec(tmp_path: Path) -> ApiSpec:
    path = tmp_path / "data.json"
    path.write_text(json.dumps(DATA_SPEC))
    return load_spec(path)


def test_resources_keyed_by_name_are_found_from_the_path(data_spec: ApiSpec) -> None:
    entities = find_entities(data_spec)
    assert entities["Collection"].id_field == "name"  # from {collectionName}
    assert entities["SearchResult"].id_field is None  # an envelope, not a resource
    (binding,) = find_bindings(data_spec, entities)
    assert (binding.entity, binding.field) == ("Collection", "name")  # the GET decides


def test_only_resources_small_enough_go_to_the_llm(data_spec: ApiSpec) -> None:
    reply = {
        "entities": [
            {"entity": "Collection", "records": [{"name": "books"}]},
            {"entity": "CollectionDeleted", "records": [{"name": "old"}]},
        ]
    }
    model = ScriptedChatModel(replies=[json.dumps(reply)])
    dataset = asyncio.run(DatasetAgent(model=model).produce(context(data_spec)))
    prompt = model.calls[0][-1].text
    assert '<entity name="Collection"' in prompt and "SearchResult" not in prompt
    assert dataset.entities["Collection"].records == [{"name": "books"}]
    assert dataset.entities["SearchResult"].records  # still has valid schema samples
    assert any("1 response schemas without a key (SearchResult)" in n for n in dataset.notes)

    tiny = DatasetAgent(model=ScriptedChatModel(replies=[]), max_chars=50)
    skipped = asyncio.run(tiny.produce(context(data_spec)))
    assert any("Collection: schema too large for one prompt" in n for n in skipped.notes)
