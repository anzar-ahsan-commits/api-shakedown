"""Seeding: create the test dataset's records on a real server before the contract tests."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from api_shakedown.contracts import template
from api_shakedown.seeding import seed_plan
from api_shakedown.spec import load_spec
from api_shakedown.spec.models import ApiSpec
from api_shakedown.testdata import Dataset, build_dataset, find_entities

NAME = {"type": "string"}
COLLECTION_FIELDS: dict[str, Any] = {
    "name": NAME,
    "fields": {"type": "array", "items": NAME},
    "synonyms": NAME,
}
COLLECTION_IN = {"type": "object", "required": ["name", "fields"], "properties": COLLECTION_FIELDS}
COLLECTION = {
    "type": "object",
    "required": ["name"],
    "properties": {**COLLECTION_FIELDS, "num_documents": {"type": "integer"}},
}
DOC = {"type": "object", "required": ["id"], "properties": {"id": NAME, "title": NAME}}
KEY_IN = {"type": "object", "properties": {"description": NAME}}
KEY = {"type": "object", "properties": {"id": {"type": "integer"}, "description": NAME}}


def _json(name: str) -> dict[str, Any]:
    return {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}}


def _param(name: str) -> dict[str, Any]:
    return {"name": name, "in": "path", "required": True, "schema": NAME}


def _ok(name: str, status: str = "200") -> dict[str, Any]:
    return {status: {"description": "ok", "content": _json(name)}}


SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Search", "version": "1"},
    "paths": {
        "/collections": {
            "post": {
                "operationId": "createCollection",
                "requestBody": {"required": True, "content": _json("CollectionIn")},
                "responses": _ok("Collection", "201"),
            }
        },
        "/collections/{collectionName}": {
            "parameters": [_param("collectionName")],
            "get": {"operationId": "getCollection", "responses": _ok("Collection")},
            "delete": {"operationId": "deleteCollection", "responses": _ok("Collection")},
        },
        "/collections/{collectionName}/docs/{docId}": {
            "parameters": [_param("collectionName"), _param("docId")],
            "put": {
                "operationId": "putDoc",
                "requestBody": {"required": True, "content": _json("Doc")},
                "responses": _ok("Doc"),
            },
            "get": {"operationId": "getDoc", "responses": _ok("Doc")},
            "delete": {"operationId": "deleteDoc", "responses": _ok("Doc")},
        },
        "/keys": {
            "post": {
                "operationId": "createKey",
                "requestBody": {"required": True, "content": _json("KeyIn")},
                "responses": _ok("Key", "201"),
            }
        },
        "/keys/{keyId}": {
            "parameters": [_param("keyId")],
            "get": {"operationId": "getKey", "responses": _ok("Key")},
        },
    },
    "components": {
        "schemas": {
            "CollectionIn": COLLECTION_IN,
            "Collection": COLLECTION,
            "Doc": DOC,
            "KeyIn": KEY_IN,
            "Key": KEY,
        }
    },
}


@pytest.fixture
def spec(tmp_path: Path) -> ApiSpec:
    path = tmp_path / "search.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    return load_spec(path)


@pytest.fixture
def dataset(spec: ApiSpec) -> Dataset:
    entities = find_entities(spec)
    entities["Collection"].records = [
        {"name": "products", "fields": ["title"], "synonyms": "missing-set", "num_documents": 2},
        {"name": "users", "fields": ["email"], "num_documents": 0},
    ]
    entities["Doc"].records = [{"id": "doc-1", "title": "Lamp"}, {"id": "doc-2"}]
    return build_dataset(spec, entities)


@pytest.fixture(scope="module")
def support() -> ModuleType:
    source = template("shakedown_support.py")
    module = ModuleType("shakedown_support")
    exec(compile(source, "shakedown_support.py", "exec"), module.__dict__)
    return module


def test_plan_creates_parents_first_with_request_fields_only(
    spec: ApiSpec, dataset: Dataset
) -> None:
    steps, _ = seed_plan(spec, dataset)
    assert [(s.create.method, s.create.path, s.record_id) for s in steps] == [
        ("post", "/collections", "products"),
        ("post", "/collections", "users"),
        ("put", "/collections/{collectionName}/docs/{docId}", "doc-1"),
        ("put", "/collections/{collectionName}/docs/{docId}", "doc-2"),
    ]
    products = steps[0]
    # num_documents is a response-only field: not sent.
    assert products.create.body == {
        "name": "products",
        "fields": ["title"],
        "synonyms": "missing-set",
    }
    assert products.fallback is not None
    assert products.fallback.body == {"name": "products", "fields": ["title"]}
    assert steps[1].fallback is None  # already minimal
    assert products.delete is not None
    assert products.delete.path_params == {"collectionName": "products"}
    # Children go under the first parent record.
    assert steps[2].create.path_params == {"collectionName": "products", "docId": "doc-1"}


def test_plan_explains_server_assigned_ids(spec: ApiSpec) -> None:
    entities = find_entities(spec)
    entities["Key"].records = [{"id": 1, "description": "admin"}]
    _, notes = seed_plan(spec, build_dataset(spec, entities))
    assert any("/keys/{keyId}" in n and "server assigns it" in n for n in notes)


class FakeServer:
    """Collections and docs in memory, with the kind of rules a real server adds."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {"users": {"name": "users"}}
        self.docs: dict[tuple[str, str], dict[str, Any]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "POST" and parts == ["collections"]:
            if body.get("synonyms"):
                return httpx.Response(404, json={"message": "Synonym set not found"})
            if body["name"] in self.collections:
                return httpx.Response(409, json={"message": "exists"})
            self.collections[body["name"]] = body
            return httpx.Response(201, json=body)
        if request.method == "DELETE" and len(parts) == 2:
            self.collections.pop(parts[1], None)
            return httpx.Response(200, json={})
        if len(parts) == 4:
            if parts[1] not in self.collections:
                return httpx.Response(404, json={"message": "Collection not found"})
            if request.method == "PUT":
                self.docs[(parts[1], parts[3])] = body
                return httpx.Response(200, json=body)
            self.docs.pop((parts[1], parts[3]), None)
            return httpx.Response(200, json={})
        return httpx.Response(404)


def test_seed_falls_back_retries_and_cleans_up(
    spec: ApiSpec, dataset: Dataset, support: ModuleType
) -> None:
    steps, _ = seed_plan(spec, dataset)
    # Docs first: their collection doesn't exist yet, so they need the second pass.
    planned = [s.model_dump() for s in reversed(steps)]
    server = FakeServer()
    with httpx.Client(transport=httpx.MockTransport(server), base_url="http://x") as client:
        created = support.seed(client, planned, "X-Shakedown-Status")
        # products needed the minimal body; users already existed (409).
        assert set(server.collections) == {"products", "users"}
        assert set(server.docs) == {("products", "doc-1"), ("products", "doc-2")}
        assert {s["record_id"] for s in created} == {"products", "doc-1", "doc-2"}

        support.unseed(client, created, "X-Shakedown-Status")
    # Only what the tests created is removed; users was there before.
    assert set(server.collections) == {"users"}
    assert server.docs == {}


def test_seed_warns_about_records_it_cannot_create(support: ModuleType) -> None:
    step = {
        "entity": "Collection",
        "record_id": "bad",
        "create": {
            "method": "post",
            "path": "/collections",
            "path_params": {},
            "query": {},
            "headers": {},
            "body": {"name": "bad", "synonyms": "x"},
            "auth": [],
        },
    }
    with (
        httpx.Client(transport=httpx.MockTransport(FakeServer()), base_url="http://x") as client,
        pytest.warns(UserWarning, match="Collection bad: 404"),
    ):
        assert support.seed(client, [step], "X-Shakedown-Status") == []


def test_deleting_cases_run_last(support: ModuleType) -> None:
    cases = [
        {"id": "del", "method": "delete", "expected_status": "200"},
        {"id": "del404", "method": "delete", "expected_status": "404"},
        {"id": "get", "method": "get", "expected_status": "200"},
    ]
    assert [c["id"] for c in support.run_order(cases)] == ["del404", "get", "del"]
