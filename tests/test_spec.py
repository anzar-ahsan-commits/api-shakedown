import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from api_shakedown.spec import SpecError, chunk_operations, load_spec
from api_shakedown.spec.models import Operation

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


def write_spec(tmp_path: Path, paths: dict[str, Any], **extra: Any) -> Path:
    document = {"openapi": "3.0.3", "info": {"title": "T", "version": "1"}, "paths": paths}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document | extra))
    return path


def ok(**body: Any) -> dict[str, Any]:
    return {"responses": {"200": {"description": "ok", **body}}}


# --- the bundled example -----------------------------------------------------


def test_petstore_operations() -> None:
    spec = load_spec(PETSTORE)
    assert spec.title == "Petstore" and spec.servers == ["http://localhost:8080/v1"]
    assert [op.id for op in spec.operations] == [
        "listPets", "createPet", "getPet", "deletePet", "get_health",
    ]  # fmt: skip
    assert spec.warnings == []
    assert spec.security_schemes["apiKey"]["name"] == "X-API-Key"


def test_petstore_refs_are_inlined() -> None:
    spec = load_spec(PETSTORE)
    assert all("$ref" not in op.to_prompt() for op in spec.operations)
    create = spec.operation("createPet")
    assert create.request_body is not None and create.request_body.required
    assert create.request_body.schema_ is not None
    assert create.request_body.schema_["required"] == ["name"]


def test_petstore_path_level_params_and_yaml_int_status_codes() -> None:
    get_pet = load_spec(PETSTORE).operation("getPet")
    assert [(p.name, p.location, p.required) for p in get_pet.parameters] == [
        ("petId", "path", True)
    ]
    assert [r.status for r in get_pet.responses] == ["200", "404"]


def test_petstore_security_global_and_overridden() -> None:
    spec = load_spec(PETSTORE)
    assert spec.operation("createPet").security == ["apiKey"]
    assert spec.operation("get_health").security == []


def test_petstore_examples_and_headers() -> None:
    spec = load_spec(PETSTORE)
    health = spec.operation("get_health").responses[0].body
    assert health is not None and (health.content_type, health.example) == ("text/plain", "ok")
    listing = spec.operation("listPets").responses[0]
    assert listing.headers == {"X-Next-Cursor": {"type": "string"}}


# --- normalization edge cases ------------------------------------------------


def test_circular_refs_stay_finite(tmp_path: Path) -> None:
    node = {
        "type": "object",
        "properties": {
            "children": {"type": "array", "items": {"$ref": "#/components/schemas/Node"}}
        },
    }
    content = {"application/json": {"schema": {"$ref": "#/components/schemas/Node"}}}
    path = write_spec(
        tmp_path, {"/tree": {"get": ok(content=content)}}, components={"schemas": {"Node": node}}
    )
    body = load_spec(path).operations[0].responses[0].body
    assert body is not None and body.schema_ is not None
    items = body.schema_["properties"]["children"]["items"]
    assert items == {"$ref": "#/components/schemas/Node", "x-circular": True}


def test_unresolvable_and_external_refs_warn(tmp_path: Path) -> None:
    content = {"application/json": {"schema": {"$ref": "other.yaml#/Pet"}}}
    params = [{"$ref": "#/components/parameters/Missing"}]
    path = write_spec(tmp_path, {"/a": {"get": ok(content=content), "parameters": params}})
    warnings = load_spec(path).warnings
    assert any("External $ref not followed: other.yaml#/Pet" in w for w in warnings)
    assert any("Unresolvable $ref: #/components/parameters/Missing" in w for w in warnings)


def test_operation_params_override_path_params(tmp_path: Path) -> None:
    shared = [{"name": "q", "in": "query", "description": "shared"}]
    own = [{"name": "q", "in": "query", "description": "own", "required": True}]
    path = write_spec(tmp_path, {"/a": {"parameters": shared, "get": ok() | {"parameters": own}}})
    (param,) = load_spec(path).operations[0].parameters
    assert (param.description, param.required) == ("own", True)


def test_duplicate_operation_ids_are_made_unique(tmp_path: Path) -> None:
    op = ok() | {"operationId": "dup"}
    spec = load_spec(write_spec(tmp_path, {"/a": {"get": op}, "/b": {"get": op}}))
    assert [o.id for o in spec.operations] == ["dup", "dup_2"]
    assert any("Duplicate operationId" in w for w in spec.warnings)


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (["application/xml", "application/json"], "application/json"),
        (["text/csv", "application/vnd.api+json"], "application/vnd.api+json"),
        (["text/csv", "text/plain"], "text/csv"),
    ],
)
def test_prefers_json_content(tmp_path: Path, types: list[str], expected: str) -> None:
    content = {t: {"schema": {"type": "string"}} for t in types}
    body = load_spec(write_spec(tmp_path, {"/a": {"get": ok(content=content)}})).operations[0]
    assert body.responses[0].body is not None
    assert body.responses[0].body.content_type == expected


# --- loading and validation --------------------------------------------------


def test_invalid_spec_loads_with_warnings(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"openapi": "3.0.3", "info": {"title": "T"}, "paths": {"/a": {"get": ok()}}})
    )
    spec = load_spec(path)
    assert len(spec.operations) == 1
    assert any("'version' is a required property" in w for w in spec.warnings)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"swagger": "2.0", "paths": {}}', "Swagger 2.0"),
        ('{"openapi": "3.0.3", "paths": {}}', "no paths"),
        ('{"info": {}, "paths": {"/a": {}}}', "Missing `openapi`"),
        ("- just\n- a list\n", "expected a mapping"),
        ("key: [unclosed", "not valid YAML"),
    ],
)
def test_unusable_specs_raise(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(content)
    with pytest.raises(SpecError, match=message):
        load_spec(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "nope.yaml")


def test_loads_from_url() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(PETSTORE.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        spec = load_spec(f"http://127.0.0.1:{server.server_port}/petstore.yaml")
        assert len(spec.operations) == 5
        with pytest.raises(SpecError, match="Could not download"):
            load_spec(f"http://127.0.0.1:{server.server_port}/missing.yaml")
    finally:
        server.shutdown()


# --- chunking ----------------------------------------------------------------


def make_op(op_id: str, tag: str, padding: int = 0) -> Operation:
    return Operation(id=op_id, method="get", path=f"/{op_id}", tags=[tag], summary="x" * padding)


def test_chunks_respect_budget_and_group_by_tag() -> None:
    ops = [make_op("a1", "a"), make_op("b1", "b"), make_op("a2", "a"), make_op("b2", "b")]
    size = len(ops[0].to_prompt())
    chunks = chunk_operations(ops, max_chars=size * 2)
    assert [[op.id for op in chunk] for chunk in chunks] == [["a1", "a2"], ["b1", "b2"]]


def test_oversized_operation_gets_own_chunk() -> None:
    ops = [make_op("small", "a"), make_op("huge", "a", padding=500), make_op("small2", "a")]
    chunks = chunk_operations(ops, max_chars=200)
    assert [[op.id for op in chunk] for chunk in chunks] == [["small"], ["huge"], ["small2"]]


def test_whole_petstore_fits_one_default_chunk() -> None:
    assert len(chunk_operations(load_spec(PETSTORE).operations)) == 1
