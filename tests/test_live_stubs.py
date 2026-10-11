"""Serve generated stubs from a real WireMock (Docker) and, for the LLM path, a real
local Ollama. Opt in with `pytest -m live`."""

import asyncio
from pathlib import Path

import httpx
import pytest
from agentpanel import AgentContext
from agentpanel.llm import check_provider, create_chat_model

from api_shakedown.agents.stubs import StubAgent
from api_shakedown.config import ShakedownSettings
from api_shakedown.spec import ApiSpec, load_spec
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.wiremock import STATUS_HEADER, StubSet

pytestmark = pytest.mark.live

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"


@pytest.fixture(scope="module")
def spec() -> ApiSpec:
    return load_spec(PETSTORE)


def generate(spec: ApiSpec, agent: StubAgent) -> StubSet:
    ctx = AgentContext(inputs={"spec": spec}, artifacts={}, reviews=[], round=0)
    return asyncio.run(agent.produce(ctx))


def load_into(client: httpx.Client, stubs: StubSet) -> None:
    client.post("/__admin/reset").raise_for_status()
    client.post("/__admin/mappings/import", json=stubs.to_wiremock()).raise_for_status()
    assert len(client.get("/__admin/mappings").json()["mappings"]) == len(stubs.mappings)


def test_wiremock_serves_offline_stubs(wiremock: httpx.Client, spec: ApiSpec) -> None:
    load_into(wiremock, generate(spec, StubAgent()))

    listing = wiremock.get("/v1/pets")
    assert listing.status_code == 200 and listing.json()[0]["name"] == "Rex"
    assert listing.headers["X-Next-Cursor"]

    assert wiremock.get("/v1/pets/42").status_code == 200
    missing = wiremock.get("/v1/pets/42", headers={STATUS_HEADER: "404"})
    assert missing.status_code == 404 and set(missing.json()) == {"code", "message"}

    unmatched = wiremock.get("/v1/pets/not-a-number")
    assert unmatched.status_code == 404 and "not matched" in unmatched.text

    assert wiremock.delete("/v1/pets/1").status_code == 204
    assert wiremock.post("/v1/pets", json={"name": "x"}).status_code == 201
    assert wiremock.post("/v1/pets", headers={STATUS_HEADER: "400"}).status_code == 400
    assert wiremock.get("/v1/pets", headers={STATUS_HEADER: "default"}).status_code == 500
    health = wiremock.get("/v1/health")
    assert health.text == "ok" and health.headers["Content-Type"].startswith("text/plain")


def test_llm_stubs_are_valid_and_served(wiremock: httpx.Client, spec: ApiSpec) -> None:
    settings = ShakedownSettings()
    status = check_provider(settings)
    if not status.ok:
        pytest.skip(status.detail)

    stubs = generate(spec, StubAgent(model=create_chat_model(settings.for_agent("stubs"))))
    fallbacks = [note for note in stubs.notes if "schema-generated sample" in note]
    print("\nnotes:", *stubs.notes, sep="\n  ")
    assert len(fallbacks) <= len(stubs.mappings) // 2, "the LLM should supply most bodies"

    load_into(wiremock, stubs)
    pet = wiremock.get("/v1/pets/1")
    schema = spec.operation("getPet").responses[0].body.schema_  # type: ignore[union-attr]
    assert schema is not None
    assert schema_errors(schema, pet.json(), spec.openapi_version) == []
    print("GET /v1/pets/1 ->", pet.json())
