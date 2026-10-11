"""Switch a generated harness's chaos scenarios on and off in a running WireMock.

The same behaviour as the harness's standalone `chaos/chaos.py`, as a library (used by
the MCP server).
"""

import json
from pathlib import Path
from typing import Any

import httpx

from api_shakedown.chaos import SCENARIOS_FILE

_ALL_CHAOS = {"matchesJsonPath": "$.shakedown.scenario"}


class ChaosControlError(Exception):
    pass


def load_scenarios(harness_dir: Path) -> dict[str, Any]:
    path = harness_dir / "chaos" / SCENARIOS_FILE
    if not path.is_file():
        raise ChaosControlError(f"No chaos scenarios at {path}; is this a generated harness?")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def deactivate(wiremock_url: str) -> None:
    """Remove every chaos mapping and reset WireMock's scenario state machines."""
    with httpx.Client(base_url=wiremock_url.rstrip("/") + "/__admin", timeout=10) as admin:
        _post(admin, "/mappings/remove-by-metadata", _ALL_CHAOS)
        _post(admin, "/scenarios/reset")


def activate(wiremock_url: str, scenarios: dict[str, Any], scenario_id: str) -> int:
    """Turn one scenario on (any other is switched off first). Returns mappings added."""
    match = next((s for s in scenarios["scenarios"] if s["id"] == scenario_id), None)
    if match is None:
        known = ", ".join(s["id"] for s in scenarios["scenarios"])
        raise ChaosControlError(f"Unknown scenario {scenario_id!r}. Known: {known}")
    deactivate(wiremock_url)
    with httpx.Client(base_url=wiremock_url.rstrip("/") + "/__admin", timeout=10) as admin:
        for mapping in match["mappings"]:
            _post(admin, "/mappings", mapping)
    return len(match["mappings"])


def _post(admin: httpx.Client, path: str, body: Any = None) -> None:
    try:
        admin.post(path, json=body).raise_for_status()
    except httpx.HTTPError as exc:
        raise ChaosControlError(f"WireMock admin call {path} failed: {exc}") from exc
