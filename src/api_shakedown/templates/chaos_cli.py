"""Switch API Shakedown chaos scenarios on and off in a running WireMock.

    python chaos.py list
    python chaos.py on <scenario-id>        # replaces any active scenario
    python chaos.py off                     # back to normal stubs

The WireMock admin URL comes from --wiremock, SHAKEDOWN_WIREMOCK_URL, or
chaos_scenarios.json. Standard library only. Also importable: `activate`, `deactivate`.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

SCENARIOS: dict[str, Any] = json.loads(
    Path(__file__).with_name("chaos_scenarios.json").read_text(encoding="utf-8")
)
_ALL_CHAOS = {"matchesJsonPath": "$.shakedown.scenario"}


def _admin(wiremock: str, path: str, body: Any = None) -> None:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        wiremock.rstrip("/") + "/__admin" + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def scenario(scenario_id: str) -> dict[str, Any]:
    for item in SCENARIOS["scenarios"]:
        if item["id"] == scenario_id:
            return dict(item)
    known = ", ".join(s["id"] for s in SCENARIOS["scenarios"])
    raise KeyError(f"Unknown scenario {scenario_id!r}. Known: {known}")


def deactivate(wiremock: str) -> None:
    """Remove every chaos mapping and reset WireMock's scenario state machines."""
    _admin(wiremock, "/mappings/remove-by-metadata", _ALL_CHAOS)
    _admin(wiremock, "/scenarios/reset")


def activate(wiremock: str, scenario_id: str) -> int:
    """Turn one scenario on (any other is switched off first). Returns mappings added."""
    mappings = scenario(scenario_id)["mappings"]
    deactivate(wiremock)
    for mapping in mappings:
        _admin(wiremock, "/mappings", mapping)
    return len(mappings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wiremock", default=None, help="WireMock base URL")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show available scenarios")
    on = sub.add_parser("on", help="activate a scenario")
    on.add_argument("scenario")
    sub.add_parser("off", help="deactivate all scenarios")
    args = parser.parse_args(argv)
    wiremock = (
        args.wiremock or os.environ.get("SHAKEDOWN_WIREMOCK_URL") or SCENARIOS["wiremock_url"]
    )

    if args.command == "list":
        for item in SCENARIOS["scenarios"]:
            print(f"{item['id']:28} [{item['category']}] {item['title']}")
            print(f"{'':28} expect: {item['expectation']}")
        return 0
    if args.command == "on":
        try:
            count = activate(wiremock, args.scenario)
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 2
        print(f"Activated {args.scenario} ({count} mappings) on {wiremock}")
        return 0
    deactivate(wiremock)
    print(f"Chaos off on {wiremock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
