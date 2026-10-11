"""The generated harness, started with its own docker-compose.yml and used as a user
would: test suites against the stubs, chaos switched on and off. Opt in with
`pytest -m live`."""

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.live

PETSTORE = Path(__file__).parents[1] / "examples" / "petstore.yaml"
PORT = "18080"  # away from the default, in case something already uses 8080


def shakedown_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    exe = Path(sys.executable).with_name("shakedown")
    return subprocess.run([str(exe), *args], capture_output=True, text=True, cwd=cwd)


def compose(harness: Path, *args: str) -> None:
    env = os.environ | {"SHAKEDOWN_WIREMOCK_PORT": PORT}
    subprocess.run(
        ["docker", "compose", "-p", f"shakedown-{harness.parent.name.lower()}", *args],
        cwd=harness, env=env, check=True, capture_output=True,
    )  # fmt: skip


def started(harness: Path) -> Iterator[httpx.Client]:
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed.")
    compose(harness, "up", "-d")
    try:
        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=10)
        deadline = time.monotonic() + 60
        while True:
            try:
                if client.get("/__admin/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("WireMock from docker-compose did not become healthy.")
            time.sleep(0.5)
        yield client
    finally:
        compose(harness, "down")


def exercise(harness: Path) -> None:
    """What the README tells users to do."""
    for client in started(harness):
        env = os.environ | {
            "SHAKEDOWN_TARGET_URL": f"http://127.0.0.1:{PORT}/v1",
            "SHAKEDOWN_SELECT_STATUS": "1",
        }
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
            capture_output=True, text=True, env=env, cwd=harness,
        )  # fmt: skip
        print(tests.stdout[-1500:])
        assert tests.returncode == 0

        chaos = [sys.executable, "chaos/chaos.py", "--wiremock", f"http://127.0.0.1:{PORT}"]
        subprocess.run([*chaos, "on", "flaky-start"], cwd=harness, check=True)
        assert [client.get("/v1/pets").status_code for _ in range(3)] == [503, 503, 200]
        subprocess.run([*chaos, "off"], cwd=harness, check=True)
        assert client.get("/v1/pets").status_code == 200


def test_offline_harness_runs_with_docker_compose(tmp_path: Path) -> None:
    result = shakedown_cli("run", str(PETSTORE), "--out", "harness", "--offline", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    exercise(tmp_path / "harness")


def test_llm_harness_runs_with_docker_compose(tmp_path: Path) -> None:
    started_at = time.monotonic()
    result = shakedown_cli("run", str(PETSTORE), "--out", "harness", cwd=tmp_path)
    print(f"\n{result.stdout}\n{result.stderr}")
    if result.returncode == 1 and "use --offline" in result.stderr:
        pytest.skip("No LLM provider available.")
    assert result.returncode == 0
    print(f"shakedown run took {time.monotonic() - started_at:.0f}s")
    print((tmp_path / "harness" / "README.md").read_text(encoding="utf-8")[-2500:])
    exercise(tmp_path / "harness")
