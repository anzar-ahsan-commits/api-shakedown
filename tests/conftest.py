import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from api_shakedown.wiremock import StubSet

WIREMOCK_IMAGE = "wiremock/wiremock:3.13.1"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]) -> None:
    """Keep the developer's real env vars and .env file out of tests."""
    for name in list(os.environ):
        if name.startswith(("AGENTPANEL_", "SHAKEDOWN_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def _request_matches(spec: dict[str, Any], handler: BaseHTTPRequestHandler) -> bool:
    url = urlsplit(handler.path)
    query = parse_qs(url.query)
    if spec["method"] != handler.command or not re.fullmatch(spec["urlPathPattern"], url.path):
        return False
    for name, matcher in spec.get("headers", {}).items():
        if handler.headers.get(name) != matcher["equalTo"]:
            return False
    return all(
        name in query and re.fullmatch(matcher["matches"], query[name][0])
        for name, matcher in spec.get("queryParameters", {}).items()
    )


class _StubHandler(BaseHTTPRequestHandler):
    """Implements the subset of WireMock matching that our generated stubs use."""

    mappings: ClassVar[list[dict[str, Any]]] = []
    received: ClassVar[list[tuple[str, str, dict[str, str]]]] = []

    def _handle(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.received.append((self.command, self.path, dict(self.headers)))
        candidates = [m for m in self.mappings if _request_matches(m["request"], self)]
        if not candidates:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Request was not matched")
            return
        response = min(candidates, key=lambda m: m.get("priority", 5))["response"]
        if "jsonBody" in response:
            payload = json.dumps(response["jsonBody"]).encode()
        else:
            payload = str(response.get("body", "")).encode()
        self.send_response(response["status"])
        for name, value in response.get("headers", {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def serve_stubs() -> Iterator[Callable[[StubSet], tuple[str, type[_StubHandler]]]]:
    """Serve a StubSet over HTTP; returns (base URL, handler class with `received`)."""
    servers: list[ThreadingHTTPServer] = []

    def start(stubs: StubSet) -> tuple[str, type[_StubHandler]]:
        handler = type("Handler", (_StubHandler,), {"mappings": stubs.mappings, "received": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}", handler

    yield start
    for server in servers:
        server.shutdown()


@pytest.fixture(scope="session")
def wiremock() -> Iterator[httpx.Client]:
    """A real WireMock in Docker (live tests only)."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed.")
    container = subprocess.run(
        ["docker", "run", "-d", "--rm", "-p", "127.0.0.1::8080", WIREMOCK_IMAGE],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    try:
        port_line = subprocess.run(
            ["docker", "port", container, "8080"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        client = httpx.Client(base_url=f"http://{port_line.strip()}", timeout=10)
        deadline = time.monotonic() + 60
        while True:
            try:
                if client.get("/__admin/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("WireMock did not become healthy within 60s.")
            time.sleep(0.5)
        yield client
    finally:
        subprocess.run(["docker", "stop", container], capture_output=True)
