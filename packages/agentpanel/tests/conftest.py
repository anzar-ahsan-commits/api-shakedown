import os

import pytest


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]) -> None:
    """Keep the developer's real env vars and .env file out of tests."""
    for name in list(os.environ):
        if name.startswith(("AGENTPANEL_", "SHAKEDOWN_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
