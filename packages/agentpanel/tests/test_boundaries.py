"""The core must stay reusable: it may not depend on any application package."""

import ast
from pathlib import Path

import agentpanel

FORBIDDEN = ("api_shakedown",)
CORE_ROOT = Path(agentpanel.__file__).parent


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_applications() -> None:
    offenders = {
        str(path.relative_to(CORE_ROOT)): name
        for path in CORE_ROOT.rglob("*.py")
        for name in _imported_modules(path)
        if name.split(".")[0] in FORBIDDEN
    }
    assert not offenders, f"agentpanel must not import application code: {offenders}"
