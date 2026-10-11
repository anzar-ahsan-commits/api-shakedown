"""Inline internal `$ref`s so each operation is self-contained in a prompt.

Circular references are kept as `$ref` (marked `x-circular`) so output stays finite.
External references (other files/URLs) are not followed; they are reported as warnings.
"""

from typing import Any
from urllib.parse import unquote

MAX_DEPTH = 12


class RefResolver:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.warnings: list[str] = []

    def resolve(self, node: Any, _stack: tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [self.resolve(item, _stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return {key: self.resolve(value, _stack) for key, value in node.items()}

        if ref in _stack:
            return {"$ref": ref, "x-circular": True}
        if len(_stack) >= MAX_DEPTH:
            return {"$ref": ref, "x-truncated": True}
        target = self._lookup(ref)
        if target is None:
            return {"$ref": ref}
        resolved = self.resolve(target, (*_stack, ref))
        # OpenAPI 3.1 allows siblings next to $ref (e.g. a local description); they win.
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        if siblings and isinstance(resolved, dict):
            resolved = {**resolved, **self.resolve(siblings, _stack)}
        return resolved

    def _lookup(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            self._warn(f"External $ref not followed: {ref}")
            return None
        node: Any = self.document
        for raw_part in ref[2:].split("/"):
            part = unquote(raw_part).replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                self._warn(f"Unresolvable $ref: {ref}")
                return None
        return node

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
