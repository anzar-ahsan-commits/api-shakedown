"""Read a spec from a file path or URL, and validate it."""

from pathlib import Path
from typing import Any, cast

import httpx
import yaml
from openapi_spec_validator import (
    OpenAPIV30SpecValidator,
    OpenAPIV31SpecValidator,
    OpenAPIV32SpecValidator,
)

MAX_SPEC_BYTES = 20 * 1024 * 1024
MAX_REPORTED_ERRORS = 20

_VALIDATORS: dict[str, Any] = {
    "3.0": OpenAPIV30SpecValidator,
    "3.1": OpenAPIV31SpecValidator,
    "3.2": OpenAPIV32SpecValidator,
}


class SpecError(Exception):
    """The spec cannot be used. The message says why."""


def read_source(source: str | Path) -> dict[str, Any]:
    """Load a spec document from a local path or http(s) URL. Accepts YAML or JSON."""
    text = _fetch(str(source)) if _is_url(source) else _read_file(Path(source))
    try:
        document = yaml.safe_load(text)  # YAML is a superset of JSON
    except yaml.YAMLError as exc:
        raise SpecError(f"{source} is not valid YAML or JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecError(f"{source} does not contain an OpenAPI document (expected a mapping).")
    return cast(dict[str, Any], _stringify_keys(document))


def check_document(document: dict[str, Any]) -> list[str]:
    """Raise SpecError if the document is unusable; return validation issues as warnings."""
    if "swagger" in document:
        raise SpecError(
            "Swagger 2.0 specs are not supported. Convert to OpenAPI 3 first, e.g. with "
            "https://converter.swagger.io or `npx swagger2openapi`."
        )
    version = str(document.get("openapi", ""))
    if not version:
        raise SpecError("Missing `openapi` version field; is this an OpenAPI document?")
    if not isinstance(document.get("paths"), dict) or not document["paths"]:
        raise SpecError("The spec defines no paths, so there is nothing to test.")

    validator = _VALIDATORS.get(version[:3])
    if validator is None:
        return [f"Unknown OpenAPI version {version}; it was not validated."]
    try:
        errors = list(validator(document).iter_errors())
    except Exception as exc:  # the validator can fail outright on e.g. broken $refs
        return [f"Validation could not complete: {exc}"]

    warnings = [
        f"{'/'.join(str(p) for p in error.absolute_path) or '(root)'}: {error.message}"
        for error in errors[:MAX_REPORTED_ERRORS]
    ]
    if len(errors) > MAX_REPORTED_ERRORS:
        warnings.append(f"... and {len(errors) - MAX_REPORTED_ERRORS} more validation issues.")
    return warnings


def _is_url(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _read_file(path: Path) -> str:
    if not path.is_file():
        raise SpecError(f"Spec file not found: {path}")
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise SpecError(f"Spec file is larger than {MAX_SPEC_BYTES // (1024 * 1024)} MB.")
    return path.read_text(encoding="utf-8")


def _fetch(url: str) -> str:
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_SPEC_BYTES:
                    raise SpecError(f"Spec at {url} exceeds {MAX_SPEC_BYTES // (1024 * 1024)} MB.")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise SpecError(f"Could not download {url}: {exc}") from exc
    return b"".join(chunks).decode(response.encoding or "utf-8")


def _stringify_keys(node: Any) -> Any:
    """YAML turns unquoted `200:` into an int key; OpenAPI keys are always strings."""
    if isinstance(node, dict):
        return {str(k): _stringify_keys(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_stringify_keys(v) for v in node]
    return node
