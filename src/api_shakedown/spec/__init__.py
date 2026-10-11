"""OpenAPI spec loading, validation, and normalisation."""

from pathlib import Path

from api_shakedown.spec.chunking import chunk_operations
from api_shakedown.spec.loader import SpecError, check_document, read_source
from api_shakedown.spec.models import ApiSpec, Operation
from api_shakedown.spec.normalize import normalize


def load_spec(source: str | Path) -> ApiSpec:
    """Load, validate, and normalize a spec. Validation issues become `ApiSpec.warnings`."""
    document = read_source(source)
    return normalize(document, warnings=check_document(document))


__all__ = ["ApiSpec", "Operation", "SpecError", "chunk_operations", "load_spec"]
