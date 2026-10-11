"""Validate JSON values against inlined OpenAPI schemas."""

from typing import Any, Literal

from openapi_schema_validator import OAS30ReadValidator, OAS30WriteValidator, OAS31Validator

MAX_ERRORS = 5


def schema_errors(
    schema: dict[str, Any],
    instance: Any,
    openapi_version: str,
    direction: Literal["read", "write"] = "read",
) -> list[str]:
    """Problems with `instance`, or [] if valid. `read` = response, `write` = request.

    Schemas that cannot be checked (e.g. a circular `$ref` left in place) count as valid.
    """
    if openapi_version.startswith("3.0"):
        validator: Any = OAS30ReadValidator if direction == "read" else OAS30WriteValidator
    else:
        validator = OAS31Validator
    try:
        errors = list(validator(schema).iter_errors(instance))
    except Exception:
        return []
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '(root)'}: {error.message}"
        for error in errors[:MAX_ERRORS]
    ]
