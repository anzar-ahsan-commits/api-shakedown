"""Normalized, prompt-friendly view of an OpenAPI spec. `$ref`s are already inlined."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

ParameterLocation = Literal["path", "query", "header", "cookie"]


class Parameter(BaseModel):
    name: str
    location: ParameterLocation
    required: bool = False
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


class Body(BaseModel):
    """A request or response body, using its preferred media type (JSON when available)."""

    content_type: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    example: Any = None
    other_content_types: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class RequestBody(Body):
    required: bool = False


class Response(BaseModel):
    status: str  # "200", "4XX", or "default"
    description: str | None = None
    headers: dict[str, dict[str, Any]] = Field(default_factory=dict)  # name -> schema
    body: Body | None = None


class Operation(BaseModel):
    id: str  # operationId, or derived from method + path; unique within the spec
    method: str  # lowercase
    path: str
    summary: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    request_body: RequestBody | None = None
    responses: list[Response] = Field(default_factory=list)
    security: list[str] = Field(default_factory=list)  # names of required security schemes
    deprecated: bool = False

    def to_prompt(self) -> str:
        """Compact JSON for LLM prompts: empty and default fields are dropped."""
        data = self.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        data = {"id": self.id, "method": self.method, "path": self.path, **data}
        return json.dumps(data, separators=(",", ":"), default=str)


class ApiSpec(BaseModel):
    title: str
    version: str
    openapi_version: str
    description: str | None = None
    servers: list[str] = Field(default_factory=list)
    security_schemes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)  # components, inlined
    operations: list[Operation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def operation(self, operation_id: str) -> Operation:
        for op in self.operations:
            if op.id == operation_id:
                return op
        raise KeyError(operation_id)
