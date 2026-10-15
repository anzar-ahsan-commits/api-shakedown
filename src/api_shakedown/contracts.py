"""Contract test suites: test cases as data, executed by a fixed pytest runner.

The LLM only proposes case inputs (`CaseProposal`); code checks them against the spec,
adds auth and the response contract, and ships them as JSON next to the runner.
"""

import json
import re
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, Field

from api_shakedown.spec.models import ApiSpec, Operation, Parameter
from api_shakedown.spec.params import query_pairs
from api_shakedown.spec.samples import sample_from_schema, valid_sample
from api_shakedown.spec.schema_check import schema_errors
from api_shakedown.testdata import Dataset
from api_shakedown.wiremock import DEFAULT_STUB_URL, STATUS_HEADER, base_path, happy_response

RUNNER_FILE = "test_contract.py"
CASES_FILE = "contract_cases.json"
SUPPORT_FILE = "shakedown_support.py"
# Headers the runner controls; proposals may not set them.
_RESERVED_HEADERS = {"authorization", "content-type", "content-length", "host", "cookie"}


class CaseProposal(BaseModel):
    """One test case as proposed by the LLM."""

    operation_id: str
    name: str = Field(description="snake_case summary, e.g. returns_404_for_unknown_id")
    expected_status: str = Field(description="One of the operation's documented statuses.")
    path_params: dict[str, Any] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None


class AuthRequirement(BaseModel):
    scheme: Literal["apiKey", "bearer", "basic"]
    location: Literal["header", "query", "cookie"] = "header"
    name: str = "Authorization"


class ContractCase(BaseModel):
    """A fully resolved case, as the runner consumes it."""

    id: str
    operation_id: str
    method: str
    path: str
    expected_status: str
    other_statuses: list[str]
    path_params: dict[str, Any]
    query: dict[str, Any]
    query_pairs: list[list[str]] = Field(default_factory=list)  # as sent on the wire
    headers: dict[str, str]
    body: Any = None
    body_content_type: str | None = None
    auth: list[AuthRequirement] = Field(default_factory=list)
    response_content_type: str | None = None
    response_schema: dict[str, Any] | None = None


class ContractSuite(BaseModel):
    title: str
    version: str
    openapi_version: str
    default_target: str
    cases: list[ContractCase]
    notes: list[str] = Field(default_factory=list)

    def files(self) -> dict[str, str]:
        """Filename -> content for the runnable suite."""
        data = self.model_dump(exclude={"notes"}) | {"status_header": STATUS_HEADER}
        return {
            RUNNER_FILE: template("contract_runner.py"),
            SUPPORT_FILE: template(SUPPORT_FILE),
            CASES_FILE: json.dumps(data, indent=2, default=str) + "\n",
        }


def template(name: str) -> str:
    return files("api_shakedown.templates").joinpath(name).read_text("utf-8")


def proposal_problems(
    spec: ApiSpec, op: Operation, proposal: CaseProposal, dataset: Dataset | None = None
) -> list[str]:
    """Why a proposal can't be used, or [] if it can."""
    documented = [r.status for r in op.responses]
    problems = []
    if proposal.expected_status not in documented:
        problems.append(f"expected_status must be one of {documented}.")
    if op.request_body is None and proposal.body is not None:
        problems.append(f"{op.method.upper()} {op.path} takes no request body; set body to null.")

    path_params = {p.name: p for p in op.parameters if p.location == "path"}
    query_params = {p.name: p for p in op.parameters if p.location == "query"}
    if missing := sorted(path_params.keys() - proposal.path_params.keys()):
        problems.append(f"missing path_params {missing}.")
    if unknown := sorted(proposal.path_params.keys() - path_params.keys()):
        problems.append(f"unknown path_params {unknown}; the operation has {sorted(path_params)}.")
    if unknown := sorted(proposal.query.keys() - query_params.keys()):
        problems.append(f"unknown query params {unknown}; documented: {sorted(query_params)}.")
    required_query = {name for name, p in query_params.items() if p.required}
    if missing := sorted(required_query - proposal.query.keys()):
        problems.append(f"missing required query params {missing}.")

    # Path values must always be valid (or the URL won't route); other inputs only for
    # success cases, since error cases may send bad input on purpose.
    for name, value in proposal.path_params.items():
        if (param := path_params.get(name)) and param.schema_:
            problems += [
                f"path param {name}={value!r} must be valid even in error cases: {error}"
                for error in schema_errors(param.schema_, value, spec.openapi_version, "write")
            ]
        known = dataset.known_ids(op.path, name) if dataset else None
        if known:
            exists = str(value) in {str(k) for k in known}
            if proposal.expected_status.startswith("2") and not exists:
                problems.append(f"path param {name}={value!r} must be an existing id: {known}.")
            if proposal.expected_status == "404" and exists:
                problems.append(f"path param {name}={value!r} exists; a 404 case needs another id.")
    if proposal.expected_status.startswith("2"):
        for name, value in proposal.query.items():
            if (param := query_params.get(name)) and param.schema_:
                problems += [
                    f"query {name}={value!r}: {error}"
                    for error in schema_errors(param.schema_, value, spec.openapi_version, "write")
                ]
        body = op.request_body
        if body is not None and body.required and proposal.body is None:
            problems.append("a success case needs a request body.")
        if body is not None and body.schema_ is not None and proposal.body is not None:
            problems += [
                f"request body: {error}"
                for error in schema_errors(
                    body.schema_, proposal.body, spec.openapi_version, "write"
                )
            ]
    return problems


def happy_proposal(
    op: Operation, dataset: Dataset | None = None, openapi_version: str = "3.1.0"
) -> CaseProposal | None:
    """Deterministic happy-path case from schema samples (and dataset ids when bound)."""
    happy = happy_response(op)
    if happy is None:
        return None

    def path_value(param: Parameter) -> Any:
        known = dataset.known_ids(op.path, param.name) if dataset else None
        return known[0] if known else sample(param.schema_)

    def sample(schema: dict[str, Any] | None) -> Any:
        value = sample_from_schema(schema)
        return 1 if value is None else value

    body = None
    if op.request_body is not None:
        body = valid_sample(
            op.request_body.schema_, op.request_body.example, openapi_version, "write"
        )
    return CaseProposal(
        operation_id=op.id,
        name="happy_path",
        expected_status=happy.status,
        path_params={p.name: path_value(p) for p in op.parameters if p.location == "path"},
        query={
            p.name: sample(p.schema_) for p in op.parameters if p.location == "query" and p.required
        },
        body=body,
    )


def build_suite(
    spec: ApiSpec, proposals: list[CaseProposal], notes: list[str] | None = None
) -> ContractSuite:
    ops = {op.id: op for op in spec.operations}
    used_ids: set[str] = set()
    cases = []
    for proposal in proposals:
        op = ops[proposal.operation_id]
        cases.append(_case(spec, op, proposal, _unique_id(op, proposal, used_ids)))
    return ContractSuite(
        title=spec.title,
        version=spec.version,
        openapi_version=spec.openapi_version,
        default_target=DEFAULT_STUB_URL + base_path(spec),
        cases=cases,
        notes=list(notes or []),
    )


def auth_requirements(spec: ApiSpec, op: Operation) -> list[AuthRequirement]:
    requirements = []
    for name in op.security:
        scheme = spec.security_schemes.get(name) or {}
        kind = scheme.get("type")
        if kind == "apiKey" and scheme.get("in") in ("header", "query", "cookie"):
            requirements.append(
                AuthRequirement(scheme="apiKey", location=scheme["in"], name=scheme["name"])
            )
        elif kind == "http" and str(scheme.get("scheme", "")).lower() == "basic":
            requirements.append(AuthRequirement(scheme="basic"))
        elif kind in ("http", "oauth2", "openIdConnect"):
            requirements.append(AuthRequirement(scheme="bearer"))
    return requirements


def _case(spec: ApiSpec, op: Operation, proposal: CaseProposal, case_id: str) -> ContractCase:
    auth = auth_requirements(spec, op)
    reserved = (
        _RESERVED_HEADERS
        | {STATUS_HEADER.lower()}
        | {a.name.lower() for a in auth if a.location == "header"}
    )
    response = next(r for r in op.responses if r.status == proposal.expected_status)
    return ContractCase(
        id=case_id,
        operation_id=op.id,
        method=op.method,
        path=op.path,
        expected_status=proposal.expected_status,
        other_statuses=[r.status for r in op.responses if r is not response],
        path_params=proposal.path_params,
        query=proposal.query,
        query_pairs=query_pairs(op, proposal.query),
        headers={k: v for k, v in proposal.headers.items() if k.lower() not in reserved},
        body=proposal.body,
        body_content_type=op.request_body.content_type if op.request_body else None,
        auth=auth,
        response_content_type=response.body.content_type if response.body else None,
        response_schema=response.body.schema_ if response.body else None,
    )


def _unique_id(op: Operation, proposal: CaseProposal, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", proposal.name.lower()).strip("_") or "case"
    base = f"{op.id}-{proposal.expected_status}-{slug}"
    case_id, n = base, 2
    while case_id in used:
        case_id, n = f"{base}_{n}", n + 1
    used.add(case_id)
    return case_id
