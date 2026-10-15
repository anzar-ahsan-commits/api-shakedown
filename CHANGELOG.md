# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-10-15

Found by running the Typesense spec (79 operations) against its stubs and a real server.

### Fixed

- Overlapping routes: `/documents/export` is no longer answered by the `/documents/{id}`
  stub. A more specific route (fewer path parameters) always wins, including during chaos.
- Query parameters follow OpenAPI serialization (`form`/`explode`, `deepObject`, arrays,
  `content`): object parameters are sent as separate keys, not as one value, and stubs no
  longer require an exploded object's name in the URL.
- Spec examples that contradict their own schemas (JSON written as text, nested examples
  of the wrong type) are repaired or replaced with valid samples, in stubs, tests and test
  data, and listed in the harness's generation notes.
- Test data on large specs: only resources (schemas with a key) get LLM-written records;
  response envelopes and schemas too large for one prompt get valid schema samples. On
  Typesense this cut the test-data step from 8 LLM calls, two of them stalling on 34 KB
  schemas, to 2.

- Local model replies are capped (`max_output_tokens`, default 8192). A small model can
  fall into a repetition loop, and Ollama then shifts its context and generates forever,
  which hung a whole run; the cap turns that into a bounded, retryable failure.
- The "Test data ready" progress line summarises big specs instead of listing every
  schema.

### Added

- Resources keyed by name or slug are recognised from their path (e.g. `name` for
  `/collections/{collectionName}`), so stubs answer `GET /collections/books` with the
  `books` record. When several operations share a path, the GET defines its resource.

## [0.1.1] - 2026-10-14

### Added

- Listed in the official MCP Registry as `io.github.anzar-ahsan-commits/api-shakedown`.
- `shakedown mcp` subcommand and an `api-shakedown` command, so MCP clients can run
  `uvx api-shakedown mcp` with nothing installed.
- Demo recording in the README.

### Changed

- The MCP server is always installed (the `[mcp]` extra is kept as a no-op).
- The reviewer reserves blockers for items that are wrong or misleading; debatable
  points become suggestions, avoiding wasted revision rounds on small models.
- `agentpanel` is re-released unchanged so both packages share the version.

## [0.1.0] - 2026-10-11

First public release of `api-shakedown` and `agentpanel`.

### Added

- `shakedown run`: OpenAPI spec in, ready-to-run harness out (WireMock stubs via
  docker-compose, contract tests, robustness probes, chaos scenarios, test data, README).
- Agents: test data, stubs, contract tests, chaos scenarios, reviewer, coordinator, with a
  bounded review/revise loop.
- Local-first LLM support (Ollama, with constrained decoding), plus OpenAI and Anthropic
  with your own key; per-agent model and reasoning settings.
- `--offline` mode: deterministic harnesses without any LLM.
- `shakedown doctor` and `shakedown inspect`.
- MCP server (`shakedown-mcp`) with tools to inspect specs, generate harnesses, and switch
  chaos scenarios on and off.
- Docker image with the CLI and the MCP server.
- `agentpanel`: the reusable multi-agent core (workflow, LLM agents, structured output,
  provider factory, test helpers), published as its own package.

[Unreleased]: https://github.com/anzar-ahsan-commits/api-shakedown/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/anzar-ahsan-commits/api-shakedown/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/anzar-ahsan-commits/api-shakedown/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anzar-ahsan-commits/api-shakedown/releases/tag/v0.1.0
