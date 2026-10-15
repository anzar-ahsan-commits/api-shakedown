# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/anzar-ahsan-commits/api-shakedown/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/anzar-ahsan-commits/api-shakedown/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anzar-ahsan-commits/api-shakedown/releases/tag/v0.1.0
