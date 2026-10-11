# Contributing to API Shakedown

Thanks for helping! Bug reports, specs that break it, and pull requests are all welcome.

## Reporting a problem

Open an [issue](https://github.com/anzar-ahsan-commits/api-shakedown/issues) with:

- what you ran (`shakedown run ...`, options, provider and model from `shakedown doctor`);
- what happened and what you expected;
- the spec, or the smallest part of it that reproduces the problem (remove anything
  private). Specs that make the agents struggle are especially useful.

## Development setup

```bash
git clone https://github.com/anzar-ahsan-commits/api-shakedown.git
cd api-shakedown
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e packages/agentpanel -e ".[dev]"
```

Before opening a pull request:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

`pytest` is offline and fast: LLMs are replaced by `agentpanel.testing.ScriptedChatModel`.
If your change affects prompts or model behaviour, also run `pytest -m live`, which needs
Ollama with `qwen3:8b` and Docker.

## How the code is organised

- `packages/agentpanel/`: the reusable orchestration core. It must stay domain-agnostic
  and never import `api_shakedown` (`tests/test_boundaries.py` enforces this).
- `src/api_shakedown/`: the application. Agents live in `agents/`; the code that checks
  their output lives next to the domain it checks (`spec/`, `wiremock.py`,
  `contracts.py`, `chaos.py`, `testdata.py`).
- `src/api_shakedown/templates/`: files copied verbatim into every harness. They run
  in users' environments, so keep their dependencies minimal (`chaos_cli.py` uses only
  the standard library).

## Principles for changes

- **LLMs propose data, code decides.** New agent output should be validated in code,
  with problems fed back to the model and a deterministic fallback.
- **Everything works offline.** `--offline` must keep producing a usable harness.
- **Tests for behaviour, not prompts.** Assert on what code guarantees; use live tests
  for model quality.
- Match the surrounding style: type hints everywhere (mypy strict), small functions,
  docstrings that say why.

## Pull requests

Keep them focused, describe the change and how you tested it, and add an entry under
"Unreleased" in [CHANGELOG.md](CHANGELOG.md). By contributing you agree that your
contribution is licensed under the [MIT License](LICENSE).
