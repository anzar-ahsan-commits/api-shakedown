# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private
reporting instead: on the repository page, go to **Security > Report a vulnerability**.

You should get a reply within a week. Please include steps to reproduce and the version
(`shakedown version`).

## Supported versions

Only the latest release receives fixes while the project is in alpha (0.x).

## Things to know

- **Specs are untrusted input.** API Shakedown parses YAML with a safe loader, caps spec
  size at 20 MB, and does not follow references to other files or URLs.
- **Spec content is sent to the configured LLM.** With the default (local Ollama) nothing
  leaves your machine; with a cloud provider, the spec and generated artifacts are sent
  to that provider.
- **Generated harnesses run code you can read.** LLMs never write executable code: they
  produce data (JSON) that fixed, reviewed runners execute.
- **Robustness probes send malformed and injection-style input** (for example SQL and
  script snippets) to the target you point them at. Only run them against systems you are
  allowed to test.
- **API keys** are read from the environment or `.env` and are never written to
  harnesses or logs. Keep `.env` out of version control (it is in `.gitignore`).
