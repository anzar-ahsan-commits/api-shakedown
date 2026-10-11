# Releasing

Two distributions are released together with the same version:
`agentpanel` (import `agentpanel`) and `api-shakedown` (which depends on it).

## 1. Prepare

1. Set the version in both `src/api_shakedown/__init__.py` and
   `packages/agentpanel/src/agentpanel/__init__.py`, and update the
   `agentpanel>=X,<Y` range in `pyproject.toml` if the core's minor version changed.
2. Run the checks: `ruff check . && mypy && pytest && pytest -m live`.
3. Build and check locally:

   ```bash
   rm -rf dist
   python -m build packages/agentpanel --outdir dist
   python -m build . --outdir dist
   twine check --strict dist/*
   ```

## 2. Try TestPyPI first (recommended for the first release)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ api-shakedown
shakedown run examples/petstore.yaml --out /tmp/harness --offline
```

## 3. Publish

Either upload by hand (`twine upload dist/*`), or let CI do it:

1. One-time: on PyPI, add a *trusted publisher* for each project (`agentpanel`,
   `api-shakedown`) pointing at this GitHub repository, workflow `release.yml`,
   environment `pypi`. For a brand-new project use a "pending publisher".
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.

The workflow fails if the tag does not match both package versions. A version can
never be re-uploaded to PyPI, so bump it if anything goes wrong after upload.
