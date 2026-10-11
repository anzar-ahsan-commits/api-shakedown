# Running API Shakedown in Docker

The image contains `shakedown`, the MCP server (`shakedown-mcp`), the OpenAI and
Anthropic extras, and the example spec at `/opt/shakedown/examples/petstore.yaml`.
It runs as a non-root user (uid 1000) in `/work`; mount the folder you want to write to.

```bash
docker build -t api-shakedown .
```

## Generate a harness

```bash
# Local Ollama on the host (the default): the image points at host.docker.internal:11434
docker run --rm -v "$PWD:/work" api-shakedown run /opt/shakedown/examples/petstore.yaml --out /work/harness

# Your own spec in the current folder
docker run --rm -v "$PWD:/work" api-shakedown run /work/openapi.yaml --out /work/harness

# No LLM at all (seconds, deterministic)
docker run --rm -v "$PWD:/work" api-shakedown run /work/openapi.yaml --out /work/harness --offline
```

Paths in the output (like `/work/harness`) are inside the container; on the host the
harness is in `./harness`. Then `cd harness && docker compose up -d` as usual.

- **Linux**: add `--add-host=host.docker.internal:host-gateway` so the container can
  reach Ollama on the host, and `--user "$(id -u):$(id -g)"` so the files belong to you.
- **PowerShell**: use `-v "${PWD}:/work"`.
- **Ollama elsewhere**: `-e SHAKEDOWN_OLLAMA_URL=http://gpu-box:11434`.
- **Check the connection**: `docker run --rm api-shakedown doctor`.

## Cloud models (bring your own key)

```bash
docker run --rm -v "$PWD:/work" \
  -e SHAKEDOWN_PROVIDER=anthropic -e ANTHROPIC_API_KEY \
  api-shakedown run /work/openapi.yaml --out /work/harness
```

`-e ANTHROPIC_API_KEY` without a value passes your shell's variable through, so the key
never appears in the command line or the image. Never bake keys into an image.

## MCP server

```bash
docker run -i --rm -v "$PWD:/work" --entrypoint shakedown-mcp api-shakedown
```

See the MCP section of the README for registering it with a client.
