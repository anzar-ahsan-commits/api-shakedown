# API Shakedown: `shakedown` (default) and the MCP server (`shakedown-mcp`).
#
#   docker build -t api-shakedown .
#   docker run --rm -v "$PWD:/work" api-shakedown run /opt/shakedown/examples/petstore.yaml \
#       --out /work/harness
#
# Ollama on the host is reached at host.docker.internal (on Linux, add
# --add-host=host.docker.internal:host-gateway). See DOCKER.md.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY packages/agentpanel packages/agentpanel
COPY pyproject.toml README.md LICENSE ./
COPY src src
COPY examples examples
RUN python -m build --wheel packages/agentpanel --outdir /dist \
 && python -m build --wheel . --outdir /dist

FROM python:3.12-slim
# Lets the MCP Registry verify this image belongs to the registered server.
LABEL io.modelcontextprotocol.server.name="io.github.anzar-ahsan-commits/api-shakedown"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHAKEDOWN_OLLAMA_URL=http://host.docker.internal:11434
COPY --from=build /dist /tmp/dist
RUN core=$(ls /tmp/dist/agentpanel-*.whl) \
 && app=$(ls /tmp/dist/api_shakedown-*.whl) \
 && pip install --no-cache-dir "$core" "${app}[cloud,mcp]" \
 && rm -rf /tmp/dist \
 && useradd --create-home --uid 1000 shakedown
COPY examples /opt/shakedown/examples
USER shakedown
WORKDIR /work
ENTRYPOINT ["shakedown"]
CMD ["--help"]
