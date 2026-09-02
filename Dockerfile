FROM python:3.12-slim

# uv is copied from its official distroless image rather than installed via a
# download script, so the build has no network dependency beyond the registry.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies are installed in their own layer, before the source is copied,
# so editing src/ or tests/ does not invalidate the dependency cache.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src/ ./src/
COPY tests/ ./tests/

RUN uv sync --locked

CMD ["uv", "run", "pytest", "-v"]
