# syntax=docker/dockerfile:1

# ── deps: resolve the locked venv in an isolated layer ────────────
FROM python:3.11.9-slim AS deps
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── runtime: minimal, non-root, venv on PATH ──────────────────────
FROM python:3.11.9-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface
# curl for the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /app/.cache && chown -R app:app /app/.cache
COPY --from=deps /app/.venv /app/.venv
COPY --chown=app:app . .
USER app
EXPOSE 8000
# Same image runs the gateway (default) or the worker (compose overrides command).
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
