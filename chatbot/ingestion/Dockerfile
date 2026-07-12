# syntax=docker/dockerfile:1

FROM python:3.11.9-slim AS deps
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.11.9-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /app/.cache /app/data && chown -R app:app /app
COPY --from=deps /app/.venv /app/.venv
COPY --chown=app:app . .
USER app
CMD ["python", "ingest.py"]
