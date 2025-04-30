FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

LABEL org.opencontainers.image.source=https://github.com/L3GJ0N/optimization-dashboard
LABEL org.opencontainers.image.description="Gradient Descent Analysis"
LABEL org.opencontainers.image.licenses=MIT

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra hosting

ADD src /app/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --extra hosting

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

EXPOSE 8000

CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout", "600", "gradient_descent.main:server", "--limit-request-line", "0", "--limit-request-field_size", "0"]
