FROM ghcr.io/astral-sh/uv:python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONBUFFERED=1 \
    UV_VIRTUALENVS_CREATE=/usr/local
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR	 /app

COPY pyproject.toml poetry.lock* uv.lock* ./
COPY llm_sdk/pyproject.toml ./llm_sdk/

RUN uv sync --no-install-project --no-dev --frozen

COPY . /app

CMD ["uv", "run", "python", "call-me-maybe.py"]
