FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/home/appuser/.venv

RUN useradd -m appuser
WORKDIR /app

COPY --chown=appuser:appuser . /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "call-me-maybe.py"]
