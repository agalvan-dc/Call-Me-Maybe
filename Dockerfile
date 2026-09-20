# Build stage: install the uv package manager (fast, deterministic)
FROM ghcr.io/astral-sh/uv:0.8.2 AS uv

# Final image: minimal Python 3.12 runtime
FROM python:3.12-slim

# Install curl for Hugging Face model download
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the build stage
COPY --from=uv /uv /uvx /bin/

# Create a non-root user matching the local user id
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME

# Prepare the workspace and the model cache owned by the non-root user
RUN mkdir -p /app /home/$USERNAME/.cache/huggingface \
    && chown -R $USERNAME:$USERNAME /app /home/$USERNAME/.cache

# Bake the project environment at build time so no downloads (and no `uv sync`)
# are needed at runtime: runtime + dev dependencies, plus the llm-sdk workspace
# member, all installed into /home/appuser/.venv.
ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/home/$USERNAME/.venv \
    PATH="/home/$USERNAME/.venv/bin:$PATH"

WORKDIR /app
COPY . /app

RUN uv sync --frozen --all-groups \
    && chown -R $USERNAME:$USERNAME /app /home/$USERNAME/.venv

# Switch to the non-root user and run the engine
USER $USERNAME

CMD ["python", "-m", "src"]