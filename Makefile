IMAGE_NAME = call-me-maybe-image
CONTAINER_NAME = call-me-maybe-dev

SHELL := /bin/bash
OS := $(shell uname -s)

HF_CACHE := $(HOME)/.cache/huggingface
UV_CACHE := $(HOME)/.cache/uv

install:
	@mkdir -p $(HF_CACHE)
	@docker build -t $(IMAGE_NAME) .
	@echo -e "\e[1;32mDocker image built with dependencies baked in\e[0m"

run:
	@mkdir -p $(HF_CACHE)
	@echo -e "\e[1;91mDetected OS: $(OS)\e[0m"
	@docker run --rm -it \
		--user 0:0 \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$$(pwd):/app:z" \
		-v "$(HF_CACHE):/home/appuser/.cache/huggingface:z" \
		--name $(CONTAINER_NAME) $(IMAGE_NAME) \
		bash -c "chown -R 0:0 /app /home/appuser/.cache/huggingface && uv run --no-sync python -m src"

shell:
	@mkdir -p $(HF_CACHE)
	docker run --rm -it \
		--user 0:0 \
		-v "$$(pwd):/app:z" \
		-v "$(HF_CACHE):/root/.cache/huggingface:z" \
		--name $(CONTAINER_NAME) $(IMAGE_NAME) /bin/bash

debug:
	@mkdir -p $(HF_CACHE)
	docker run --rm -it \
		--user 0:0 \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$$(pwd):/app:z" \
		-v "$(HF_CACHE):/home/appuser/.cache/huggingface:z" \
		$(IMAGE_NAME) bash -c "chown -R 0:0 /app /home/appuser/.cache/huggingface && uv run --no-sync python -m pdb -m src"

lint:
	@mkdir -p $(HF_CACHE)
	docker run --rm --user 0:0 -e PYTHONDONTWRITEBYTECODE=1 -v "$$(pwd):/app:z" $(IMAGE_NAME) bash -c "flake8 . --extend-exclude=llm_sdk && mypy . --exclude=llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs"

lint-strict:
	@mkdir -p $(HF_CACHE)
	docker run --rm --user 0:0 -e PYTHONDONTWRITEBYTECODE=1 -v "$$(pwd):/app:z" $(IMAGE_NAME) bash -c "flake8 . --extend-exclude=llm_sdk && mypy . --exclude=llm_sdk --strict"

clean:
	@echo "Cleaning cache and output files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -f data/output/*.json output/*.json 2>/dev/null || true
	@rm -rf .mypy_cache .venv 2>/dev/null || true
	@echo "Cleaning Docker environment..."
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	@docker image prune -f
	@docker builder prune -f
	@echo -e "\e[1;32mDocker and residues cleaned\e[0m"

fclean: clean
	@echo "Removing downloaded model weights and uv package cache..."
	@rm -rf $(HF_CACHE) $(UV_CACHE)
	@echo -e "\e[1;32mAll caches (HuggingFace & uv) completely removed\e[0m"

.PHONY: install run shell debug lint lint-strict clean fclean