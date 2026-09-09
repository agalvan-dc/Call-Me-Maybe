IMAGE_NAME = call-me-maybe-image
CONTAINER_NAME = call-me-maybe-dev

OS := $(shell uname -s)

build:
	@docker build -t $(IMAGE_NAME)
	@echo -e "\e[1;32mDocker image mounted\e[0m"

run:
	@echo -e "\e[1;91mDetected OS: $(OS)\e[0m"
	@docker run --rm -it \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$$(pwd):/app:z" \
--name $(CONTAINER_NAME) $(IMAGE_NAME) uv run call-me-maybe.py
shell:
	docker run --rm -it -v "$$(pwd):/app:z" --name $(CONTAINER_NAME) $(IMAGE_NAME) /bin/bash

debug:
	docker run --rm -it \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$$(pwd):/app:z" \
		$(IMAGE_NAME) uv run python -m pdb call-me-maybe.py 

lint:
	docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$$(pwd):/app:z" $(IMAGE_NAME) bash -c "uv run flake8 . && uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs"

lint-strict:
	docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$$(pwd):/app:z" $(IMAGE_NAME) bash -c "uv run flake8 . && uv run mypy . --strict"

clean:
	@echo "Cleaning cache files (resolving Docker root permissions)..."
	@echo "Cleaning local files and JSON configurations..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.json" -delete 2>/dev/null || true
	@rm -rf .mypy_cache poetry.lock 2>/dev/null || true
	@echo "Cleaning Docker environment..."
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	@docker image prune -f
	@echo -e "\e[1;32mDocker and residues cleaned\e[0m"	

.PHONY: build run shell debug lint lint-strict clean
