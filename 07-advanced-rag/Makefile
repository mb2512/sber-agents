.PHONY: install run dataset dataset-upload

install:
	@echo "Installing dependencies with Python 3.13 (Python 3.14 not yet supported)..."
	@echo "Using Python 3.13 to avoid compatibility issues with pydantic-core..."
	UV_HTTP_TIMEOUT=300 uv sync --python 3.13

run:
	uv run python src/bot.py

dataset:
	uv run python src/dataset_synthesizer.py --create

dataset-upload:
	uv run python src/dataset_synthesizer.py --upload

