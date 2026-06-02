.PHONY: test sync

# Install the core (CPU) runtime + dev group into the project venv (uv-managed).
# For a CUDA venv instead:  uv sync --group dev --extra gpu
sync:
	uv sync --group dev

test:
	uv run pytest
