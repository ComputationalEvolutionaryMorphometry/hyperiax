.PHONY: test sync

# Install all dev + extras into the project venv (uv-managed).
sync:
	uv sync --all-extras --group dev

test:
	uv run pytest
