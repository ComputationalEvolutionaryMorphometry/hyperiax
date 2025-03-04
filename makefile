.PHONY: test local-notebook-test

test:
	python -m pytest -s

local-notebook-test:
	@echo "Running all notebooks in examples directory..."
	@find examples -name "*.ipynb" -type f -print -exec sh -c "echo 'Testing {}' && jupyter nbconvert --execute --inplace {} || echo 'Error in {}'" \;
	@echo "Notebook testing complete."