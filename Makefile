.PHONY: install test lint check baseline

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check: lint test

baseline:
	python -m longdoc_transformer_classifier.training.train_baseline --max-train-samples 5000 --max-test-samples 1000
