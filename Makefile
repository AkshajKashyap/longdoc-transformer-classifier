.PHONY: install test lint check baseline analyze-ag-news analyze-longdoc baseline-ag-news baseline-longdoc

LONGDOC_DATASET ?= arxiv

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check: lint test

baseline: baseline-ag-news

analyze-ag-news:
	python -m longdoc_transformer_classifier.training.analyze_dataset --dataset ag_news --max-train-samples 1000 --max-test-samples 500

analyze-longdoc:
	python -m longdoc_transformer_classifier.training.analyze_dataset --dataset $(LONGDOC_DATASET) --max-train-samples 1000 --max-test-samples 500

baseline-ag-news:
	python -m longdoc_transformer_classifier.training.train_baseline --dataset ag_news --max-train-samples 5000 --max-test-samples 1000

baseline-longdoc:
	python -m longdoc_transformer_classifier.training.train_baseline --dataset $(LONGDOC_DATASET) --max-train-samples 1000 --max-test-samples 500
