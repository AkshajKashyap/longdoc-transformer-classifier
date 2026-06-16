# LongDoc Transformer Classifier

This project is a long-document NLP classification workbench. The longer-term goal is to compare:

- TF-IDF + Logistic Regression baseline
- truncated transformer baseline
- chunked transformer baseline
- summarization-first classifier
- optional Longformer/BigBird baseline

## Current Milestone

Milestone 1 is intentionally modest: a reliable classical baseline pipeline on Hugging Face `ag_news`.

It includes dataset loading, deterministic sample caps, TF-IDF features, Logistic Regression, evaluation metrics, tests, and report generation. It does not include transformer training, summarization, APIs, dashboards, or monitoring yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Run Checks

```bash
make check
```

This runs:

```bash
ruff check .
pytest -q
```

## Train The Baseline

```bash
python -m longdoc_transformer_classifier.training.train_baseline --max-train-samples 5000 --max-test-samples 1000
```

or:

```bash
make baseline
```

The baseline writes:

- `reports/baseline_ag_news.md`
- `reports/baseline_ag_news_metrics.json`

## What This Baseline Proves

The first milestone proves that the project can load a text classification dataset, create reproducible features, train a simple classifier, compute comparable metrics, and save reports. That gives future transformer experiments a real benchmark instead of a vibes-only comparison.

## What Comes Next

Next milestones can add a truncated transformer baseline, chunked long-document handling, summarization-first classification, and eventually long-context transformer baselines. Each should reuse the same data, metrics, and reporting foundations where possible.
