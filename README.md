# LongDoc Transformer Classifier

This project is a long-document NLP classification workbench. The longer-term goal is to compare:

- TF-IDF + Logistic Regression baseline
- truncated transformer baseline
- chunked transformer baseline
- summarization-first classifier
- optional Longformer/BigBird baseline

## Current Milestone

Milestone 4 keeps the modeling intentionally modest while making the project structurally ready for long-document experiments.

The project now supports:

- `ag_news`, used as a fast smoke baseline
- `arxiv`, backed by Hugging Face `ccdv/arxiv-classification`, used as the first real long-document classification dataset
- document length analysis with rough 512-token BERT truncation checks
- fixed-size overlapping word chunking utilities
- TF-IDF + Logistic Regression baselines for both datasets
- naive truncated transformer baselines
- chunked transformer baselines with document-level aggregation

It still does not include summarization, long-context transformers, APIs, dashboards, or monitoring.

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
python -m longdoc_transformer_classifier.training.train_baseline --dataset ag_news --max-train-samples 5000 --max-test-samples 1000
```

or:

```bash
make baseline
```

Run the long-document baseline with:

```bash
python -m longdoc_transformer_classifier.training.train_baseline --dataset arxiv --max-train-samples 1000 --max-test-samples 500
```

Baseline reports use dataset-specific filenames:

- `reports/baseline_{dataset_name}.md`
- `reports/baseline_{dataset_name}_metrics.json`

## Analyze Dataset Lengths

AG News is short and useful for fast pipeline checks:

```bash
python -m longdoc_transformer_classifier.training.analyze_dataset --dataset ag_news --max-train-samples 1000 --max-test-samples 500
```

The arXiv dataset is closer to the real project goal because each example is a paper-length scientific document with a subject-area label:

```bash
python -m longdoc_transformer_classifier.training.analyze_dataset --dataset arxiv --max-train-samples 1000 --max-test-samples 500
```

Length reports are saved as:

- `reports/{dataset_name}_length_analysis.md`
- `reports/{dataset_name}_length_analysis.json`

## Why AG News Is Only A Smoke Baseline

`ag_news` downloads quickly, has clean labels, and keeps tests and local smoke runs cheap. It is not a long-document dataset, so it does not answer whether truncation will discard important evidence.

## Why arXiv Is Closer To The Goal

`ccdv/arxiv-classification` provides long scientific documents with classification labels. It lets the project measure document lengths, estimate how often a 512-token transformer input would truncate the document, and prove the same baseline pipeline can run on realistic long texts.

## Why Chunking Matters

Most standard transformer classifiers can only read a fixed-size window. For long documents, a later model will need to split each document into chunks, classify or encode those chunks, and aggregate evidence back to the original document. This milestone adds deterministic word chunking with configurable chunk size and overlap so that later transformer experiments can share one tested chunking foundation.

## Truncated Transformer Baseline

Milestone 3 adds a deliberately naive transformer baseline: tokenize each raw document once, truncate to `max_length`, and train a standard sequence classifier on only that prefix. This is not a long-document solution. It matters because it gives the later chunking and summarization milestones a fair comparison against the simplest transformer approach.

Use `prajjwal1/bert-tiny` for quick CPU smoke runs:

```bash
python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset ag_news --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 128
```

```bash
python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 4 --max-length 512
```

`distilbert-base-uncased` is the default and a more realistic baseline model once you are ready for a slower run:

```bash
python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset arxiv --model-name distilbert-base-uncased --max-train-samples 1000 --max-test-samples 500 --epochs 1 --batch-size 8 --max-length 512
```

The truncated transformer reports are saved as:

- `reports/truncated_transformer_{dataset_name}.md`
- `reports/truncated_transformer_{dataset_name}_metrics.json`

## Chunked Transformer Baseline

Truncation is naive because a standard transformer sees only the first `max_length` tokens. For arXiv papers, that often means the model never sees most of the document.

The chunked baseline is structurally better for long documents: it splits each document into overlapping word chunks, trains a transformer classifier on those chunks, then aggregates chunk predictions back to one document-level prediction. This differs from summarization because it does not rewrite or compress the document, and it differs from Longformer/BigBird because it still uses a regular fixed-window encoder.

Chunk labels are imperfect: every chunk inherits the parent document label, even if that chunk does not contain the evidence for the label. The report calls this out explicitly and computes metrics only at the document level.

Supported aggregation strategies:

- `mean_proba`: average class probabilities across chunks
- `max_proba`: use the most confident chunk's probability distribution
- `majority_vote`: vote over chunk-level predicted classes

Smoke run for AG News:

```bash
python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset ag_news --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 128 --chunk-size 100 --chunk-overlap 20 --max-chunks-per-doc 4 --aggregation mean_proba
```

Smoke run for arXiv:

```bash
python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba
```

The chunked transformer reports are saved as:

- `reports/chunked_transformer_{dataset_name}.md`
- `reports/chunked_transformer_{dataset_name}_metrics.json`

## What This Baseline Proves

The first four milestones prove that the project can load short and long text classification datasets, measure document lengths, create reproducible features, train classical, truncated-transformer, and chunked-transformer baselines, compute comparable document-level metrics, chunk long documents, and save reports. That gives future transformer experiments a real benchmark instead of a vibes-only comparison.

## What Comes Next

Next milestones can add summarization-first classification and eventually long-context transformer baselines. Each should reuse the same data, chunking, metrics, and reporting foundations where possible.
