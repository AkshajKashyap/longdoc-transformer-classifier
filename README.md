# LongDoc Transformer Classifier

[![CI](https://github.com/AkshajKashyap/longdoc-transformer-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/AkshajKashyap/longdoc-transformer-classifier/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

Reproducible benchmark for long-document classification across lexical, truncation, chunking,
summarization, and long-context transformer strategies.

## Problem

Normal text classifiers often assume the whole input fits in a short context window. Long documents,
like arXiv papers, can be tens of thousands of words, so standard transformer classifiers either
truncate evidence, require chunk aggregation, compress the document first, or use more expensive
long-context architectures.

This repo is an honest benchmark project, not a production app and not a state-of-the-art claim.

## Methods Implemented

| Method | What It Tests |
| --- | --- |
| TF-IDF + Logistic Regression | Strong lexical baseline over full document text |
| TF-IDF sweep | Tuned classical reference point |
| Truncated transformer | Prefix-only fixed-window baseline |
| Chunked transformer | Manual long-document structure with document-level aggregation |
| Summary-first classifier | Compression before classification |
| Long-context transformer | Longformer-style architecture without manual chunk aggregation |

## Headline Results

Current reports show TF-IDF as the strongest baseline under constrained training.

| Dataset | Current Best | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| `ag_news` | TF-IDF baseline | 0.7400 | 0.7376 |
| `arxiv` | Stronger TF-IDF baseline | 0.8340 | 0.8320 |
| `arxiv` | TF-IDF sweep best | 0.8180 | 0.8155 |

Transformer scores in this repo are smoke scores. Tiny/frozen runs validate infrastructure and expose
tradeoffs; they are not final model rankings.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
```

## Benchmark Commands

```bash
make benchmark-quick
make sweep-arxiv
python -m longdoc_transformer_classifier.training.compare_reports
python -m longdoc_transformer_classifier.training.plot_comparison
make final-reports
```

Optional smoke runs:

```bash
make benchmark-transformers
make long-context-arxiv
make summary-arxiv
```

Large model and dataset downloads are opt-in and cached locally under ignored directories.

## What To Look At First

- `reports/project_summary.md`
- `reports/model_comparison.md`
- `docs/interview_notes.md`
- `docs/reproducibility.md`

## Reports And Docs

- `reports/model_comparison.md`
- `reports/model_card.md`
- `reports/project_summary.md`
- `reports/repo_health_check.md`
- `reports/figures/`
- `docs/method_notes.md`
- `docs/interview_notes.md`
- `docs/limitations.md`
- `docs/reproducibility.md`

## Limitations

- Smoke sample sizes are not final rankings.
- arXiv labels may be predictable from lexical cues.
- Chunk labels are weak inherited labels.
- Summary-first classification can remove class evidence.
- Long-context transformers are expensive and still bounded by `max_length`.

## Repo Structure

```text
src/longdoc_transformer_classifier/
  data.py                  dataset loading
  features.py              TF-IDF features
  chunking.py              word chunking
  chunk_selection.py       chunk selection heuristics
  models/                  baseline model wrappers
  evaluation/              metrics
  training/                runnable experiment scripts
  visualization.py         comparison plots
docs/                      methodology and reproducibility notes
reports/                   generated benchmark reports
tests/                     lightweight unit tests
```

## Honest Claim

I built a reproducible long-document classification benchmark comparing lexical, truncation, chunking,
summarization, and long-context transformer strategies, and showed why simple baselines can outperform
poorly adapted neural methods under constrained training.
