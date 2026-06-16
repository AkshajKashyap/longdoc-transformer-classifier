# Model Card

## Project Overview

This project is a reproducible long-document classification benchmark comparing lexical, truncation, chunking, summarization, and long-context transformer strategies.

## Intended Use

Use this repo to study baseline behavior and engineering tradeoffs for long-document classification experiments. It is not a production classifier.

## Datasets

- `ag_news`: short-text smoke dataset.
- `arxiv`: long-document target dataset from `ccdv/arxiv-classification`.

## Methods Compared

- Chunked Transformer
- Long-context Transformer
- Summary-First Classifier
- TF-IDF + Logistic Regression
- TF-IDF Sweep
- Truncated Transformer

## Headline Results

- `ag_news`: `baseline_ag_news` with macro-F1 0.7376 and accuracy 0.7400.
- `arxiv`: `baseline_arxiv` with macro-F1 0.8320 and accuracy 0.8340.

## Limitations

- Smoke sample sizes are not final rankings.
- Tiny and frozen transformer runs validate infrastructure more than performance.
- TF-IDF can exploit lexical label cues without modeling discourse.
- Long-context models remain bounded by max length and compute.

## Ethical And Interpretability Notes

The benchmark favors transparent reporting: confusion matrices, per-class F1, and method limitations are saved with reports. Results should not be used for high-stakes document triage without domain validation and bias checks.

## Reproducibility Commands

```bash
make install
make check
python -m longdoc_transformer_classifier.training.compare_reports
python -m longdoc_transformer_classifier.training.plot_comparison
```

## What This Project Does Not Claim

It does not claim state-of-the-art accuracy. The strongest honest claim is that it builds a clean benchmark showing why simple baselines can outperform poorly adapted neural methods under constrained training.
