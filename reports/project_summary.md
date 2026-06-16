# Project Summary

## One-Paragraph Summary

I built a reproducible long-document classification benchmark comparing lexical, truncation, chunking, summarization, and long-context transformer strategies.

## What Was Built

- Dataset loaders for short and long text classification.
- Length analysis, chunking, chunk selection, and summary caching.
- Classical, truncated, chunked, summary-first, and Longformer-style baselines.
- Unified metrics, comparison reports, plots, and portfolio documentation.

## Why Long-Document Classification Is Hard

Long documents often exceed standard transformer context windows, so naive models either truncate evidence, require chunk aggregation, summarize away details, or pay higher compute costs for long-context architectures.

## What Each Baseline Proved

- TF-IDF proved a strong lexical floor.
- Truncation showed the weakness of prefix-only transformers.
- Chunking made the pipeline structurally long-document aware.
- Summary-first tested compression before classification.
- Long-context transformers added the missing architecture-level comparison.

## Current Best Result

- `ag_news`: `baseline_ag_news` with macro-F1 0.7376 and accuracy 0.7400.
- `arxiv`: `baseline_arxiv` with macro-F1 0.8320 and accuracy 0.8340.

## Key Engineering Choices

The repo keeps experiments deterministic where possible, separates data/features/models, writes serializable reports, avoids app/server overbuild, and keeps expensive model runs opt-in.

## What I Would Improve Next

With more compute I would run larger TF-IDF sweeps, fully fine-tune long-context models, try learned chunk retrieval, improve summary selection, and add confidence calibration.
