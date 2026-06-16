# Method Notes

## AG News As A Smoke Dataset

AG News stays in the project because it is small, fast to download, and useful for validating the
classification pipeline. It is not a long-document benchmark. Its role is to catch data loading,
feature extraction, evaluation, and report-generation regressions cheaply.

## arXiv As The Long-Document Target

The `arxiv` dataset uses Hugging Face `ccdv/arxiv-classification`. It is closer to the real project
goal because examples are scientific documents with subject-area labels, and many documents are far
longer than a standard 512-token transformer window.

## TF-IDF Baseline

The TF-IDF + Logistic Regression baseline proves that the repo can load a dataset, build reproducible
features, train a classical classifier, compute document-level metrics, and save reports. It is a
strong lexical baseline, especially when labels are tied to domain-specific vocabulary.

## Truncation Baseline

The truncated transformer baseline tokenizes each document once and keeps only the first fixed window.
It proves what a naive transformer setup can and cannot see. For long documents, it is mostly a
measurement of truncation risk rather than a real long-document solution.

## Chunked Aggregation Baseline

The chunked transformer baseline splits documents into overlapping word chunks, classifies chunks, and
aggregates chunk predictions back to document-level predictions. It proves the project can preserve
document identity through chunking and evaluate at the correct level, while also exposing the weakness
of inherited chunk labels.

Chunk selection matters when a document has more chunks than the configured `max_chunks_per_doc` cap.
The baseline now supports `first_k`, `uniform_k`, `idf_top_k`, and `longest_k`. `first_k` preserves the
original prefix-only behavior. `uniform_k` spreads selected chunks across the document. `idf_top_k`
fits a simple IDF scorer on training chunks only and applies that scorer to train and test chunks. These
strategies are selection heuristics, not learned evidence retrieval.

## Summary-First Baseline

The summary-first classifier compresses documents before classification. It proves the project can
cache generated summaries and compare a compression strategy against truncation and chunking. It can
fail when the summarizer sees only the front portion of a document or removes label evidence.

## What Each Method Proves

- TF-IDF proves the classical lexical floor and reporting pipeline.
- Truncation proves how much evidence a fixed-window transformer discards.
- Chunking proves the project can structurally process more of each document.
- Summary-first proves compression can be benchmarked without changing the evaluation contract.
