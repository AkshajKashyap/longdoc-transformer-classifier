# Interview Notes

## Project Summary

This project is a long-document classification benchmark. It starts with a classical baseline and then
adds controlled smoke baselines for truncation, chunking, and summarization-first classification.

## Core Long-Document Problem

Many documents are longer than the input window of common transformer classifiers. If the model only
reads the beginning of a document, it may miss the evidence needed for the label.

## Why BERT Token Limits Matter

Standard BERT-style classifiers usually operate around a 512-token limit. Long scientific documents can
be many times longer, so naive truncation turns classification into prefix classification.

## Why TF-IDF Can Beat Tiny Transformers

TF-IDF can perform well on small smoke runs because it sees the full document vocabulary and many labels
have strong lexical cues. Tiny transformers are used here to validate training and reporting plumbing,
not to establish final performance.

## Why Chunk Labels Are Weak Labels

Chunked training gives every chunk the parent document label. That is noisy because a chunk may be
background, methodology, or boilerplate rather than the part that supports the class.

## Why Summary-First Classification Can Fail

Summarization compresses the document before classification. The summarizer can miss later sections,
omit rare class evidence, or produce summaries optimized for readability rather than classification.

## Improvements With More Compute

With more compute, the next steps would be larger transformer smoke runs, better chunk selection,
hierarchical aggregation, stronger summarizers, calibration checks, and eventually true long-context
models such as Longformer or BigBird.

