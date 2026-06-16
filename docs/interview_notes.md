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

## Why Chunk Selection Matters

If a long document produces many chunks and the pipeline keeps only the first few, the model is still
mostly doing prefix classification. Uniform selection improves rough document coverage, and IDF top-k
adds a lexical informativeness heuristic, but neither strategy learns which chunks truly contain
evidence for the label.

## Why Summary-First Classification Can Fail

Summarization compresses the document before classification. The summarizer can miss later sections,
omit rare class evidence, or produce summaries optimized for readability rather than classification.

## Why Add Longformer After Truncation And Chunking

Truncation is the naive fixed-window baseline, and chunking is a manual way to expose more of the
document. Longformer-style models add the missing architectural comparison: a transformer designed to
process longer sequences directly.

The tradeoff is cost. Long-context models use more memory and compute, so smoke runs use tiny sample
sizes, batch size 1, shorter-than-maximum context windows, and sometimes a frozen encoder. TF-IDF can
still win these smoke runs because it sees the full document vocabulary cheaply and the neural setup has
little data or adaptation time.

## Improvements With More Compute

With more compute, the next steps would be larger transformer smoke runs, better chunk selection,
hierarchical aggregation, stronger summarizers, calibration checks, and eventually true long-context
models such as Longformer or BigBird.

## Talk Track

### 30-Second Explanation

I built a reproducible benchmark for long-document classification. It compares TF-IDF, truncated
transformers, chunked transformers, summary-first classification, and Longformer-style long-context
models on short and long datasets, with honest reports showing that simple lexical baselines can beat
poorly adapted neural methods under constrained training.

### 2-Minute Explanation

The core problem is that many documents are far longer than normal transformer context windows. I first
implemented a strong TF-IDF baseline, then measured document lengths, added chunking and chunk
selection, tested truncation, summary-first classification, and finally a conservative Longformer
baseline. The repo emphasizes reproducibility: every run writes metrics, reports, comparison tables, and
plots. The main result is not that TF-IDF is universally better, but that a benchmark must include
strong simple baselines before interpreting expensive neural runs.

### Likely Interviewer Questions

**Why did TF-IDF beat transformers?**  
The transformer runs are tiny, frozen or undertrained smoke checks, while TF-IDF sees the full document
vocabulary and arXiv labels have strong lexical cues.

**Why use Longformer?**  
It adds the missing architectural baseline: a transformer designed for longer windows, unlike
truncation or manual chunk aggregation.

**Why not just summarize everything?**  
Summaries can omit class evidence, and summarizers have their own input limits. It is a useful baseline,
not a universal solution.

**What is weak about chunk labels?**  
Every chunk inherits the document label, even if that chunk contains no evidence for the label.

**How would you improve this with more compute?**  
Run larger sweeps, fully fine-tune long-context models, learn chunk selection, improve summary
selection, and add calibration.

**What did you learn?**  
Long-document modeling is as much an evaluation and evidence-coverage problem as a model-choice
problem. Strong baselines keep the comparison honest.
