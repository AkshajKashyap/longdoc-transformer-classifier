# LongDoc Transformer Classifier

This repo is a long-document NLP classification benchmark. The purpose is to compare simple,
reproducible baselines before investing in heavier long-context modeling.

The longer-term comparison plan is:

- TF-IDF + Logistic Regression baseline
- truncated transformer baseline
- chunked transformer baseline
- summarization-first classifier
- optional BigBird baseline later

## Current Status

Milestone 8 adds a conservative long-context transformer baseline on top of the existing benchmark
project and chunk-selection infrastructure.

The project includes:

- `ag_news` as a fast smoke dataset
- `arxiv`, backed by Hugging Face `ccdv/arxiv-classification`, as the long-document target
- document length analysis with rough 512-token truncation checks
- deterministic word chunking utilities
- chunk selection strategies for capped chunked transformer runs
- a conservative Longformer-style long-context transformer smoke baseline
- TF-IDF, truncated transformer, chunked transformer, and summary-first smoke baselines
- unified comparison reports and matplotlib figures
- method notes, interview notes, and limitations docs

The repo still does not include FastAPI, Streamlit, monitoring, BigBird, or fine-tuned BART.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
```

Run the practical quick benchmark on the long-document dataset:

```bash
python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset arxiv --quick
```

The quick benchmark runs length analysis, the TF-IDF baseline, and report comparison. It skips
transformer and summary smoke runs unless requested.

## Benchmark Commands

Quick classical benchmark:

```bash
make benchmark-quick
```

Include tiny transformer smoke runs:

```bash
make benchmark-transformers
```

Include a conservative long-context transformer smoke run:

```bash
python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset arxiv --quick --include-long-context
```

Include tiny transformer and summary-first smoke runs:

```bash
make benchmark-full-smoke
```

Refresh comparison reports:

```bash
python -m longdoc_transformer_classifier.training.compare_reports
```

Generate figures:

```bash
python -m longdoc_transformer_classifier.training.plot_comparison
```

Figures are written to:

- `reports/figures/model_macro_f1_by_method.png`
- `reports/figures/model_accuracy_by_method.png`

## Dataset Analysis

AG News is short and useful for fast pipeline checks:

```bash
python -m longdoc_transformer_classifier.training.analyze_dataset --dataset ag_news --max-train-samples 1000 --max-test-samples 500
```

arXiv is closer to the real project goal because examples are long scientific documents:

```bash
python -m longdoc_transformer_classifier.training.analyze_dataset --dataset arxiv --max-train-samples 1000 --max-test-samples 500
```

Length reports are saved as:

- `reports/{dataset_name}_length_analysis.md`
- `reports/{dataset_name}_length_analysis.json`

## Baseline Training

AG News smoke baseline:

```bash
python -m longdoc_transformer_classifier.training.train_baseline --dataset ag_news --max-train-samples 5000 --max-test-samples 1000
```

arXiv long-document baseline:

```bash
python -m longdoc_transformer_classifier.training.train_baseline --dataset arxiv --max-train-samples 1000 --max-test-samples 500
```

Baseline reports are saved as:

- `reports/baseline_{dataset_name}.md`
- `reports/baseline_{dataset_name}_metrics.json`

## Transformer Smoke Runs

Truncated transformer smoke run:

```bash
python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 4 --max-length 512
```

Chunked transformer smoke run:

```bash
python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba
```

Uniform chunk selection:

```bash
python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba --chunk-selection uniform_k
```

IDF top-k chunk selection:

```bash
python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba --chunk-selection idf_top_k
```

Summary-first smoke run:

```bash
python -m longdoc_transformer_classifier.training.train_summary_classifier --dataset arxiv --max-train-samples 30 --max-test-samples 15 --summarizer-model sshleifer/distilbart-cnn-12-6 --summary-max-input-tokens 1024 --summary-max-new-tokens 120 --summary-min-new-tokens 30 --summary-num-beams 2 --classifier tfidf
```

## Long-Context Transformer Baseline

Longformer-style models are different from truncation because the architecture is designed for longer
input windows. They are also different from chunking because the model receives one longer tokenized
document directly rather than separate chunk predictions that must be aggregated.

The smoke baseline is intentionally conservative:

```bash
python -m longdoc_transformer_classifier.training.train_long_context_transformer --dataset arxiv --model-name allenai/longformer-base-4096 --max-train-samples 20 --max-test-samples 10 --epochs 1 --batch-size 1 --max-length 1024 --freeze-encoder
```

Use `batch-size 1` and modest `max_length` values unless you know your GPU memory budget. A
`max_length=1024` smoke run does not use Longformer's full 4096-token capacity, and `--freeze-encoder`
limits adaptation. These runs exist to benchmark the missing architectural family, not to claim final
performance.

## Current Best Result

In the current smoke reports, TF-IDF + Logistic Regression is the strongest method on both AG News and
arXiv by macro-F1. That is expected: TF-IDF sees the full document vocabulary, while the tiny transformer
runs are intentionally small pipeline checks.

Do not overinterpret smoke transformer scores. `prajjwal1/bert-tiny` validates training, chunking,
aggregation, and reporting mechanics; it is not a serious performance baseline.

## Why AG News Is Only A Smoke Baseline

`ag_news` downloads quickly, has clean labels, and keeps local checks cheap. It is not a long-document
dataset, so it does not answer whether truncation will discard important evidence.

## Why arXiv Is The Long-Document Target

`ccdv/arxiv-classification` provides long scientific documents with classification labels. It lets the
project measure document lengths, estimate how often a 512-token input would truncate the document, and
prove that the same pipeline can run on realistic long texts.

## Why Chunking Matters

Most standard transformer classifiers can only read a fixed-size window. For long documents, a model
needs to split each document into chunks, classify or encode those chunks, and aggregate evidence back
to the original document. This repo keeps chunking deterministic and tested so later transformer
experiments share the same foundation.

## Chunk Selection Strategies

The original chunked baseline used `first_k`, which selects only the first `max_chunks_per_doc` chunks.
That is a weak default for very long documents because it may ignore the middle and end entirely.

Supported strategies:

- `first_k`: preserves the original behavior and selects the first chunks.
- `uniform_k`: selects chunks approximately evenly across the document, improving rough coverage of
  beginning, middle, and end.
- `idf_top_k`: fits a lightweight IDF scorer on training chunks only and selects lexically informative
  chunks at train and test time.
- `longest_k`: selects the longest chunks as a simple sanity baseline.

These are unsupervised heuristics. They make the capped chunked baseline less naive, but they are not
learned semantic evidence retrieval.

Practical Make targets:

```bash
make chunked-arxiv-uniform
make chunked-arxiv-idf
```

## Documentation

- `docs/method_notes.md` explains what each method proves.
- `docs/interview_notes.md` gives concise project talking points.
- `docs/limitations.md` lists the caveats behind the current smoke results.

## What Comes Next

Future milestones can add stronger summary models, better chunk selection, hierarchical aggregation,
hierarchical aggregation, larger long-context runs, and eventually BigBird baselines. Each should reuse
the same data, chunking, metrics, comparison, and plotting foundations.
