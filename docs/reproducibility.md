# Reproducibility

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
```

## Cache Behavior

Hugging Face datasets and models are cached under `data/hf_cache/` when scripts download them. Summary
records are cached under `data/processed/summaries/`. These directories are intentionally ignored by
Git because they can become large.

## Dataset Download Notes

The first run of `ag_news` or `arxiv` commands may download data from Hugging Face. Later runs reuse the
local cache when available. CI does not download datasets or models.

## Quick Benchmark

```bash
python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset arxiv --quick
```

## Regenerate Comparison Reports And Plots

```bash
python -m longdoc_transformer_classifier.training.compare_reports
python -m longdoc_transformer_classifier.training.plot_comparison
python -m longdoc_transformer_classifier.training.write_final_reports
```

## TF-IDF Sweep

```bash
python -m longdoc_transformer_classifier.training.sweep_tfidf_baseline --dataset arxiv --max-train-samples 3000 --max-test-samples 1000
```

## Large Artifacts

Large model weights, dataset caches, raw data, processed summaries, and binary checkpoints are not
committed. Recreate them with the commands above when needed.

