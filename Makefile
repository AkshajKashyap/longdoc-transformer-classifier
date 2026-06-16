.PHONY: install test lint check baseline analyze-ag-news analyze-longdoc baseline-ag-news baseline-longdoc truncated-ag-news truncated-arxiv chunked-ag-news chunked-arxiv chunked-arxiv-uniform chunked-arxiv-idf long-context-arxiv long-context-ag-news summary-arxiv summary-ag-news compare-reports benchmark-quick benchmark-transformers benchmark-full-smoke plots sweep-arxiv final-reports health-check release-check

LONGDOC_DATASET ?= arxiv

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check: lint test

baseline: baseline-ag-news

analyze-ag-news:
	python -m longdoc_transformer_classifier.training.analyze_dataset --dataset ag_news --max-train-samples 1000 --max-test-samples 500

analyze-longdoc:
	python -m longdoc_transformer_classifier.training.analyze_dataset --dataset $(LONGDOC_DATASET) --max-train-samples 1000 --max-test-samples 500

baseline-ag-news:
	python -m longdoc_transformer_classifier.training.train_baseline --dataset ag_news --max-train-samples 5000 --max-test-samples 1000

baseline-longdoc:
	python -m longdoc_transformer_classifier.training.train_baseline --dataset $(LONGDOC_DATASET) --max-train-samples 1000 --max-test-samples 500

truncated-ag-news:
	python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset ag_news --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 128

truncated-arxiv:
	python -m longdoc_transformer_classifier.training.train_truncated_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 4 --max-length 512

chunked-ag-news:
	python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset ag_news --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 128 --chunk-size 100 --chunk-overlap 20 --max-chunks-per-doc 4 --aggregation mean_proba

chunked-arxiv:
	python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba

chunked-arxiv-uniform:
	python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba --chunk-selection uniform_k

chunked-arxiv-idf:
	python -m longdoc_transformer_classifier.training.train_chunked_transformer --dataset arxiv --model-name prajjwal1/bert-tiny --max-train-samples 100 --max-test-samples 50 --epochs 1 --batch-size 8 --max-length 256 --chunk-size 220 --chunk-overlap 40 --max-chunks-per-doc 8 --aggregation mean_proba --chunk-selection idf_top_k

long-context-arxiv:
	python -m longdoc_transformer_classifier.training.train_long_context_transformer --dataset arxiv --model-name allenai/longformer-base-4096 --max-train-samples 20 --max-test-samples 10 --epochs 1 --batch-size 1 --max-length 1024 --freeze-encoder

long-context-ag-news:
	python -m longdoc_transformer_classifier.training.train_long_context_transformer --dataset ag_news --model-name allenai/longformer-base-4096 --max-train-samples 20 --max-test-samples 10 --epochs 1 --batch-size 1 --max-length 512 --freeze-encoder

summary-arxiv:
	python -m longdoc_transformer_classifier.training.train_summary_classifier --dataset arxiv --max-train-samples 30 --max-test-samples 15 --summarizer-model sshleifer/distilbart-cnn-12-6 --summary-max-input-tokens 1024 --summary-max-new-tokens 120 --summary-min-new-tokens 30 --summary-num-beams 2 --classifier tfidf

summary-ag-news:
	python -m longdoc_transformer_classifier.training.train_summary_classifier --dataset ag_news --max-train-samples 30 --max-test-samples 15 --summarizer-model sshleifer/distilbart-cnn-12-6 --summary-max-input-tokens 512 --summary-max-new-tokens 80 --summary-min-new-tokens 20 --summary-num-beams 2 --classifier tfidf

compare-reports:
	python -m longdoc_transformer_classifier.training.compare_reports

benchmark-quick:
	python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset $(LONGDOC_DATASET) --quick

benchmark-transformers:
	python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset $(LONGDOC_DATASET) --quick --include-transformers

benchmark-full-smoke:
	python -m longdoc_transformer_classifier.training.run_benchmark_suite --dataset $(LONGDOC_DATASET) --quick --include-transformers --include-summary

plots:
	python -m longdoc_transformer_classifier.training.plot_comparison

sweep-arxiv:
	python -m longdoc_transformer_classifier.training.sweep_tfidf_baseline --dataset arxiv --max-train-samples 3000 --max-test-samples 1000

final-reports: compare-reports plots
	python -m longdoc_transformer_classifier.training.write_final_reports

health-check:
	python -m longdoc_transformer_classifier.training.write_final_reports

release-check:
	ruff check .
	pytest -q
	python -m longdoc_transformer_classifier.training.compare_reports
	python -m longdoc_transformer_classifier.training.plot_comparison
