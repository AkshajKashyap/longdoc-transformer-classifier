import json

from longdoc_transformer_classifier.training.compare_reports import (
    build_comparison,
    load_metric_rows,
    parse_metrics_file,
    run,
)


def test_comparison_report_parses_mocked_metrics_json_files(tmp_path):
    _write_metrics(
        tmp_path / "baseline_arxiv_metrics.json",
        {"dataset_name": "arxiv", "accuracy": 0.7, "macro_f1": 0.6},
    )
    _write_metrics(
        tmp_path / "summary_classifier_arxiv_metrics.json",
        {
            "dataset_name": "arxiv",
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "summarizer_model": "fake-summarizer",
            "classifier": "tfidf",
        },
    )

    rows = load_metric_rows(tmp_path)

    assert [row["method"] for row in rows] == ["baseline_arxiv", "summary_classifier_arxiv"]
    assert rows[1]["dataset"] == "arxiv"
    assert rows[1]["macro_f1"] == 0.75
    assert "fake-summarizer" in rows[1]["key_settings"]


def test_comparison_report_identifies_best_method_by_macro_f1(tmp_path):
    _write_metrics(
        tmp_path / "baseline_ag_news_metrics.json",
        {"dataset_name": "ag_news", "accuracy": 0.5, "macro_f1": 0.4},
    )
    _write_metrics(
        tmp_path / "chunked_transformer_ag_news_metrics.json",
        {"dataset_name": "ag_news", "accuracy": 0.6, "macro_f1": 0.55},
    )

    comparison = build_comparison(load_metric_rows(tmp_path))

    assert comparison["best_by_dataset"]["ag_news"]["method"] == "chunked_transformer_ag_news"
    assert comparison["best_by_dataset"]["ag_news"]["macro_f1"] == 0.55


def test_comparison_report_writes_markdown_and_json(tmp_path):
    _write_metrics(
        tmp_path / "baseline_arxiv_metrics.json",
        {"dataset_name": "arxiv", "accuracy": 0.7, "macro_f1": 0.6},
    )

    comparison = run(tmp_path)

    assert (tmp_path / "model_comparison.md").exists()
    assert (tmp_path / "model_comparison.json").exists()
    assert comparison["rows"][0]["method"] == "baseline_arxiv"


def test_comparison_parsing_handles_missing_optional_fields(tmp_path):
    metrics_path = tmp_path / "custom_arxiv_metrics.json"
    _write_metrics(metrics_path, {"metadata": {"method": "custom_arxiv", "dataset": "arxiv"}})

    row = parse_metrics_file(metrics_path)

    assert row["method"] == "custom_arxiv"
    assert row["dataset"] == "arxiv"
    assert row["accuracy"] is None
    assert row["macro_f1"] is None
    assert row["train_samples"] is None
    assert row["test_samples"] is None
    assert row["model_name"] == "n/a"
    assert row["key_limitation"]
    assert row["structural_takeaway"]


def test_comparison_report_includes_chunk_selection_in_settings(tmp_path):
    metrics_path = tmp_path / "chunked_transformer_arxiv_metrics.json"
    _write_metrics(
        metrics_path,
        {
            "dataset_name": "arxiv",
            "accuracy": 0.4,
            "macro_f1": 0.3,
            "model_name": "fake-model",
            "chunk_selection": "uniform_k",
            "aggregation": "mean_proba",
            "max_chunks_per_doc": 8,
        },
    )

    row = parse_metrics_file(metrics_path)

    assert "chunk_selection=uniform_k" in row["key_settings"]
    assert "aggregation=mean_proba" in row["key_settings"]
    assert "max_chunks_per_doc=8" in row["key_settings"]


def test_comparison_report_parses_long_context_transformer_metrics(tmp_path):
    metrics_path = tmp_path / "long_context_transformer_arxiv_metrics.json"
    _write_metrics(
        metrics_path,
        {
            "dataset_name": "arxiv",
            "accuracy": 0.4,
            "macro_f1": 0.3,
            "model_name": "allenai/longformer-base-4096",
            "max_length": 1024,
            "freeze_encoder": True,
            "trainable_parameter_count": 123,
            "gradient_accumulation_steps": 2,
        },
    )

    row = parse_metrics_file(metrics_path)

    assert row["method_family"] == "long_context_transformer"
    assert row["method_family_label"] == "Long-context Transformer"
    assert "max_length=1024" in row["key_settings"]
    assert "freeze_encoder=True" in row["key_settings"]


def _write_metrics(path, metrics):
    path.write_text(json.dumps(metrics), encoding="utf-8")
