import json

from longdoc_transformer_classifier.training.compare_reports import (
    build_comparison,
    load_metric_rows,
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


def _write_metrics(path, metrics):
    path.write_text(json.dumps(metrics), encoding="utf-8")
