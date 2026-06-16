from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longdoc_transformer_classifier.config import (
    BaselineModelConfig,
    DatasetConfig,
    ReportConfig,
    TfidfConfig,
)
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.features import fit_tfidf, transform_tfidf
from longdoc_transformer_classifier.models.baseline import predict, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--max-train-samples", type=int, default=5_000)
    parser.add_argument("--max-test-samples", type=int, default=1_000)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run(args)
    print_metrics(metrics)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    ngram_range = _parse_ngram_range(args.ngram_min, args.ngram_max)
    dataset_config = DatasetConfig(
        name=args.dataset,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    feature_config = TfidfConfig(max_features=args.max_features, ngram_range=ngram_range)
    model_config = BaselineModelConfig(
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    report_config = ReportConfig(reports_dir=args.reports_dir)

    dataset = load_text_classification_dataset(dataset_config)
    vectorizer, train_features = fit_tfidf(dataset.train_texts, feature_config)
    test_features = transform_tfidf(vectorizer, dataset.test_texts)
    classifier = train(train_features, dataset.train_labels, model_config)
    predictions = predict(classifier, test_features)
    metrics = compute_classification_metrics(
        dataset.test_labels,
        predictions,
        label_names=dataset.label_names,
    )
    metrics_with_metadata = {
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "train_size": len(dataset.train_texts),
        "test_size": len(dataset.test_texts),
        **metrics,
    }

    report_config.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_config.reports_dir / f"baseline_{dataset.dataset_name}_metrics.json"
    report_path = report_config.reports_dir / f"baseline_{dataset.dataset_name}.md"
    metrics_path.write_text(
        json.dumps(metrics_with_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _build_markdown_report(
            metrics=metrics_with_metadata,
            dataset_name=dataset.dataset_name,
            hf_path=dataset.hf_path,
            text_field=dataset.text_field,
            label_field=dataset.label_field,
            train_size=len(dataset.train_texts),
            test_size=len(dataset.test_texts),
            max_features=feature_config.max_features,
            ngram_range=feature_config.ngram_range,
            random_state=model_config.random_state,
        ),
        encoding="utf-8",
    )

    return metrics_with_metadata


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"TF-IDF + Logistic Regression baseline on {metrics['dataset_name']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print("Per-class F1:")
    for label_name, score in metrics["per_class_f1"].items():
        print(f"  {label_name}: {score:.4f}")


def _parse_ngram_range(ngram_min: int, ngram_max: int) -> tuple[int, int]:
    if ngram_min < 1:
        msg = "--ngram-min must be at least 1."
        raise ValueError(msg)
    if ngram_max < ngram_min:
        msg = "--ngram-max must be greater than or equal to --ngram-min."
        raise ValueError(msg)
    return ngram_min, ngram_max


def _build_markdown_report(
    metrics: dict[str, Any],
    dataset_name: str,
    hf_path: str,
    text_field: str,
    label_field: str,
    train_size: int,
    test_size: int,
    max_features: int,
    ngram_range: tuple[int, int],
    random_state: int,
) -> str:
    lines = [
        f"# {dataset_name} Baseline Report",
        "",
        "## Model",
        "",
        "TF-IDF features with Logistic Regression.",
        "",
        "## Dataset",
        "",
        f"- Dataset name: `{dataset_name}`",
        f"- Hugging Face dataset: `{hf_path}`",
        f"- Text field: `{text_field}`",
        f"- Label field: `{label_field}`",
        "",
        "## Configuration",
        "",
        f"- Train samples: {train_size}",
        f"- Test samples: {test_size}",
        f"- Max TF-IDF features: {max_features}",
        f"- N-gram range: {ngram_range[0]}-{ngram_range[1]}",
        f"- Random state: {random_state}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        "",
        "## Per-Class F1",
        "",
        "| Class | F1 |",
        "| --- | ---: |",
    ]
    for label_name, score in metrics["per_class_f1"].items():
        lines.append(f"| {label_name} | {score:.4f} |")

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            _format_confusion_matrix(metrics["label_names"], metrics["confusion_matrix"]),
            "",
        ]
    )
    return "\n".join(lines)


def _format_confusion_matrix(label_names: list[str], matrix: list[list[int]]) -> str:
    header = "| true/pred | " + " | ".join(label_names) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in label_names) + " |"
    rows = [header, separator]
    for label_name, row in zip(label_names, matrix, strict=True):
        rows.append(f"| {label_name} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
