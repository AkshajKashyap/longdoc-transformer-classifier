from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from longdoc_transformer_classifier.config import BaselineModelConfig, DatasetConfig, TfidfConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.features import fit_tfidf, transform_tfidf
from longdoc_transformer_classifier.models.baseline import predict, train
from longdoc_transformer_classifier.models.transformer_baseline import (
    TruncatedTextDataset,
    load_sequence_classifier,
    load_tokenizer,
)
from longdoc_transformer_classifier.summarization import (
    DEFAULT_SUMMARIZER_MODEL,
    HuggingFaceSummarizer,
    SummaryGenerationConfig,
    build_summary_cache_path,
    calculate_summary_statistics,
    generate_summary_records,
    load_summary_records,
    save_summary_records,
    set_summarization_seed,
    summary_texts_and_labels,
)
from longdoc_transformer_classifier.training.train_truncated_transformer import (
    evaluate_model,
    set_seed,
    train_loop,
)

SUPPORTED_CLASSIFIERS = {"tfidf", "transformer"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a summary-first classifier baseline.")
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--max-train-samples", type=int, default=100)
    parser.add_argument("--max-test-samples", type=int, default=50)
    parser.add_argument("--summarizer-model", default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument("--summary-max-input-tokens", type=int, default=1024)
    parser.add_argument("--summary-max-new-tokens", type=int, default=160)
    parser.add_argument("--summary-min-new-tokens", type=int, default=40)
    parser.add_argument("--summary-num-beams", type=int, default=2)
    parser.add_argument("--summary-batch-size", type=int, default=2)
    parser.add_argument("--force-regenerate-summaries", action="store_true")
    parser.add_argument("--classifier", choices=sorted(SUPPORTED_CLASSIFIERS), default="tfidf")
    parser.add_argument("--classifier-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--summary-cache-dir", type=Path, default=Path("data/processed/summaries"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/hf_cache/models"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print_summary(report)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    set_seed(args.random_state)
    set_summarization_seed(args.random_state)

    dataset = load_text_classification_dataset(
        DatasetConfig(
            name=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )
    summary_config = SummaryGenerationConfig(
        max_input_tokens=args.summary_max_input_tokens,
        max_new_tokens=args.summary_max_new_tokens,
        min_new_tokens=args.summary_min_new_tokens,
        num_beams=args.summary_num_beams,
    )
    train_cache_path = build_summary_cache_path(
        args.summary_cache_dir,
        dataset.dataset_name,
        "train",
        args.summarizer_model,
        len(dataset.train_texts),
        summary_config,
    )
    test_cache_path = build_summary_cache_path(
        args.summary_cache_dir,
        dataset.dataset_name,
        "test",
        args.summarizer_model,
        len(dataset.test_texts),
        summary_config,
    )

    records_by_split = load_or_generate_summaries(
        train_texts=dataset.train_texts,
        train_labels=dataset.train_labels,
        test_texts=dataset.test_texts,
        test_labels=dataset.test_labels,
        train_cache_path=train_cache_path,
        test_cache_path=test_cache_path,
        summarizer_model=args.summarizer_model,
        model_cache_dir=args.model_cache_dir,
        config=summary_config,
        summary_batch_size=args.summary_batch_size,
        force_regenerate=args.force_regenerate_summaries,
    )
    train_records = records_by_split["train_records"]
    test_records = records_by_split["test_records"]
    train_summary_texts, train_labels = summary_texts_and_labels(train_records)
    test_summary_texts, test_labels = summary_texts_and_labels(test_records)

    if args.classifier == "tfidf":
        metrics = train_tfidf_summary_classifier(
            train_summary_texts,
            train_labels,
            test_summary_texts,
            test_labels,
            dataset.label_names,
            args,
        )
        classifier_device = "cpu"
    else:
        metrics = train_transformer_summary_classifier(
            train_summary_texts,
            train_labels,
            test_summary_texts,
            test_labels,
            dataset.label_names,
            args,
        )
        classifier_device = metrics.pop("classifier_device")

    summary_statistics = calculate_summary_statistics([*train_records, *test_records])
    report_path = args.reports_dir / f"summary_classifier_{dataset.dataset_name}.md"
    model_name = (
        f"{args.summarizer_model} + {args.classifier_model}"
        if args.classifier == "transformer"
        else f"{args.summarizer_model} + TF-IDF"
    )
    limitations = [
        "This is not full summarizer fine-tuning.",
        "The summarizer may only see the first part of very long documents.",
        "Summaries may remove class-discriminative details.",
        "Summary-first classification trades recall for compression.",
    ]
    report = {
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "summarizer_model": args.summarizer_model,
        "classifier": args.classifier,
        "classifier_model": args.classifier_model if args.classifier == "transformer" else None,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "train_size": len(dataset.train_texts),
        "test_size": len(dataset.test_texts),
        "summary_generation": {
            "summary_max_input_tokens": args.summary_max_input_tokens,
            "summary_max_new_tokens": args.summary_max_new_tokens,
            "summary_min_new_tokens": args.summary_min_new_tokens,
            "summary_num_beams": args.summary_num_beams,
            "summary_batch_size": args.summary_batch_size,
        },
        "cache": {
            "train_cache_path": str(train_cache_path),
            "test_cache_path": str(test_cache_path),
            "train_cache_status": records_by_split["train_cache_status"],
            "test_cache_status": records_by_split["test_cache_status"],
            "force_regenerate_summaries": args.force_regenerate_summaries,
        },
        "summary_statistics": summary_statistics,
        "classifier_settings": {
            "max_features": args.max_features if args.classifier == "tfidf" else None,
            "ngram_range": [args.ngram_min, args.ngram_max] if args.classifier == "tfidf" else None,
            "epochs": args.epochs if args.classifier == "transformer" else None,
            "batch_size": args.batch_size if args.classifier == "transformer" else None,
            "max_length": args.max_length if args.classifier == "transformer" else None,
            "learning_rate": args.learning_rate if args.classifier == "transformer" else None,
        },
        "classifier_device": classifier_device,
        "random_state": args.random_state,
        "limitations": limitations,
        "metadata": {
            "method": f"summary_classifier_{dataset.dataset_name}",
            "dataset": dataset.dataset_name,
            "model_name": model_name,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "report_path": str(report_path),
            "limitations": limitations,
        },
        **metrics,
    }
    write_reports(report, args.reports_dir)
    return report


def load_or_generate_summaries(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    train_cache_path: Path,
    test_cache_path: Path,
    summarizer_model: str,
    model_cache_dir: Path,
    config: SummaryGenerationConfig,
    summary_batch_size: int,
    force_regenerate: bool,
) -> dict[str, Any]:
    train_cache_exists = train_cache_path.exists()
    test_cache_exists = test_cache_path.exists()
    if train_cache_exists and test_cache_exists and not force_regenerate:
        return {
            "train_records": load_summary_records(train_cache_path),
            "test_records": load_summary_records(test_cache_path),
            "train_cache_status": "loaded",
            "test_cache_status": "loaded",
        }

    summarizer = HuggingFaceSummarizer(summarizer_model, cache_dir=model_cache_dir)
    if train_cache_exists and not force_regenerate:
        train_records = load_summary_records(train_cache_path)
        train_status = "loaded"
    else:
        train_records = generate_summary_records(
            train_texts,
            train_labels,
            summarizer,
            config,
            batch_size=summary_batch_size,
        )
        save_summary_records(train_records, train_cache_path)
        train_status = "regenerated"

    if test_cache_exists and not force_regenerate:
        test_records = load_summary_records(test_cache_path)
        test_status = "loaded"
    else:
        test_records = generate_summary_records(
            test_texts,
            test_labels,
            summarizer,
            config,
            batch_size=summary_batch_size,
        )
        save_summary_records(test_records, test_cache_path)
        test_status = "regenerated"

    return {
        "train_records": train_records,
        "test_records": test_records,
        "train_cache_status": train_status,
        "test_cache_status": test_status,
    }


def train_tfidf_summary_classifier(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    label_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    vectorizer, train_features = fit_tfidf(
        train_texts,
        TfidfConfig(max_features=args.max_features, ngram_range=(args.ngram_min, args.ngram_max)),
    )
    test_features = transform_tfidf(vectorizer, test_texts)
    classifier = train(train_features, train_labels, BaselineModelConfig())
    predictions = predict(classifier, test_features)
    return compute_classification_metrics(test_labels, predictions, label_names=label_names)


def train_transformer_summary_classifier(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    label_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.classifier_model, cache_dir=str(args.model_cache_dir))
    train_dataset = TruncatedTextDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset = TruncatedTextDataset(test_texts, test_labels, tokenizer, args.max_length)
    generator = torch.Generator()
    generator.manual_seed(args.random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = load_sequence_classifier(
        args.classifier_model,
        num_labels=len(label_names),
        cache_dir=str(args.model_cache_dir),
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_loop(model, train_loader, optimizer, device, args.epochs)
    predictions = evaluate_model(model, test_loader, device)
    metrics = compute_classification_metrics(test_labels, predictions, label_names=label_names)
    metrics["classifier_device"] = str(device)
    return metrics


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = report["dataset_name"]
    json_path = reports_dir / f"summary_classifier_{dataset_name}_metrics.json"
    markdown_path = reports_dir / f"summary_classifier_{dataset_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(report: dict[str, Any]) -> str:
    stats = report["summary_statistics"]
    generation = report["summary_generation"]
    cache = report["cache"]
    lines = [
        f"# {report['dataset_name']} Summary-First Classifier Report",
        "",
        "## Dataset",
        "",
        f"- Dataset name: `{report['dataset_name']}`",
        f"- Hugging Face dataset: `{report['hf_path']}`",
        f"- Text field: `{report['text_field']}`",
        f"- Label field: `{report['label_field']}`",
        "",
        "## Configuration",
        "",
        f"- Summarizer model: `{report['summarizer_model']}`",
        f"- Classifier: `{report['classifier']}`",
        f"- Classifier model: `{report['classifier_model'] or 'n/a'}`",
        f"- Max train samples: {report['max_train_samples']}",
        f"- Max test samples: {report['max_test_samples']}",
        f"- Actual train size: {report['train_size']}",
        f"- Actual test size: {report['test_size']}",
        f"- Summary max input tokens: {generation['summary_max_input_tokens']}",
        f"- Summary max new tokens: {generation['summary_max_new_tokens']}",
        f"- Summary min new tokens: {generation['summary_min_new_tokens']}",
        f"- Summary num beams: {generation['summary_num_beams']}",
        f"- Summary batch size: {generation['summary_batch_size']}",
        f"- Classifier device: {report['classifier_device']}",
        "",
        "## Cache",
        "",
        f"- Train cache path: `{cache['train_cache_path']}`",
        f"- Test cache path: `{cache['test_cache_path']}`",
        f"- Train cache status: `{cache['train_cache_status']}`",
        f"- Test cache status: `{cache['test_cache_status']}`",
        "",
        "## Summary Statistics",
        "",
        f"- Average original word count: {stats['average_original_word_count']:.1f}",
        f"- Average summarizer input tokens: "
        f"{stats['average_summarizer_input_token_count']:.1f}",
        f"- Average summarizer input words: {stats['average_summarizer_input_word_count']:.1f}",
        f"- Average summary word count: {stats['average_summary_word_count']:.1f}",
        f"- Average compression ratio: {stats['average_compression_ratio']:.4f}",
        "- Documents where summarizer input was shorter than original: "
        f"{stats['documents_with_summarizer_input_shorter_than_original']} "
        f"({stats['percent_summarizer_input_shorter_than_original']:.2f}%)",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {report['accuracy']:.4f}",
        f"- Macro-F1: {report['macro_f1']:.4f}",
        "",
        "## Per-Class F1",
        "",
        "| Class | F1 |",
        "| --- | ---: |",
    ]
    for label_name, score in report["per_class_f1"].items():
        lines.append(f"| {label_name} | {score:.4f} |")

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            _format_confusion_matrix(report["label_names"], report["confusion_matrix"]),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    stats = report["summary_statistics"]
    print(f"Summary-first classifier on {report['dataset_name']}")
    print(f"Summarizer: {report['summarizer_model']}")
    print(f"Classifier: {report['classifier']}")
    print(f"Accuracy:   {report['accuracy']:.4f}")
    print(f"Macro-F1:   {report['macro_f1']:.4f}")
    print(f"Avg summary words: {stats['average_summary_word_count']:.1f}")
    print(
        "Summarizer input shorter than original: "
        f"{stats['percent_summarizer_input_shorter_than_original']:.2f}%"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        msg = "--max-train-samples must be greater than 0."
        raise ValueError(msg)
    if args.max_test_samples is not None and args.max_test_samples <= 0:
        msg = "--max-test-samples must be greater than 0."
        raise ValueError(msg)
    if args.summary_max_input_tokens <= 0:
        msg = "--summary-max-input-tokens must be greater than 0."
        raise ValueError(msg)
    if args.summary_max_new_tokens <= 0:
        msg = "--summary-max-new-tokens must be greater than 0."
        raise ValueError(msg)
    if args.summary_min_new_tokens < 0:
        msg = "--summary-min-new-tokens must be greater than or equal to 0."
        raise ValueError(msg)
    if args.summary_min_new_tokens > args.summary_max_new_tokens:
        msg = "--summary-min-new-tokens must be less than or equal to --summary-max-new-tokens."
        raise ValueError(msg)
    if args.summary_num_beams <= 0:
        msg = "--summary-num-beams must be greater than 0."
        raise ValueError(msg)
    if args.summary_batch_size <= 0:
        msg = "--summary-batch-size must be greater than 0."
        raise ValueError(msg)
    if args.classifier == "transformer":
        if args.epochs <= 0:
            msg = "--epochs must be greater than 0."
            raise ValueError(msg)
        if args.batch_size <= 0:
            msg = "--batch-size must be greater than 0."
            raise ValueError(msg)
        if args.max_length <= 0:
            msg = "--max-length must be greater than 0."
            raise ValueError(msg)
    if args.ngram_min < 1:
        msg = "--ngram-min must be at least 1."
        raise ValueError(msg)
    if args.ngram_max < args.ngram_min:
        msg = "--ngram-max must be greater than or equal to --ngram-min."
        raise ValueError(msg)


def _format_confusion_matrix(label_names: list[str], matrix: list[list[int]]) -> str:
    header = "| true/pred | " + " | ".join(label_names) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in label_names) + " |"
    rows = [header, separator]
    for label_name, row in zip(label_names, matrix, strict=True):
        rows.append(f"| {label_name} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
