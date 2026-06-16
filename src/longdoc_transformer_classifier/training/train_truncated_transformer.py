from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.length_analysis import summarize_numbers
from longdoc_transformer_classifier.models.transformer_baseline import (
    TruncatedTextDataset,
    load_sequence_classifier,
    load_tokenizer,
)

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a truncated transformer baseline.")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/hf_cache/models"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run(args)
    print_summary(metrics)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_text_classification_dataset(
        DatasetConfig(
            name=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )
    tokenizer = load_tokenizer(args.model_name, cache_dir=str(args.model_cache_dir))
    train_dataset = TruncatedTextDataset(
        dataset.train_texts,
        dataset.train_labels,
        tokenizer,
        args.max_length,
    )
    test_dataset = TruncatedTextDataset(
        dataset.test_texts,
        dataset.test_labels,
        tokenizer,
        args.max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = load_sequence_classifier(
        args.model_name,
        num_labels=len(dataset.label_names),
        cache_dir=str(args.model_cache_dir),
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_loop(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )
    predictions = evaluate_model(model, test_loader, device)
    metrics = compute_classification_metrics(
        dataset.test_labels,
        predictions,
        label_names=dataset.label_names,
    )
    report_path = args.reports_dir / f"truncated_transformer_{dataset.dataset_name}.md"
    limitations = [
        "Only the first fixed token window can affect predictions.",
        "Tiny smoke models validate plumbing, not final performance.",
    ]
    report = {
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "train_size": len(dataset.train_texts),
        "test_size": len(dataset.test_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "random_state": args.random_state,
        "device": str(device),
        "limitations": limitations,
        "metadata": {
            "method": f"truncated_transformer_{dataset.dataset_name}",
            "dataset": dataset.dataset_name,
            "model_name": args.model_name,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "report_path": str(report_path),
            "limitations": limitations,
        },
        "truncation_note": (
            "This is a truncation baseline: each document is tokenized once and only the "
            "first max_length tokens are available to the classifier."
        ),
        "truncation_diagnostics": build_truncation_diagnostics(
            dataset.test_texts,
            max_length=args.max_length,
        ),
        **metrics,
    }
    write_reports(report, args.reports_dir)
    return report


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if step == 1 or step == len(train_loader) or step % 10 == 0:
                mean_loss = running_loss / step
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"loss {mean_loss:.4f}"
                )


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> list[int]:
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(int(label) for label in batch_predictions.cpu().tolist())
    return predictions


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_truncation_diagnostics(texts: list[str], max_length: int) -> dict[str, Any]:
    whitespace_counts = [len(text.split()) for text in texts]
    stats = summarize_numbers(whitespace_counts)
    documents_above = sum(count > max_length for count in whitespace_counts)
    percent_above = 100.0 * documents_above / len(texts) if texts else 0.0
    return {
        "max_length": max_length,
        "document_count": len(texts),
        "mean_whitespace_tokens": stats["mean"],
        "median_whitespace_tokens": stats["median"],
        "p95_whitespace_tokens": stats["p95"],
        "documents_above_max_length": int(documents_above),
        "percent_above_max_length": float(percent_above),
    }


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = report["dataset_name"]
    json_path = reports_dir / f"truncated_transformer_{dataset_name}_metrics.json"
    markdown_path = reports_dir / f"truncated_transformer_{dataset_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(report: dict[str, Any]) -> str:
    diagnostics = report["truncation_diagnostics"]
    lines = [
        f"# {report['dataset_name']} Truncated Transformer Report",
        "",
        "> This is a truncation baseline, not a long-document solution. Each document is "
        "tokenized once and only the first `max_length` tokens can affect the prediction.",
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
        f"- Model name: `{report['model_name']}`",
        f"- Max length: {report['max_length']}",
        f"- Max train samples: {report['max_train_samples']}",
        f"- Max test samples: {report['max_test_samples']}",
        f"- Actual train size: {report['train_size']}",
        f"- Actual test size: {report['test_size']}",
        f"- Epochs: {report['epochs']}",
        f"- Batch size: {report['batch_size']}",
        f"- Learning rate: {report['learning_rate']}",
        f"- Device: {report['device']}",
        "",
        "## Truncation Diagnostics",
        "",
        f"- Mean whitespace tokens: {_format_optional_float(diagnostics['mean_whitespace_tokens'])}",
        f"- Median whitespace tokens: "
        f"{_format_optional_float(diagnostics['median_whitespace_tokens'])}",
        f"- P95 whitespace tokens: {_format_optional_float(diagnostics['p95_whitespace_tokens'])}",
        f"- Documents above max length: {diagnostics['documents_above_max_length']} "
        f"({diagnostics['percent_above_max_length']:.2f}%)",
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
        ]
    )
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    diagnostics = report["truncation_diagnostics"]
    print(f"Truncated transformer baseline on {report['dataset_name']}")
    print(f"Model:     {report['model_name']}")
    print(f"Device:    {report['device']}")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Macro-F1:  {report['macro_f1']:.4f}")
    print(
        "Above max length: "
        f"{diagnostics['documents_above_max_length']} "
        f"({diagnostics['percent_above_max_length']:.2f}%)"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        msg = "--epochs must be greater than 0."
        raise ValueError(msg)
    if args.batch_size <= 0:
        msg = "--batch-size must be greater than 0."
        raise ValueError(msg)
    if args.max_length <= 0:
        msg = "--max-length must be greater than 0."
        raise ValueError(msg)


def _format_confusion_matrix(label_names: list[str], matrix: list[list[int]]) -> str:
    header = "| true/pred | " + " | ".join(label_names) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in label_names) + " |"
    rows = [header, separator]
    for label_name, row in zip(label_names, matrix, strict=True):
        rows.append(f"| {label_name} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
