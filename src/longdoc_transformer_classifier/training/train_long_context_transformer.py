from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.length_analysis import summarize_numbers
from longdoc_transformer_classifier.models.transformer_baseline import tokenize_texts
from longdoc_transformer_classifier.training.train_truncated_transformer import (
    move_batch_to_device,
    set_seed,
)

DEFAULT_LONG_CONTEXT_MODEL = "allenai/longformer-base-4096"


class LongContextTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            msg = "texts and labels must have the same length."
            raise ValueError(msg)

        self.encodings = tokenize_texts(texts, tokenizer, max_length)
        self.labels = torch.tensor([int(label) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a long-context transformer baseline.")
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--model-name", default=DEFAULT_LONG_CONTEXT_MODEL)
    parser.add_argument("--max-train-samples", type=int, default=20)
    parser.add_argument("--max-test-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_text_classification_dataset(
        DatasetConfig(
            name=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(args.model_cache_dir),
        use_fast=False,
    )
    train_dataset = LongContextTextDataset(
        dataset.train_texts,
        dataset.train_labels,
        tokenizer,
        args.max_length,
    )
    test_dataset = LongContextTextDataset(
        dataset.test_texts,
        dataset.test_labels,
        tokenizer,
        args.max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(args.random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(dataset.label_names),
        cache_dir=str(args.model_cache_dir),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_encoder:
        freeze_base_encoder(model)
    parameter_counts = count_parameters(model)

    model.to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    predictions = evaluate_model(model, test_loader, device)
    metrics = compute_classification_metrics(
        dataset.test_labels,
        predictions,
        label_names=dataset.label_names,
    )
    diagnostics = build_long_context_diagnostics(dataset.test_texts, args.max_length)
    report_path = args.reports_dir / f"long_context_transformer_{dataset.dataset_name}.md"
    limitations = [
        "Long-context transformers still have a maximum context window.",
        "max_length=1024 smoke runs do not use the full 4096-token Longformer capacity.",
        "Freezing the encoder limits adaptation.",
        "Tiny sample results are not final model rankings.",
        "Long-context models are more expensive than TF-IDF or chunk heuristics.",
    ]
    report = {
        "method": f"long_context_transformer_{dataset.dataset_name}",
        "dataset": dataset.dataset_name,
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "freeze_encoder": args.freeze_encoder,
        "trainable_parameter_count": parameter_counts["trainable_parameter_count"],
        "total_parameter_count": parameter_counts["total_parameter_count"],
        "random_state": args.random_state,
        "device": str(device),
        "diagnostics": diagnostics,
        "limitations": limitations,
        "limitation": limitations[0],
        "structural_takeaway": (
            "Uses a transformer architecture designed for longer contexts, avoiding manual "
            "chunk aggregation but still bounded by max_length and compute."
        ),
        "metadata": {
            "method": f"long_context_transformer_{dataset.dataset_name}",
            "dataset": dataset.dataset_name,
            "model_name": args.model_name,
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


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epochs: int,
    gradient_accumulation_steps: int,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**build_model_inputs(batch, model, include_labels=True))
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(outputs.loss.item())
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
            outputs = model(**build_model_inputs(batch, model, include_labels=False))
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(int(label) for label in batch_predictions.cpu().tolist())
    return predictions


def build_model_inputs(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    include_labels: bool,
) -> dict[str, torch.Tensor]:
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if include_labels:
        inputs["labels"] = batch["labels"]
    if should_add_global_attention(model):
        inputs["global_attention_mask"] = build_global_attention_mask(batch["input_ids"])
    return inputs


def should_add_global_attention(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    return model_type == "longformer"


def build_global_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(input_ids)
    if mask.ndim != 2:
        msg = "input_ids must be a 2D tensor with shape [batch, sequence_length]."
        raise ValueError(msg)
    if mask.shape[1] > 0:
        mask[:, 0] = 1
    return mask


def freeze_base_encoder(model: torch.nn.Module) -> None:
    base_model = get_base_encoder(model)
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    if count_parameters(model)["trainable_parameter_count"] == 0:
        msg = "Freezing the encoder left no trainable parameters."
        raise ValueError(msg)


def get_base_encoder(model: torch.nn.Module) -> torch.nn.Module:
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, torch.nn.Module):
        return base_model

    base_model_prefix = getattr(model, "base_model_prefix", None)
    if base_model_prefix:
        candidate = getattr(model, str(base_model_prefix), None)
        if isinstance(candidate, torch.nn.Module):
            return candidate

    for attribute_name in ("longformer", "bert", "roberta", "distilbert", "deberta", "electra"):
        candidate = getattr(model, attribute_name, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate

    msg = (
        "Could not identify a base encoder to freeze for this model. "
        "Run without --freeze-encoder or add model-specific freeze support."
    )
    raise ValueError(msg)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
    }


def build_long_context_diagnostics(texts: list[str], max_length: int) -> dict[str, Any]:
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
    json_path = reports_dir / f"long_context_transformer_{dataset_name}_metrics.json"
    markdown_path = reports_dir / f"long_context_transformer_{dataset_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(report: dict[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    lines = [
        f"# {report['dataset_name']} Long-Context Transformer Report",
        "",
        "> This baseline uses a sequence classifier capable of longer context windows. It is "
        "still bounded by `max_length` and available compute.",
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
        f"- Gradient accumulation steps: {report['gradient_accumulation_steps']}",
        f"- Freeze encoder: {report['freeze_encoder']}",
        f"- Trainable parameters: {report['trainable_parameter_count']}",
        f"- Total parameters: {report['total_parameter_count']}",
        f"- Learning rate: {report['learning_rate']}",
        f"- Device: {report['device']}",
        "",
        "## Length Diagnostics",
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
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    diagnostics = report["diagnostics"]
    print(f"Long-context transformer baseline on {report['dataset_name']}")
    print(f"Model:      {report['model_name']}")
    print(f"Device:     {report['device']}")
    print(f"Max length: {report['max_length']}")
    print(f"Freeze encoder: {report['freeze_encoder']}")
    print(
        "Trainable params: "
        f"{report['trainable_parameter_count']}/{report['total_parameter_count']}"
    )
    print(f"Accuracy:   {report['accuracy']:.4f}")
    print(f"Macro-F1:   {report['macro_f1']:.4f}")
    print(
        "Above max length: "
        f"{diagnostics['documents_above_max_length']} "
        f"({diagnostics['percent_above_max_length']:.2f}%)"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        msg = "--max-train-samples must be greater than 0."
        raise ValueError(msg)
    if args.max_test_samples is not None and args.max_test_samples <= 0:
        msg = "--max-test-samples must be greater than 0."
        raise ValueError(msg)
    if args.epochs <= 0:
        msg = "--epochs must be greater than 0."
        raise ValueError(msg)
    if args.batch_size <= 0:
        msg = "--batch-size must be greater than 0."
        raise ValueError(msg)
    if args.max_length <= 0:
        msg = "--max-length must be greater than 0."
        raise ValueError(msg)
    if args.gradient_accumulation_steps <= 0:
        msg = "--gradient-accumulation-steps must be greater than 0."
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
