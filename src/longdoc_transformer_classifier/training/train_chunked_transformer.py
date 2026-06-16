from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from longdoc_transformer_classifier.chunking import DocumentChunk, chunk_documents
from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.models.transformer_baseline import (
    load_sequence_classifier,
    load_tokenizer,
    tokenize_texts,
)
from longdoc_transformer_classifier.training.train_truncated_transformer import (
    DEFAULT_MODEL_NAME,
    move_batch_to_device,
    set_seed,
    train_loop,
)

SUPPORTED_AGGREGATIONS = {"mean_proba", "max_proba", "majority_vote"}


@dataclass(frozen=True)
class ChunkedDocumentExamples:
    chunk_texts: list[str]
    chunk_labels: list[int]
    document_ids: list[int]
    chunks_per_document_before_cap: list[int]
    chunks_per_document_after_cap: list[int]


class ChunkedTextDataset(Dataset):
    def __init__(
        self,
        chunk_texts: Sequence[str],
        chunk_labels: Sequence[int],
        document_ids: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(chunk_texts) != len(chunk_labels) or len(chunk_texts) != len(document_ids):
            msg = "chunk_texts, chunk_labels, and document_ids must have the same length."
            raise ValueError(msg)

        self.encodings = tokenize_texts(chunk_texts, tokenizer, max_length)
        self.labels = torch.tensor([int(label) for label in chunk_labels], dtype=torch.long)
        self.document_ids = torch.tensor([int(document_id) for document_id in document_ids])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "document_ids": self.document_ids[index],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chunked transformer baseline.")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--max-chunks-per-doc", type=int, default=8)
    parser.add_argument("--aggregation", choices=sorted(SUPPORTED_AGGREGATIONS), default="mean_proba")
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
    train_chunks = expand_documents_to_chunks(
        dataset.train_texts,
        dataset.train_labels,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )
    test_chunks = expand_documents_to_chunks(
        dataset.test_texts,
        dataset.test_labels,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )

    tokenizer = load_tokenizer(args.model_name, cache_dir=str(args.model_cache_dir))
    train_dataset = ChunkedTextDataset(
        train_chunks.chunk_texts,
        train_chunks.chunk_labels,
        train_chunks.document_ids,
        tokenizer,
        args.max_length,
    )
    test_dataset = ChunkedTextDataset(
        test_chunks.chunk_texts,
        test_chunks.chunk_labels,
        test_chunks.document_ids,
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

    model = load_sequence_classifier(
        args.model_name,
        num_labels=len(dataset.label_names),
        cache_dir=str(args.model_cache_dir),
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_loop(model, train_loader, optimizer, device, args.epochs)
    chunk_probabilities, document_ids = predict_chunk_probabilities(model, test_loader, device)
    predictions = aggregate_chunk_probabilities(
        chunk_probabilities,
        document_ids,
        aggregation=args.aggregation,
        num_documents=len(dataset.test_texts),
    )
    metrics = compute_classification_metrics(
        dataset.test_labels,
        predictions,
        label_names=dataset.label_names,
    )
    diagnostics = build_chunk_diagnostics(
        train_chunks,
        test_chunks,
        max_chunks_per_doc=args.max_chunks_per_doc,
        aggregation=args.aggregation,
    )
    report = {
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "model_name": args.model_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "max_length": args.max_length,
        "max_chunks_per_doc": args.max_chunks_per_doc,
        "aggregation": args.aggregation,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "train_size": len(dataset.train_texts),
        "test_size": len(dataset.test_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "random_state": args.random_state,
        "device": str(device),
        "limitation_note": (
            "Chunk labels are weak labels inherited from the parent document; not every chunk "
            "necessarily contains evidence for the document label."
        ),
        "chunk_diagnostics": diagnostics,
        **metrics,
    }
    write_reports(report, args.reports_dir)
    return report


def expand_documents_to_chunks(
    texts: Sequence[str],
    labels: Sequence[int],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks_per_doc: int,
) -> ChunkedDocumentExamples:
    if len(texts) != len(labels):
        msg = "texts and labels must have the same length."
        raise ValueError(msg)
    if max_chunks_per_doc <= 0:
        msg = "max_chunks_per_doc must be greater than 0."
        raise ValueError(msg)

    raw_chunks = chunk_documents(
        texts,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        document_ids=list(range(len(texts))),
    )
    chunks_by_document: dict[int, list[DocumentChunk]] = defaultdict(list)
    for chunk in raw_chunks:
        chunks_by_document[int(chunk.document_id)].append(chunk)

    chunk_texts: list[str] = []
    chunk_labels: list[int] = []
    document_ids: list[int] = []
    chunks_before_cap: list[int] = []
    chunks_after_cap: list[int] = []

    for document_id, (text, label) in enumerate(zip(texts, labels, strict=True)):
        document_chunks = chunks_by_document.get(document_id)
        if not document_chunks:
            document_chunks = [
                DocumentChunk(document_id=document_id, chunk_id=0, text="", start_word=0, end_word=0)
            ]

        chunks_before_cap.append(len(document_chunks))
        capped_chunks = document_chunks[:max_chunks_per_doc]
        chunks_after_cap.append(len(capped_chunks))

        for chunk in capped_chunks:
            chunk_texts.append(chunk.text)
            chunk_labels.append(int(label))
            document_ids.append(document_id)

    return ChunkedDocumentExamples(
        chunk_texts=chunk_texts,
        chunk_labels=chunk_labels,
        document_ids=document_ids,
        chunks_per_document_before_cap=chunks_before_cap,
        chunks_per_document_after_cap=chunks_after_cap,
    )


def predict_chunk_probabilities(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    document_ids: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            document_ids.extend(int(value) for value in batch["document_ids"].tolist())
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probabilities.append(torch.softmax(outputs.logits, dim=-1).cpu())

    return torch.cat(probabilities, dim=0), document_ids


def aggregate_chunk_probabilities(
    chunk_probabilities: torch.Tensor | Sequence[Sequence[float]],
    document_ids: Sequence[int],
    aggregation: str,
    num_documents: int,
) -> list[int]:
    if aggregation not in SUPPORTED_AGGREGATIONS:
        supported = ", ".join(sorted(SUPPORTED_AGGREGATIONS))
        msg = f"Unsupported aggregation '{aggregation}'. Supported values: {supported}."
        raise ValueError(msg)
    if num_documents <= 0:
        msg = "num_documents must be greater than 0."
        raise ValueError(msg)

    probabilities = torch.as_tensor(chunk_probabilities, dtype=torch.float)
    if len(probabilities) != len(document_ids):
        msg = "chunk_probabilities and document_ids must have the same length."
        raise ValueError(msg)

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, document_id in enumerate(document_ids):
        grouped_indices[int(document_id)].append(index)

    predictions: list[int] = []
    for document_id in range(num_documents):
        indices = grouped_indices.get(document_id)
        if not indices:
            msg = f"No chunks found for document id {document_id}."
            raise ValueError(msg)

        document_probabilities = probabilities[indices]
        if aggregation == "mean_proba":
            prediction = int(torch.argmax(document_probabilities.mean(dim=0)).item())
        elif aggregation == "max_proba":
            confidence = document_probabilities.max(dim=1).values
            most_confident_index = int(torch.argmax(confidence).item())
            prediction = int(torch.argmax(document_probabilities[most_confident_index]).item())
        else:
            chunk_predictions = torch.argmax(document_probabilities, dim=1).tolist()
            vote_counts = Counter(int(value) for value in chunk_predictions)
            prediction = min(
                vote_counts,
                key=lambda class_id: (-vote_counts[class_id], class_id),
            )
        predictions.append(prediction)

    return predictions


def build_chunk_diagnostics(
    train_chunks: ChunkedDocumentExamples,
    test_chunks: ChunkedDocumentExamples,
    max_chunks_per_doc: int,
    aggregation: str,
) -> dict[str, Any]:
    return {
        "train": _summarize_chunk_split(train_chunks, max_chunks_per_doc),
        "test": _summarize_chunk_split(test_chunks, max_chunks_per_doc),
        "max_chunks_per_doc": max_chunks_per_doc,
        "aggregation": aggregation,
    }


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = report["dataset_name"]
    json_path = reports_dir / f"chunked_transformer_{dataset_name}_metrics.json"
    markdown_path = reports_dir / f"chunked_transformer_{dataset_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(report: dict[str, Any]) -> str:
    diagnostics = report["chunk_diagnostics"]
    train_diagnostics = diagnostics["train"]
    test_diagnostics = diagnostics["test"]
    lines = [
        f"# {report['dataset_name']} Chunked Transformer Report",
        "",
        "> Chunk labels are weak labels inherited from the parent document. A chunk may not "
        "contain the evidence that justifies the document label.",
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
        f"- Chunk size: {report['chunk_size']}",
        f"- Chunk overlap: {report['chunk_overlap']}",
        f"- Max length: {report['max_length']}",
        f"- Max chunks per document: {report['max_chunks_per_doc']}",
        f"- Aggregation: `{report['aggregation']}`",
        f"- Max train samples: {report['max_train_samples']}",
        f"- Max test samples: {report['max_test_samples']}",
        f"- Actual train size: {report['train_size']}",
        f"- Actual test size: {report['test_size']}",
        f"- Epochs: {report['epochs']}",
        f"- Batch size: {report['batch_size']}",
        f"- Learning rate: {report['learning_rate']}",
        f"- Device: {report['device']}",
        "",
        "## Chunk Diagnostics",
        "",
        "| Split | Documents | Chunks | Avg Chunks/Doc | Max Before Cap | >1 Chunk | Hit Cap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _format_chunk_diagnostic_row("train", train_diagnostics),
        _format_chunk_diagnostic_row("test", test_diagnostics),
        "",
        "## Document-Level Metrics",
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
    diagnostics = report["chunk_diagnostics"]
    print(f"Chunked transformer baseline on {report['dataset_name']}")
    print(f"Model:      {report['model_name']}")
    print(f"Device:     {report['device']}")
    print(f"Aggregation:{report['aggregation']}")
    print(f"Train chunks: {diagnostics['train']['generated_chunks']}")
    print(f"Test chunks:  {diagnostics['test']['generated_chunks']}")
    print(f"Accuracy:   {report['accuracy']:.4f}")
    print(f"Macro-F1:   {report['macro_f1']:.4f}")


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
    if args.max_chunks_per_doc <= 0:
        msg = "--max-chunks-per-doc must be greater than 0."
        raise ValueError(msg)


def _summarize_chunk_split(
    chunks: ChunkedDocumentExamples,
    max_chunks_per_doc: int,
) -> dict[str, Any]:
    document_count = len(chunks.chunks_per_document_after_cap)
    generated_chunks = len(chunks.chunk_texts)
    before_cap_counts = chunks.chunks_per_document_before_cap
    after_cap_counts = chunks.chunks_per_document_after_cap
    more_than_one_count = sum(count > 1 for count in before_cap_counts)
    hit_cap_count = sum(count > max_chunks_per_doc for count in before_cap_counts)
    return {
        "original_documents": document_count,
        "generated_chunks": generated_chunks,
        "average_chunks_per_document": (
            generated_chunks / document_count if document_count else 0.0
        ),
        "max_chunks_per_document_before_cap": max(before_cap_counts) if before_cap_counts else 0,
        "percent_documents_with_more_than_one_chunk": (
            100.0 * more_than_one_count / document_count if document_count else 0.0
        ),
        "documents_hitting_max_chunks_per_doc_cap": int(hit_cap_count),
        "percent_documents_hitting_max_chunks_per_doc_cap": (
            100.0 * hit_cap_count / document_count if document_count else 0.0
        ),
        "chunks_per_document_after_cap": after_cap_counts,
    }


def _format_chunk_diagnostic_row(split_name: str, diagnostics: dict[str, Any]) -> str:
    return (
        f"| {split_name} | {diagnostics['original_documents']} | "
        f"{diagnostics['generated_chunks']} | "
        f"{diagnostics['average_chunks_per_document']:.2f} | "
        f"{diagnostics['max_chunks_per_document_before_cap']} | "
        f"{diagnostics['percent_documents_with_more_than_one_chunk']:.2f}% | "
        f"{diagnostics['percent_documents_hitting_max_chunks_per_doc_cap']:.2f}% |"
    )


def _format_confusion_matrix(label_names: list[str], matrix: list[list[int]]) -> str:
    header = "| true/pred | " + " | ".join(label_names) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in label_names) + " |"
    rows = [header, separator]
    for label_name, row in zip(label_names, matrix, strict=True):
        rows.append(f"| {label_name} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
