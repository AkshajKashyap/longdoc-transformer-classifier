from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"


@dataclass(frozen=True)
class SummaryGenerationConfig:
    max_input_tokens: int = 1024
    max_new_tokens: int = 160
    min_new_tokens: int = 40
    num_beams: int = 2


@dataclass(frozen=True)
class SummaryRecord:
    doc_id: int
    label: int
    original_word_count: int
    summarizer_input_token_count: int
    summarizer_input_word_count: int
    summary_word_count: int
    compression_ratio: float
    summary_text: str


class HuggingFaceSummarizer:
    def __init__(
        self,
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        cache_dir: str | Path = Path("data/hf_cache/models"),
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model.to(self.device)
        self.model.eval()

    def summarize_batch(
        self,
        texts: Sequence[str],
        config: SummaryGenerationConfig,
    ) -> tuple[list[str], list[int], list[int]]:
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=config.max_input_tokens,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.min_new_tokens,
                num_beams=config.num_beams,
                do_sample=False,
            )

        summaries = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        input_token_counts = [int(mask.sum().item()) for mask in attention_mask.cpu()]
        input_word_counts = [
            len(
                self.tokenizer.decode(ids[:token_count], skip_special_tokens=True).split()
            )
            for ids, token_count in zip(encoded["input_ids"], input_token_counts, strict=True)
        ]
        return summaries, input_token_counts, input_word_counts


def set_summarization_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_summary_records(
    texts: Sequence[str],
    labels: Sequence[int],
    summarizer: HuggingFaceSummarizer,
    config: SummaryGenerationConfig,
    batch_size: int = 2,
) -> list[SummaryRecord]:
    if len(texts) != len(labels):
        msg = "texts and labels must have the same length."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be greater than 0."
        raise ValueError(msg)

    records: list[SummaryRecord] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = list(texts[start:end])
        batch_labels = list(labels[start:end])
        summaries, input_token_counts, input_word_counts = summarizer.summarize_batch(
            batch_texts,
            config,
        )
        for offset, (text, label, summary) in enumerate(
            zip(batch_texts, batch_labels, summaries, strict=True)
        ):
            original_word_count = len(text.split())
            summary_word_count = len(summary.split())
            compression_ratio = (
                summary_word_count / original_word_count if original_word_count else 0.0
            )
            records.append(
                SummaryRecord(
                    doc_id=start + offset,
                    label=int(label),
                    original_word_count=original_word_count,
                    summarizer_input_token_count=input_token_counts[offset],
                    summarizer_input_word_count=input_word_counts[offset],
                    summary_word_count=summary_word_count,
                    compression_ratio=float(compression_ratio),
                    summary_text=summary,
                )
            )
        print(f"Summarized {end}/{len(texts)} documents")
    return records


def build_summary_cache_path(
    cache_dir: Path,
    dataset_name: str,
    split: str,
    summarizer_model: str,
    sample_size: int,
    config: SummaryGenerationConfig,
) -> Path:
    key = build_summary_cache_key(dataset_name, split, summarizer_model, sample_size, config)
    slug = model_slug(summarizer_model)
    return cache_dir / f"{dataset_name}_{split}_{slug}_{key}.jsonl"


def build_summary_cache_key(
    dataset_name: str,
    split: str,
    summarizer_model: str,
    sample_size: int,
    config: SummaryGenerationConfig,
) -> str:
    payload = {
        "dataset_name": dataset_name,
        "split": split,
        "summarizer_model": summarizer_model,
        "sample_size": sample_size,
        **asdict(config),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def save_summary_records(records: Sequence[SummaryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_summary_records(path: Path) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(SummaryRecord(**json.loads(line)))
    return records


def summary_texts_and_labels(records: Sequence[SummaryRecord]) -> tuple[list[str], list[int]]:
    return [record.summary_text for record in records], [int(record.label) for record in records]


def calculate_summary_statistics(records: Sequence[SummaryRecord]) -> dict[str, Any]:
    document_count = len(records)
    original_counts = [record.original_word_count for record in records]
    input_token_counts = [record.summarizer_input_token_count for record in records]
    input_word_counts = [record.summarizer_input_word_count for record in records]
    summary_counts = [record.summary_word_count for record in records]
    compression_ratios = [record.compression_ratio for record in records]
    truncated_count = sum(
        record.summarizer_input_word_count < record.original_word_count for record in records
    )
    return {
        "document_count": document_count,
        "average_original_word_count": _mean(original_counts),
        "average_summarizer_input_token_count": _mean(input_token_counts),
        "average_summarizer_input_word_count": _mean(input_word_counts),
        "average_summary_word_count": _mean(summary_counts),
        "average_compression_ratio": _mean(compression_ratios),
        "documents_with_summarizer_input_shorter_than_original": int(truncated_count),
        "percent_summarizer_input_shorter_than_original": (
            100.0 * truncated_count / document_count if document_count else 0.0
        ),
    }


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
