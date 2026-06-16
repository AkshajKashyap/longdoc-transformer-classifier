from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from longdoc_transformer_classifier.data import TextClassificationDataset

BERT_WHITESPACE_TOKEN_LIMIT = 512


def analyze_dataset_lengths(
    dataset: TextClassificationDataset,
    bert_token_limit: int = BERT_WHITESPACE_TOKEN_LIMIT,
) -> dict[str, Any]:
    train_summary = summarize_document_lengths(dataset.train_texts, bert_token_limit)
    test_summary = summarize_document_lengths(dataset.test_texts, bert_token_limit)
    combined_summary = summarize_document_lengths(
        [*dataset.train_texts, *dataset.test_texts],
        bert_token_limit,
    )

    return {
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "label_names": dataset.label_names,
        "bert_whitespace_token_limit": bert_token_limit,
        "splits": {
            "train": train_summary,
            "test": test_summary,
            "combined": combined_summary,
        },
    }


def summarize_document_lengths(
    texts: list[str],
    bert_token_limit: int = BERT_WHITESPACE_TOKEN_LIMIT,
) -> dict[str, Any]:
    character_lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    whitespace_token_counts = word_counts

    above_limit_count = sum(count > bert_token_limit for count in whitespace_token_counts)
    percent_above_limit = (
        100.0 * above_limit_count / len(whitespace_token_counts)
        if whitespace_token_counts
        else 0.0
    )

    return {
        "document_count": len(texts),
        "character_lengths": summarize_numbers(character_lengths),
        "word_counts": summarize_numbers(word_counts),
        "whitespace_token_counts": summarize_numbers(whitespace_token_counts),
        "documents_above_bert_limit": int(above_limit_count),
        "percent_above_bert_limit": float(percent_above_limit),
    }


def summarize_numbers(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def build_length_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        f"# {analysis['dataset_name']} Length Analysis",
        "",
        f"- Hugging Face dataset: `{analysis['hf_path']}`",
        f"- Text field: `{analysis['text_field']}`",
        f"- Label field: `{analysis['label_field']}`",
        f"- BERT limit approximation: {analysis['bert_whitespace_token_limit']} whitespace tokens",
        "",
        "## Split Summary",
        "",
    ]

    for split_name, summary in analysis["splits"].items():
        lines.extend(_format_split_summary(split_name, summary))

    return "\n".join(lines) + "\n"


def _format_split_summary(split_name: str, summary: dict[str, Any]) -> list[str]:
    token_stats = summary["whitespace_token_counts"]
    char_stats = summary["character_lengths"]
    return [
        f"### {split_name.title()}",
        "",
        f"- Documents: {summary['document_count']}",
        f"- Characters mean / median / p95 / max: {_stat_line(char_stats)}",
        f"- Whitespace tokens mean / median / p95 / max: {_stat_line(token_stats)}",
        f"- Documents above BERT limit: {summary['documents_above_bert_limit']} "
        f"({summary['percent_above_bert_limit']:.2f}%)",
        "",
    ]


def _stat_line(stats: dict[str, int | float | None]) -> str:
    if stats["count"] == 0:
        return "n/a"
    return (
        f"{stats['mean']:.1f} / {stats['median']:.1f} / "
        f"{stats['p95']:.1f} / {stats['max']}"
    )


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)
