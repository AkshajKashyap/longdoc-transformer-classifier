from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.length_analysis import (
    analyze_dataset_lengths,
    build_length_analysis_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze text lengths for a supported dataset.")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--max-train-samples", type=int, default=1_000)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = run(args)
    print_summary(analysis)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_text_classification_dataset(
        DatasetConfig(
            name=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )
    analysis = analyze_dataset_lengths(dataset)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset.dataset_name
    json_path = args.reports_dir / f"{dataset_name}_length_analysis.json"
    markdown_path = args.reports_dir / f"{dataset_name}_length_analysis.md"

    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_length_analysis_markdown(analysis), encoding="utf-8")
    return analysis


def print_summary(analysis: dict[str, Any]) -> None:
    combined = analysis["splits"]["combined"]
    token_stats = combined["whitespace_token_counts"]
    print(f"Length analysis for {analysis['dataset_name']} ({analysis['hf_path']})")
    print(f"Documents: {combined['document_count']}")
    print(f"Whitespace-token mean: {token_stats['mean']:.1f}")
    print(f"Whitespace-token p95:  {token_stats['p95']:.1f}")
    print(
        "Above 512 whitespace tokens: "
        f"{combined['documents_above_bert_limit']} "
        f"({combined['percent_above_bert_limit']:.2f}%)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
