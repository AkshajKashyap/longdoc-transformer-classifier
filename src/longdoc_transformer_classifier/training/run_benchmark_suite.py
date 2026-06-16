from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from longdoc_transformer_classifier.benchmark_config import (
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_CHUNK_SELECTION,
    DEFAULT_REPORTS_DIR,
    QUICK_SAMPLE_CONFIG,
    STANDARD_SAMPLE_CONFIG,
    TINY_TRANSFORMER_MODEL,
    BenchmarkSampleConfig,
)
from longdoc_transformer_classifier.chunk_selection import CHUNK_SELECTION_STRATEGIES


@dataclass(frozen=True)
class BenchmarkCommand:
    name: str
    module: str
    args: tuple[str, ...]
    description: str

    def to_argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small reproducible benchmark suite.")
    parser.add_argument("--dataset", default=DEFAULT_BENCHMARK_DATASET)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument("--include-summary", action="store_true")
    parser.add_argument(
        "--chunk-selection",
        choices=sorted(CHUNK_SELECTION_STRATEGIES),
        default=DEFAULT_CHUNK_SELECTION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_command_plan(
        dataset=args.dataset,
        quick=args.quick,
        include_transformers=args.include_transformers,
        include_summary=args.include_summary,
        output_dir=args.output_dir,
        chunk_selection=args.chunk_selection,
    )
    return run_command_plan(commands)


def build_command_plan(
    dataset: str,
    quick: bool,
    include_transformers: bool,
    include_summary: bool,
    output_dir: Path,
    chunk_selection: str = DEFAULT_CHUNK_SELECTION,
) -> list[BenchmarkCommand]:
    samples = QUICK_SAMPLE_CONFIG if quick else STANDARD_SAMPLE_CONFIG
    output_dir_text = str(output_dir)
    commands = [
        _analysis_command(dataset, samples, output_dir_text),
        _baseline_command(dataset, samples, output_dir_text),
    ]

    if include_transformers:
        commands.extend(
            [
                _truncated_transformer_command(dataset, samples, output_dir_text),
                _chunked_transformer_command(dataset, samples, output_dir_text, chunk_selection),
            ]
        )

    if include_summary:
        commands.append(_summary_command(dataset, samples, output_dir_text))

    commands.append(_compare_command(output_dir_text))
    return commands


def run_command_plan(commands: list[BenchmarkCommand]) -> int:
    print("Benchmark command plan:", flush=True)
    for index, command in enumerate(commands, start=1):
        print(f"{index}. {command.name}: {command.description}", flush=True)

    for command in commands:
        argv = command.to_argv()
        print("", flush=True)
        print("$ " + " ".join(argv), flush=True)
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError as error:
            print("", flush=True)
            print(f"Benchmark step failed: {command.name}", flush=True)
            print(f"Exit code: {error.returncode}", flush=True)
            return int(error.returncode)
    return 0


def _analysis_command(
    dataset: str,
    samples: BenchmarkSampleConfig,
    output_dir: str,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="length-analysis",
        module="longdoc_transformer_classifier.training.analyze_dataset",
        args=(
            "--dataset",
            dataset,
            "--max-train-samples",
            str(samples.analysis_train_samples),
            "--max-test-samples",
            str(samples.analysis_test_samples),
            "--reports-dir",
            output_dir,
        ),
        description="Measure document lengths and BERT-window truncation risk.",
    )


def _baseline_command(
    dataset: str,
    samples: BenchmarkSampleConfig,
    output_dir: str,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="tfidf-baseline",
        module="longdoc_transformer_classifier.training.train_baseline",
        args=(
            "--dataset",
            dataset,
            "--max-train-samples",
            str(samples.baseline_train_samples),
            "--max-test-samples",
            str(samples.baseline_test_samples),
            "--reports-dir",
            output_dir,
        ),
        description="Train the classical TF-IDF + Logistic Regression baseline.",
    )


def _truncated_transformer_command(
    dataset: str,
    samples: BenchmarkSampleConfig,
    output_dir: str,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="truncated-transformer-smoke",
        module="longdoc_transformer_classifier.training.train_truncated_transformer",
        args=(
            "--dataset",
            dataset,
            "--model-name",
            TINY_TRANSFORMER_MODEL,
            "--max-train-samples",
            str(samples.transformer_train_samples),
            "--max-test-samples",
            str(samples.transformer_test_samples),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-length",
            "256" if dataset == "arxiv" else "128",
            "--reports-dir",
            output_dir,
        ),
        description="Run a tiny transformer truncation smoke check.",
    )


def _chunked_transformer_command(
    dataset: str,
    samples: BenchmarkSampleConfig,
    output_dir: str,
    chunk_selection: str,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="chunked-transformer-smoke",
        module="longdoc_transformer_classifier.training.train_chunked_transformer",
        args=(
            "--dataset",
            dataset,
            "--model-name",
            TINY_TRANSFORMER_MODEL,
            "--max-train-samples",
            str(samples.transformer_train_samples),
            "--max-test-samples",
            str(samples.transformer_test_samples),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--max-length",
            "256" if dataset == "arxiv" else "128",
            "--chunk-size",
            "220" if dataset == "arxiv" else "100",
            "--chunk-overlap",
            "40" if dataset == "arxiv" else "20",
            "--max-chunks-per-doc",
            "8" if dataset == "arxiv" else "4",
            "--aggregation",
            "mean_proba",
            "--chunk-selection",
            chunk_selection,
            "--reports-dir",
            output_dir,
        ),
        description="Run a tiny chunked transformer smoke check.",
    )


def _summary_command(
    dataset: str,
    samples: BenchmarkSampleConfig,
    output_dir: str,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="summary-first-smoke",
        module="longdoc_transformer_classifier.training.train_summary_classifier",
        args=(
            "--dataset",
            dataset,
            "--max-train-samples",
            str(samples.summary_train_samples),
            "--max-test-samples",
            str(samples.summary_test_samples),
            "--summary-max-input-tokens",
            "1024" if dataset == "arxiv" else "512",
            "--summary-max-new-tokens",
            "120" if dataset == "arxiv" else "80",
            "--summary-min-new-tokens",
            "30" if dataset == "arxiv" else "20",
            "--summary-num-beams",
            "2",
            "--classifier",
            "tfidf",
            "--reports-dir",
            output_dir,
        ),
        description="Run summary-first classification using cached summaries when available.",
    )


def _compare_command(output_dir: str) -> BenchmarkCommand:
    return BenchmarkCommand(
        name="compare-reports",
        module="longdoc_transformer_classifier.training.compare_reports",
        args=("--reports-dir", output_dir),
        description="Refresh the unified model comparison report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
