from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHOD_ORDER = [
    "baseline_ag_news",
    "baseline_arxiv",
    "truncated_transformer_ag_news",
    "truncated_transformer_arxiv",
    "chunked_transformer_ag_news",
    "chunked_transformer_arxiv",
    "summary_classifier_ag_news",
    "summary_classifier_arxiv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare available model reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison = run(args.reports_dir)
    print_summary(comparison)
    return 0


def run(reports_dir: Path = Path("reports")) -> dict[str, Any]:
    rows = load_metric_rows(reports_dir)
    comparison = build_comparison(rows)
    write_comparison_reports(comparison, reports_dir)
    return comparison


def load_metric_rows(reports_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in METHOD_ORDER:
        path = reports_dir / f"{method_name}_metrics.json"
        if path.exists():
            rows.append(parse_metrics_file(path))

    known_paths = {reports_dir / f"{method_name}_metrics.json" for method_name in METHOD_ORDER}
    for path in sorted(reports_dir.glob("*_metrics.json")):
        if path not in known_paths and not path.name.startswith("model_comparison"):
            rows.append(parse_metrics_file(path))
    return rows


def parse_metrics_file(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    method_name = path.name.removesuffix("_metrics.json")
    return {
        "method": method_name,
        "method_family": infer_method_family(method_name),
        "dataset": str(metrics.get("dataset_name") or infer_dataset_name(method_name)),
        "accuracy": _optional_float(metrics.get("accuracy")),
        "macro_f1": _optional_float(metrics.get("macro_f1")),
        "key_settings": summarize_settings(method_name, metrics),
        "metrics_path": str(path),
    }


def build_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        macro_f1 = row.get("macro_f1")
        if macro_f1 is None:
            continue
        dataset = row["dataset"]
        current_best = best_by_dataset.get(dataset)
        if current_best is None or macro_f1 > current_best["macro_f1"]:
            best_by_dataset[dataset] = row

    return {
        "rows": rows,
        "best_by_dataset": {
            dataset: {
                "method": row["method"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
            }
            for dataset, row in sorted(best_by_dataset.items())
        },
        "warning": (
            "Smoke runs with tiny models and tiny sample sizes are useful engineering checks, "
            "not final model rankings."
        ),
    }


def write_comparison_reports(comparison: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "model_comparison.json"
    markdown_path = reports_dir / "model_comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(comparison), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# Model Comparison",
        "",
        "> Smoke runs with tiny models and tiny sample sizes are useful engineering checks, "
        "not final model rankings.",
        "",
        "## Results",
        "",
        "| Method | Dataset | Accuracy | Macro-F1 | Key Settings |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| `{row['method']}` | `{row['dataset']}` | "
            f"{_format_metric(row['accuracy'])} | {_format_metric(row['macro_f1'])} | "
            f"{row['key_settings']} |"
        )

    lines.extend(["", "## Best By Dataset", ""])
    for dataset, row in comparison["best_by_dataset"].items():
        lines.append(
            f"- `{dataset}`: `{row['method']}` by macro-F1 "
            f"({_format_metric(row['macro_f1'])})"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TF-IDF proves the classical lexical baseline and often remains strong on small samples.",
            "- Truncated transformers show what happens when only the first fixed window is visible.",
            "- Chunked transformers make the model structurally long-document aware by aggregating chunk predictions.",
            "- Summary-first classification tests compression before classification, but can lose discriminative details.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_settings(method_name: str, metrics: dict[str, Any]) -> str:
    if method_name.startswith("baseline_"):
        return "TF-IDF + Logistic Regression"
    if method_name.startswith("truncated_transformer_"):
        return (
            f"model={metrics.get('model_name', 'n/a')}, "
            f"max_length={metrics.get('max_length', 'n/a')}"
        )
    if method_name.startswith("chunked_transformer_"):
        return (
            f"model={metrics.get('model_name', 'n/a')}, "
            f"chunk_size={metrics.get('chunk_size', 'n/a')}, "
            f"overlap={metrics.get('chunk_overlap', 'n/a')}, "
            f"cap={metrics.get('max_chunks_per_doc', 'n/a')}, "
            f"aggregation={metrics.get('aggregation', 'n/a')}"
        )
    if method_name.startswith("summary_classifier_"):
        classifier_model = metrics.get("classifier_model") or "n/a"
        return (
            f"summarizer={metrics.get('summarizer_model', 'n/a')}, "
            f"classifier={metrics.get('classifier', 'n/a')}, "
            f"classifier_model={classifier_model}"
        )
    return "n/a"


def infer_method_family(method_name: str) -> str:
    if method_name.startswith("baseline_"):
        return "tfidf_baseline"
    if method_name.startswith("truncated_transformer_"):
        return "truncated_transformer"
    if method_name.startswith("chunked_transformer_"):
        return "chunked_transformer"
    if method_name.startswith("summary_classifier_"):
        return "summary_classifier"
    return "unknown"


def infer_dataset_name(method_name: str) -> str:
    for prefix in ("baseline_", "truncated_transformer_", "chunked_transformer_", "summary_classifier_"):
        if method_name.startswith(prefix):
            return method_name.removeprefix(prefix)
    return "unknown"


def print_summary(comparison: dict[str, Any]) -> None:
    print(f"Compared {len(comparison['rows'])} report(s)")
    for dataset, row in comparison["best_by_dataset"].items():
        print(f"Best {dataset}: {row['method']} (macro-F1 {row['macro_f1']:.4f})")


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
