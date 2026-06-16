from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longdoc_transformer_classifier.benchmark_config import (
    METHOD_FAMILY_LABELS,
    METHOD_LIMITATIONS,
    METHOD_STRUCTURAL_TAKEAWAYS,
    REPORT_METHOD_ORDER,
)

METHOD_ORDER = REPORT_METHOD_ORDER


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
        path = (
            reports_dir / f"{method_name}.json"
            if method_name.startswith("tfidf_sweep_")
            else reports_dir / f"{method_name}_metrics.json"
        )
        if path.exists():
            rows.append(parse_metrics_file(path))

    known_paths = {reports_dir / f"{method_name}_metrics.json" for method_name in METHOD_ORDER}
    known_paths.update(reports_dir / f"{method_name}.json" for method_name in METHOD_ORDER)
    for path in sorted(reports_dir.glob("*_metrics.json")):
        if path not in known_paths and not path.name.startswith("model_comparison"):
            rows.append(parse_metrics_file(path))
    for path in sorted(reports_dir.glob("tfidf_sweep_*.json")):
        if path not in known_paths:
            rows.append(parse_metrics_file(path))
    return rows


def parse_metrics_file(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metadata = metrics.get("metadata") if isinstance(metrics.get("metadata"), dict) else {}
    path_method_name = path.name.removesuffix("_metrics.json").removesuffix(".json")
    method_name = str(_first_present(metadata.get("method"), path_method_name))
    method_family = infer_method_family(method_name)
    accuracy = _first_present(metadata.get("accuracy"), metrics.get("accuracy"))
    macro_f1 = _first_present(metadata.get("macro_f1"), metrics.get("macro_f1"))
    if method_name.startswith("tfidf_sweep_") and metrics.get("best_result"):
        accuracy = metrics["best_result"].get("accuracy")
        macro_f1 = metrics["best_result"].get("macro_f1")
    return {
        "method": method_name,
        "method_family": method_family,
        "method_family_label": METHOD_FAMILY_LABELS.get(method_family, method_family),
        "dataset": str(
            _first_present(
                metadata.get("dataset"),
                metrics.get("dataset_name"),
                infer_dataset_name(method_name),
            )
        ),
        "accuracy": _optional_float(accuracy),
        "macro_f1": _optional_float(macro_f1),
        "train_samples": _optional_int(
            _first_present(
                metadata.get("max_train_samples"),
                metrics.get("max_train_samples"),
                metrics.get("train_size"),
            )
        ),
        "test_samples": _optional_int(
            _first_present(
                metadata.get("max_test_samples"),
                metrics.get("max_test_samples"),
                metrics.get("test_size"),
            )
        ),
        "model_name": infer_model_name(method_name, metrics, metadata),
        "key_settings": summarize_settings(method_name, metrics),
        "key_limitation": summarize_limitation(method_family, metrics, metadata),
        "structural_takeaway": METHOD_STRUCTURAL_TAKEAWAYS.get(
            method_family,
            METHOD_STRUCTURAL_TAKEAWAYS["unknown"],
        ),
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

    lines.extend(
        [
            "",
            "## Benchmark View",
            "",
            "| Method | Family | Dataset | Accuracy | Macro-F1 | Train Samples | "
            "Test Samples | Model | Key Limitation | Structural Takeaway |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in comparison["rows"]:
        lines.append(
            f"| `{row['method']}` | {_clean_cell(row['method_family_label'])} | "
            f"`{row['dataset']}` | {_format_metric(row['accuracy'])} | "
            f"{_format_metric(row['macro_f1'])} | {_format_int(row['train_samples'])} | "
            f"{_format_int(row['test_samples'])} | {_clean_cell(row['model_name'])} | "
            f"{_clean_cell(row['key_limitation'])} | "
            f"{_clean_cell(row['structural_takeaway'])} |"
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
        tfidf_config = metrics.get("tfidf_config") if isinstance(metrics.get("tfidf_config"), dict) else {}
        model_config = metrics.get("model_config") if isinstance(metrics.get("model_config"), dict) else {}
        if tfidf_config or model_config:
            return (
                "TF-IDF + Logistic Regression, "
                f"max_features={tfidf_config.get('max_features', 'n/a')}, "
                f"ngram={tfidf_config.get('ngram_range', 'n/a')}, "
                f"min_df={tfidf_config.get('min_df', 'n/a')}, "
                f"max_df={tfidf_config.get('max_df', 'n/a')}, "
                f"sublinear_tf={tfidf_config.get('sublinear_tf', 'n/a')}, "
                f"class_weight={model_config.get('class_weight', 'none')}"
            )
        return "TF-IDF + Logistic Regression"
    if method_name.startswith("tfidf_sweep_"):
        best = metrics.get("best_result", {})
        return f"best_setting={best.get('setting', 'n/a')}"
    if method_name.startswith("truncated_transformer_"):
        return (
            f"model={metrics.get('model_name', 'n/a')}, "
            f"max_length={metrics.get('max_length', 'n/a')}"
        )
    if method_name.startswith("chunked_transformer_"):
        return (
            f"model={metrics.get('model_name', 'n/a')}, "
            f"chunk_selection={metrics.get('chunk_selection', 'first_k')}, "
            f"aggregation={metrics.get('aggregation', 'n/a')}, "
            f"max_chunks_per_doc={metrics.get('max_chunks_per_doc', 'n/a')}, "
            f"chunk_size={metrics.get('chunk_size', 'n/a')}, "
            f"overlap={metrics.get('chunk_overlap', 'n/a')}"
        )
    if method_name.startswith("long_context_transformer_"):
        return (
            f"model={metrics.get('model_name', 'n/a')}, "
            f"max_length={metrics.get('max_length', 'n/a')}, "
            f"freeze_encoder={metrics.get('freeze_encoder', 'n/a')}, "
            f"trainable_params={metrics.get('trainable_parameter_count', 'n/a')}, "
            f"gradient_accumulation={metrics.get('gradient_accumulation_steps', 'n/a')}"
        )
    if method_name.startswith("summary_classifier_"):
        classifier_model = metrics.get("classifier_model") or "n/a"
        return (
            f"summarizer={metrics.get('summarizer_model', 'n/a')}, "
            f"classifier={metrics.get('classifier', 'n/a')}, "
            f"classifier_model={classifier_model}"
        )
    return "n/a"


def infer_model_name(
    method_name: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    metadata_model = metadata.get("model_name")
    if metadata_model:
        return str(metadata_model)
    if metrics.get("model_name"):
        return str(metrics["model_name"])
    if method_name.startswith("baseline_"):
        return "TF-IDF + Logistic Regression"
    if method_name.startswith("summary_classifier_"):
        summarizer = metrics.get("summarizer_model")
        classifier = metrics.get("classifier")
        classifier_model = metrics.get("classifier_model")
        if classifier_model:
            return f"{summarizer} + {classifier_model}"
        if summarizer and classifier:
            return f"{summarizer} + {classifier}"
        if summarizer:
            return str(summarizer)
    return "n/a"


def summarize_limitation(
    method_family: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    limitations = metadata.get("limitations") or metrics.get("limitations")
    if isinstance(limitations, list) and limitations:
        return str(limitations[0])
    if isinstance(limitations, str) and limitations:
        return limitations
    if metrics.get("limitation_note"):
        return str(metrics["limitation_note"])
    if metrics.get("truncation_note"):
        return str(metrics["truncation_note"])
    return METHOD_LIMITATIONS.get(method_family, METHOD_LIMITATIONS["unknown"])


def infer_method_family(method_name: str) -> str:
    if method_name.startswith("baseline_"):
        return "tfidf_baseline"
    if method_name.startswith("tfidf_sweep_"):
        return "tfidf_sweep"
    if method_name.startswith("truncated_transformer_"):
        return "truncated_transformer"
    if method_name.startswith("chunked_transformer_"):
        return "chunked_transformer"
    if method_name.startswith("long_context_transformer_"):
        return "long_context_transformer"
    if method_name.startswith("summary_classifier_"):
        return "summary_classifier"
    return "unknown"


def infer_dataset_name(method_name: str) -> str:
    for prefix in (
        "baseline_",
        "tfidf_sweep_",
        "truncated_transformer_",
        "chunked_transformer_",
        "long_context_transformer_",
        "summary_classifier_",
    ):
        if method_name.startswith(prefix):
            return method_name.removeprefix(prefix)
    return "unknown"


def print_summary(comparison: dict[str, Any]) -> None:
    print(f"Compared {len(comparison['rows'])} report(s)")
    for dataset, row in comparison["best_by_dataset"].items():
        print(f"Best {dataset}: {row['method']} (macro-F1 {row['macro_f1']:.4f})")


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _clean_cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
