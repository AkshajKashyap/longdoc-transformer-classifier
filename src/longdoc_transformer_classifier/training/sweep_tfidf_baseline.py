from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longdoc_transformer_classifier.config import BaselineModelConfig, DatasetConfig, TfidfConfig
from longdoc_transformer_classifier.data import load_text_classification_dataset
from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics
from longdoc_transformer_classifier.features import fit_tfidf, transform_tfidf
from longdoc_transformer_classifier.models.baseline import predict, train

DEFAULT_SWEEP_SETTINGS = [
    (50_000, (1, 1), False, None),
    (50_000, (1, 2), False, None),
    (100_000, (1, 2), True, None),
    (100_000, (1, 2), True, "balanced"),
]


@dataclass(frozen=True)
class SweepSetting:
    max_features: int
    ngram_range: tuple[int, int]
    sublinear_tf: bool
    class_weight: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "sublinear_tf": self.sublinear_tf,
            "class_weight": self.class_weight,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small TF-IDF baseline sweep.")
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--max-train-samples", type=int, default=3_000)
    parser.add_argument("--max-test-samples", type=int, default=1_000)
    parser.add_argument("--min-df", type=_parse_int_or_float, default=2)
    parser.add_argument("--max-df", type=_parse_int_or_float, default=0.95)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print_summary(report)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_text_classification_dataset(
        DatasetConfig(
            name=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )
    settings = build_default_grid()
    results = []
    for index, setting in enumerate(settings, start=1):
        print(f"Running sweep setting {index}/{len(settings)}: {setting.to_dict()}", flush=True)
        result = evaluate_setting(
            setting=setting,
            train_texts=dataset.train_texts,
            train_labels=dataset.train_labels,
            test_texts=dataset.test_texts,
            test_labels=dataset.test_labels,
            label_names=dataset.label_names,
            min_df=args.min_df,
            max_df=args.max_df,
            solver=args.solver,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        results.append(result)

    best_result = select_best_result(results)
    report = {
        "method": f"tfidf_sweep_{dataset.dataset_name}",
        "dataset": dataset.dataset_name,
        "dataset_name": dataset.dataset_name,
        "hf_path": dataset.hf_path,
        "text_field": dataset.text_field,
        "label_field": dataset.label_field,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "train_size": len(dataset.train_texts),
        "test_size": len(dataset.test_texts),
        "min_df": args.min_df,
        "max_df": args.max_df,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "results": results,
        "best_result": best_result,
        "warning": "This is still a classical lexical baseline, not neural long-context reasoning.",
    }
    write_reports(report, args.reports_dir)
    return report


def build_default_grid() -> list[SweepSetting]:
    return [
        SweepSetting(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            class_weight=class_weight,
        )
        for max_features, ngram_range, sublinear_tf, class_weight in DEFAULT_SWEEP_SETTINGS
    ]


def evaluate_setting(
    setting: SweepSetting,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    label_names: list[str],
    min_df: int | float,
    max_df: int | float,
    solver: str,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    feature_config = TfidfConfig(
        max_features=setting.max_features,
        ngram_range=setting.ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=setting.sublinear_tf,
    )
    model_config = BaselineModelConfig(
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
        class_weight=setting.class_weight,
    )
    vectorizer, train_features = fit_tfidf(train_texts, feature_config)
    test_features = transform_tfidf(vectorizer, test_texts)
    classifier = train(train_features, train_labels, model_config)
    predictions = predict(classifier, test_features)
    metrics = compute_classification_metrics(test_labels, predictions, label_names=label_names)
    return {
        "setting": setting.to_dict(),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        msg = "Cannot select a best result from an empty sweep."
        raise ValueError(msg)
    return max(
        results,
        key=lambda result: (
            float(result["macro_f1"]),
            float(result["accuracy"]),
            -int(result["setting"]["max_features"]),
        ),
    )


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = report["dataset_name"]
    json_path = reports_dir / f"tfidf_sweep_{dataset_name}.json"
    markdown_path = reports_dir / f"tfidf_sweep_{dataset_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def build_markdown_report(report: dict[str, Any]) -> str:
    best = report["best_result"]
    lines = [
        f"# {report['dataset_name']} TF-IDF Sweep",
        "",
        "> This is still a classical lexical baseline, not neural long-context reasoning.",
        "",
        "## Dataset",
        "",
        f"- Dataset name: `{report['dataset_name']}`",
        f"- Hugging Face dataset: `{report['hf_path']}`",
        f"- Train samples: {report['train_size']}",
        f"- Test samples: {report['test_size']}",
        "",
        "## Fixed Settings",
        "",
        f"- Min document frequency: {report['min_df']}",
        f"- Max document frequency: {report['max_df']}",
        f"- Solver: `{report['solver']}`",
        f"- Max iterations: {report['max_iter']}",
        "",
        "## Best Setting By Macro-F1",
        "",
        f"- Accuracy: {best['accuracy']:.4f}",
        f"- Macro-F1: {best['macro_f1']:.4f}",
        f"- Settings: `{best['setting']}`",
        "",
        "## Results",
        "",
        "| Max Features | N-gram Range | Sublinear TF | Class Weight | Accuracy | Macro-F1 |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    for result in report["results"]:
        setting = result["setting"]
        lines.append(
            f"| {setting['max_features']} | {setting['ngram_range'][0]}-{setting['ngram_range'][1]} | "
            f"{setting['sublinear_tf']} | {setting['class_weight'] or 'none'} | "
            f"{result['accuracy']:.4f} | {result['macro_f1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(report: dict[str, Any]) -> None:
    best = report["best_result"]
    print(f"TF-IDF sweep on {report['dataset_name']}")
    print(f"Tried {len(report['results'])} setting(s)")
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print(f"Best macro-F1: {best['macro_f1']:.4f}")
    print(f"Best setting: {best['setting']}")


def _parse_int_or_float(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
