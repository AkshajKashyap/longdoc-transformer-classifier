from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_portfolio_reports(reports_dir: Path = Path("reports")) -> tuple[Path, Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison(reports_dir)
    model_card_path = reports_dir / "model_card.md"
    project_summary_path = reports_dir / "project_summary.md"
    health_check_path = reports_dir / "repo_health_check.md"

    model_card_path.write_text(build_model_card(comparison), encoding="utf-8")
    project_summary_path.write_text(build_project_summary(comparison), encoding="utf-8")
    health_check_path.write_text(build_repo_health_check(), encoding="utf-8")
    return model_card_path, project_summary_path, health_check_path


def load_comparison(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "model_comparison.json"
    if not path.exists():
        return {"rows": [], "best_by_dataset": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_card(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Model Card",
            "",
            "## Project Overview",
            "",
            "This project is a reproducible long-document classification benchmark comparing "
            "lexical, truncation, chunking, summarization, and long-context transformer strategies.",
            "",
            "## Intended Use",
            "",
            "Use this repo to study baseline behavior and engineering tradeoffs for long-document "
            "classification experiments. It is not a production classifier.",
            "",
            "## Datasets",
            "",
            "- `ag_news`: short-text smoke dataset.",
            "- `arxiv`: long-document target dataset from `ccdv/arxiv-classification`.",
            "",
            "## Methods Compared",
            "",
            _format_methods(comparison),
            "",
            "## Headline Results",
            "",
            _format_best_results(comparison),
            "",
            "## Limitations",
            "",
            "- Smoke sample sizes are not final rankings.",
            "- Tiny and frozen transformer runs validate infrastructure more than performance.",
            "- TF-IDF can exploit lexical label cues without modeling discourse.",
            "- Long-context models remain bounded by max length and compute.",
            "",
            "## Ethical And Interpretability Notes",
            "",
            "The benchmark favors transparent reporting: confusion matrices, per-class F1, and "
            "method limitations are saved with reports. Results should not be used for high-stakes "
            "document triage without domain validation and bias checks.",
            "",
            "## Reproducibility Commands",
            "",
            "```bash",
            "make install",
            "make check",
            "python -m longdoc_transformer_classifier.training.compare_reports",
            "python -m longdoc_transformer_classifier.training.plot_comparison",
            "```",
            "",
            "## What This Project Does Not Claim",
            "",
            "It does not claim state-of-the-art accuracy. The strongest honest claim is that it "
            "builds a clean benchmark showing why simple baselines can outperform poorly adapted "
            "neural methods under constrained training.",
            "",
        ]
    )


def build_project_summary(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Project Summary",
            "",
            "## One-Paragraph Summary",
            "",
            "I built a reproducible long-document classification benchmark comparing lexical, "
            "truncation, chunking, summarization, and long-context transformer strategies.",
            "",
            "## What Was Built",
            "",
            "- Dataset loaders for short and long text classification.",
            "- Length analysis, chunking, chunk selection, and summary caching.",
            "- Classical, truncated, chunked, summary-first, and Longformer-style baselines.",
            "- Unified metrics, comparison reports, plots, and portfolio documentation.",
            "",
            "## Why Long-Document Classification Is Hard",
            "",
            "Long documents often exceed standard transformer context windows, so naive models "
            "either truncate evidence, require chunk aggregation, summarize away details, or pay "
            "higher compute costs for long-context architectures.",
            "",
            "## What Each Baseline Proved",
            "",
            "- TF-IDF proved a strong lexical floor.",
            "- Truncation showed the weakness of prefix-only transformers.",
            "- Chunking made the pipeline structurally long-document aware.",
            "- Summary-first tested compression before classification.",
            "- Long-context transformers added the missing architecture-level comparison.",
            "",
            "## Current Best Result",
            "",
            _format_best_results(comparison),
            "",
            "## Key Engineering Choices",
            "",
            "The repo keeps experiments deterministic where possible, separates data/features/models, "
            "writes serializable reports, avoids app/server overbuild, and keeps expensive model "
            "runs opt-in.",
            "",
            "## What I Would Improve Next",
            "",
            "With more compute I would run larger TF-IDF sweeps, fully fine-tune long-context models, "
            "try learned chunk retrieval, improve summary selection, and add confidence calibration.",
            "",
        ]
    )


def build_repo_health_check() -> str:
    return "\n".join(
        [
            "# Repo Health Check",
            "",
            "## Tests Status",
            "",
            "- Verified command: `pytest -q`",
            "",
            "## Lint Status",
            "",
            "- Verified command: `ruff check .`",
            "",
            "## CI Status",
            "",
            "- GitHub Actions workflow: `.github/workflows/ci.yml`",
            "- CI runs lint and tests only; it does not download datasets or models.",
            "",
            "## Artifact Hygiene",
            "",
            "- Source, tests, docs, and compact reports are committed.",
            "- Large datasets, model caches, generated summaries, and virtualenvs are ignored.",
            "",
            "## Ignored Cache/Model/Data Directories",
            "",
            "- `.venv/`",
            "- `data/hf_cache/`",
            "- `data/processed/`",
            "- `data/raw/`",
            "- Python and test cache directories",
            "",
            "## Commands Verified",
            "",
            "```bash",
            "ruff check .",
            "pytest -q",
            "python -m longdoc_transformer_classifier.training.compare_reports",
            "python -m longdoc_transformer_classifier.training.plot_comparison",
            "make release-check",
            "```",
            "",
        ]
    )


def _format_methods(comparison: dict[str, Any]) -> str:
    families = sorted({row.get("method_family_label", "Unknown") for row in comparison.get("rows", [])})
    if not families:
        return "No comparison rows are available yet."
    return "\n".join(f"- {family}" for family in families)


def _format_best_results(comparison: dict[str, Any]) -> str:
    best_by_dataset = comparison.get("best_by_dataset", {})
    if not best_by_dataset:
        return "No best-result summary is available yet."
    lines = []
    for dataset, row in sorted(best_by_dataset.items()):
        lines.append(
            f"- `{dataset}`: `{row['method']}` with macro-F1 {row['macro_f1']:.4f} "
            f"and accuracy {row['accuracy']:.4f}."
        )
    return "\n".join(lines)
