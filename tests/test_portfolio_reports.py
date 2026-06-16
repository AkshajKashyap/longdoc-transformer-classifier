from longdoc_transformer_classifier.portfolio_reports import (
    build_model_card,
    build_project_summary,
    build_repo_health_check,
)


def test_model_card_writer_includes_required_sections():
    text = build_model_card(_comparison())

    for heading in [
        "## Project Overview",
        "## Intended Use",
        "## Datasets",
        "## Methods Compared",
        "## Headline Results",
        "## Limitations",
        "## Ethical And Interpretability Notes",
        "## Reproducibility Commands",
        "## What This Project Does Not Claim",
    ]:
        assert heading in text


def test_project_summary_writer_includes_required_sections():
    text = build_project_summary(_comparison())

    for heading in [
        "## One-Paragraph Summary",
        "## What Was Built",
        "## Why Long-Document Classification Is Hard",
        "## What Each Baseline Proved",
        "## Current Best Result",
        "## Key Engineering Choices",
        "## What I Would Improve Next",
    ]:
        assert heading in text


def test_repo_health_check_mentions_lint_tests_and_ci():
    text = build_repo_health_check()

    assert "pytest -q" in text
    assert "ruff check ." in text
    assert ".github/workflows/ci.yml" in text


def _comparison():
    return {
        "rows": [{"method_family_label": "TF-IDF + Logistic Regression"}],
        "best_by_dataset": {
            "arxiv": {
                "method": "tfidf_sweep_arxiv",
                "macro_f1": 0.8,
                "accuracy": 0.82,
            }
        },
    }
