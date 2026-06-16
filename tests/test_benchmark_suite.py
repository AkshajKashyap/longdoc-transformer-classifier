from pathlib import Path

from longdoc_transformer_classifier.training.run_benchmark_suite import build_command_plan


def test_quick_benchmark_plan_skips_expensive_models_by_default():
    plan = build_command_plan(
        dataset="arxiv",
        quick=True,
        include_transformers=False,
        include_summary=False,
        output_dir=Path("reports"),
    )

    modules = [command.module for command in plan]

    assert modules == [
        "longdoc_transformer_classifier.training.analyze_dataset",
        "longdoc_transformer_classifier.training.train_baseline",
        "longdoc_transformer_classifier.training.compare_reports",
    ]
    assert "longdoc_transformer_classifier.training.train_summary_classifier" not in modules
    assert plan[0].args[plan[0].args.index("--max-train-samples") + 1] == "100"
    assert plan[1].args[plan[1].args.index("--max-test-samples") + 1] == "50"


def test_quick_benchmark_plan_includes_optional_transformers_and_summary():
    plan = build_command_plan(
        dataset="arxiv",
        quick=True,
        include_transformers=True,
        include_summary=True,
        output_dir=Path("reports"),
    )

    modules = [command.module for command in plan]

    assert "longdoc_transformer_classifier.training.train_truncated_transformer" in modules
    assert "longdoc_transformer_classifier.training.train_chunked_transformer" in modules
    assert "longdoc_transformer_classifier.training.train_summary_classifier" in modules

