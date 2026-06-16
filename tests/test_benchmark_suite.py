from pathlib import Path

from longdoc_transformer_classifier.training.run_benchmark_suite import build_command_plan


def test_quick_benchmark_plan_skips_expensive_models_by_default():
    plan = build_command_plan(
        dataset="arxiv",
        quick=True,
        include_transformers=False,
        include_long_context=False,
        include_summary=False,
        output_dir=Path("reports"),
        chunk_selection="first_k",
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
        include_long_context=False,
        include_summary=True,
        output_dir=Path("reports"),
        chunk_selection="uniform_k",
    )

    modules = [command.module for command in plan]

    assert "longdoc_transformer_classifier.training.train_truncated_transformer" in modules
    assert "longdoc_transformer_classifier.training.train_chunked_transformer" in modules
    assert "longdoc_transformer_classifier.training.train_summary_classifier" in modules
    chunked_command = plan[3]
    assert chunked_command.args[chunked_command.args.index("--chunk-selection") + 1] == "uniform_k"


def test_quick_benchmark_plan_includes_long_context_only_when_requested():
    default_plan = build_command_plan(
        dataset="arxiv",
        quick=True,
        include_transformers=False,
        include_long_context=False,
        include_summary=False,
        output_dir=Path("reports"),
    )
    long_context_plan = build_command_plan(
        dataset="arxiv",
        quick=True,
        include_transformers=False,
        include_long_context=True,
        include_summary=False,
        output_dir=Path("reports"),
    )

    default_modules = [command.module for command in default_plan]
    long_context_modules = [command.module for command in long_context_plan]

    assert "longdoc_transformer_classifier.training.train_long_context_transformer" not in default_modules
    assert "longdoc_transformer_classifier.training.train_long_context_transformer" in long_context_modules
    command = long_context_plan[2]
    assert command.args[command.args.index("--batch-size") + 1] == "1"
    assert "--freeze-encoder" in command.args
