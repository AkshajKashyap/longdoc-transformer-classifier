import json

from longdoc_transformer_classifier.visualization import plot_model_comparison


def test_plot_generation_works_on_mocked_model_comparison(tmp_path):
    comparison_path = tmp_path / "model_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "baseline_arxiv",
                        "method_family_label": "TF-IDF + Logistic Regression",
                        "dataset": "arxiv",
                        "accuracy": 0.7,
                        "macro_f1": 0.6,
                    },
                    {
                        "method": "summary_classifier_arxiv",
                        "method_family_label": "Summary-First Classifier",
                        "dataset": "arxiv",
                        "accuracy": None,
                        "macro_f1": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    macro_f1_path, accuracy_path = plot_model_comparison(comparison_path)

    assert macro_f1_path.exists()
    assert accuracy_path.exists()
    assert macro_f1_path.stat().st_size > 0
    assert accuracy_path.stat().st_size > 0
