import pytest

from longdoc_transformer_classifier.training.sweep_tfidf_baseline import select_best_result


def test_tfidf_sweep_result_selection_uses_macro_f1_then_accuracy():
    results = [
        {"setting": {"max_features": 1000}, "accuracy": 0.8, "macro_f1": 0.7},
        {"setting": {"max_features": 2000}, "accuracy": 0.75, "macro_f1": 0.72},
        {"setting": {"max_features": 3000}, "accuracy": 0.9, "macro_f1": 0.72},
    ]

    best = select_best_result(results)

    assert best["setting"]["max_features"] == 3000


def test_tfidf_sweep_result_selection_rejects_empty_results():
    with pytest.raises(ValueError, match="empty sweep"):
        select_best_result([])

