from longdoc_transformer_classifier.evaluation.metrics import compute_classification_metrics


def test_metrics_dictionary_structure():
    metrics = compute_classification_metrics(
        y_true=[0, 1, 1, 2],
        y_pred=[0, 1, 2, 2],
        label_names=["World", "Sports", "Business"],
    )

    assert set(metrics) == {
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "confusion_matrix",
        "labels",
        "label_names",
    }
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["macro_f1"], float)
    assert set(metrics["per_class_f1"]) == {"World", "Sports", "Business"}
    assert metrics["labels"] == [0, 1, 2]
    assert metrics["label_names"] == ["World", "Sports", "Business"]
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
