from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    true_labels = [int(label) for label in y_true]
    predicted_labels = [int(label) for label in y_pred]
    if len(true_labels) != len(predicted_labels):
        msg = "y_true and y_pred must have the same length."
        raise ValueError(msg)
    if not true_labels:
        msg = "Cannot compute metrics for an empty label set."
        raise ValueError(msg)

    labels = _resolve_labels(true_labels, predicted_labels, label_names)
    resolved_label_names = _resolve_label_names(labels, label_names)
    per_class_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": {
            label_name: float(score)
            for label_name, score in zip(resolved_label_names, per_class_f1, strict=True)
        },
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
        "labels": [int(label) for label in labels],
        "label_names": resolved_label_names,
    }


def _resolve_labels(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    label_names: Sequence[str] | None,
) -> list[int]:
    if label_names is not None:
        return list(range(len(label_names)))
    return sorted(set(true_labels) | set(predicted_labels))


def _resolve_label_names(labels: Sequence[int], label_names: Sequence[str] | None) -> list[str]:
    if label_names is not None:
        return [str(label_name) for label_name in label_names]
    return [str(label) for label in labels]
