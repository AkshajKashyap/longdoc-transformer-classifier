from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.linear_model import LogisticRegression

from longdoc_transformer_classifier.config import BaselineModelConfig


def create_classifier(config: BaselineModelConfig | None = None) -> LogisticRegression:
    config = config or BaselineModelConfig()
    return LogisticRegression(
        max_iter=config.max_iter,
        random_state=config.random_state,
        solver=config.solver,
        class_weight=config.class_weight,
    )


def train(
    features: Any,
    labels: Sequence[int],
    config: BaselineModelConfig | None = None,
) -> LogisticRegression:
    classifier = create_classifier(config)
    classifier.fit(features, labels)
    return classifier


def predict(classifier: LogisticRegression, features: Any) -> list[int]:
    return [int(label) for label in classifier.predict(features)]


def predict_proba(classifier: LogisticRegression, features: Any) -> list[list[float]]:
    probabilities = classifier.predict_proba(features)
    return [[float(value) for value in row] for row in probabilities]
