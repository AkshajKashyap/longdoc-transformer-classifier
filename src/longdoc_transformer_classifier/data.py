from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from longdoc_transformer_classifier.config import DatasetConfig

AG_NEWS_DATASET_NAME = "ag_news"
AG_NEWS_HF_PATH = "fancyzhx/ag_news"


@dataclass(frozen=True)
class TextClassificationDataset:
    name: str
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]


def load_text_classification_dataset(config: DatasetConfig) -> TextClassificationDataset:
    if config.name != AG_NEWS_DATASET_NAME:
        msg = f"Unsupported dataset '{config.name}'. Only 'ag_news' is supported today."
        raise ValueError(msg)

    return load_ag_news(
        max_train_samples=config.max_train_samples,
        max_test_samples=config.max_test_samples,
        text_column=config.text_column,
        label_column=config.label_column,
    )


def load_ag_news(
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> TextClassificationDataset:
    dataset = load_dataset(AG_NEWS_HF_PATH)
    label_names = _extract_label_names(dataset, label_column=label_column)

    train_split = _limit_split_by_label(
        dataset["train"], max_train_samples, label_column=label_column
    )
    test_split = _limit_split_by_label(dataset["test"], max_test_samples, label_column=label_column)

    train_texts, train_labels = _split_to_lists(train_split, text_column, label_column)
    test_texts, test_labels = _split_to_lists(test_split, text_column, label_column)

    return TextClassificationDataset(
        name=AG_NEWS_DATASET_NAME,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
    )


def _limit_split_by_label(split: Any, max_samples: int | None, label_column: str) -> Any:
    if max_samples is None or max_samples >= len(split):
        return split
    if max_samples < 0:
        msg = "max_samples must be non-negative."
        raise ValueError(msg)
    if max_samples == 0:
        return split.select([])

    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(split[label_column]):
        label_to_indices[int(label)].append(index)

    ordered_labels = sorted(label_to_indices)
    if not ordered_labels:
        return split.select([])

    base_quota, remainder = divmod(max_samples, len(ordered_labels))
    selected_indices: list[int] = []
    for label_position, label in enumerate(ordered_labels):
        quota = base_quota + int(label_position < remainder)
        selected_indices.extend(label_to_indices[label][:quota])

    return split.select(sorted(selected_indices))


def _extract_label_names(dataset: Any, label_column: str) -> list[str]:
    label_feature = dataset["train"].features.get(label_column)
    label_names = getattr(label_feature, "names", None)
    if label_names:
        return [str(name) for name in label_names]

    labels: set[int] = set()
    for split_name in ("train", "test"):
        labels.update(int(label) for label in dataset[split_name][label_column])
    return [str(label) for label in sorted(labels)]


def _split_to_lists(split: Any, text_column: str, label_column: str) -> tuple[list[str], list[int]]:
    texts = [str(text) for text in split[text_column]]
    labels = [int(label) for label in split[label_column]]
    return texts, labels
