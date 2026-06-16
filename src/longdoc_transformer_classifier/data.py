from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from longdoc_transformer_classifier.config import DatasetConfig

AG_NEWS_DATASET_NAME = "ag_news"
AG_NEWS_HF_PATH = "fancyzhx/ag_news"
ARXIV_DATASET_NAME = "arxiv"
ARXIV_HF_PATH = "ccdv/arxiv-classification"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    hf_path: str
    train_split: str
    test_split: str
    text_field: str
    label_field: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    AG_NEWS_DATASET_NAME: DatasetSpec(
        dataset_name=AG_NEWS_DATASET_NAME,
        hf_path=AG_NEWS_HF_PATH,
        train_split="train",
        test_split="test",
        text_field="text",
        label_field="label",
    ),
    ARXIV_DATASET_NAME: DatasetSpec(
        dataset_name=ARXIV_DATASET_NAME,
        hf_path=ARXIV_HF_PATH,
        train_split="train",
        test_split="test",
        text_field="text",
        label_field="label",
    ),
}


@dataclass(frozen=True)
class TextClassificationDataset:
    dataset_name: str
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]
    text_field: str
    label_field: str
    hf_path: str

    @property
    def name(self) -> str:
        return self.dataset_name


def load_text_classification_dataset(config: DatasetConfig) -> TextClassificationDataset:
    dataset_name = normalize_dataset_name(config.name)
    if dataset_name not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        msg = f"Unsupported dataset '{config.name}'. Supported datasets: {supported}."
        raise ValueError(msg)

    spec = DATASET_SPECS[dataset_name]
    if spec.dataset_name == AG_NEWS_DATASET_NAME:
        return load_ag_news(
            max_train_samples=config.max_train_samples,
            max_test_samples=config.max_test_samples,
            cache_dir=config.cache_dir,
            text_field=config.text_field or spec.text_field,
            label_field=config.label_field or spec.label_field,
        )
    if spec.dataset_name == ARXIV_DATASET_NAME:
        return load_arxiv_classification(
            max_train_samples=config.max_train_samples,
            max_test_samples=config.max_test_samples,
            cache_dir=config.cache_dir,
            text_field=config.text_field or spec.text_field,
            label_field=config.label_field or spec.label_field,
        )

    msg = f"No loader registered for dataset '{config.name}'."
    raise ValueError(msg)


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized in {AG_NEWS_HF_PATH, "fancyzhx_ag_news"}:
        return AG_NEWS_DATASET_NAME
    if normalized in {ARXIV_HF_PATH, "ccdv_arxiv_classification", "arxiv-classification"}:
        return ARXIV_DATASET_NAME
    return normalized


def load_ag_news(
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    cache_dir: str | Path | None = Path("data/hf_cache"),
    text_field: str = "text",
    label_field: str = "label",
) -> TextClassificationDataset:
    return _load_hf_text_classification_dataset(
        spec=DATASET_SPECS[AG_NEWS_DATASET_NAME],
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        cache_dir=cache_dir,
        text_field=text_field,
        label_field=label_field,
    )


def load_arxiv_classification(
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    cache_dir: str | Path | None = Path("data/hf_cache"),
    text_field: str = "text",
    label_field: str = "label",
) -> TextClassificationDataset:
    return _load_hf_text_classification_dataset(
        spec=DATASET_SPECS[ARXIV_DATASET_NAME],
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        cache_dir=cache_dir,
        text_field=text_field,
        label_field=label_field,
    )


def _load_hf_text_classification_dataset(
    spec: DatasetSpec,
    max_train_samples: int | None,
    max_test_samples: int | None,
    cache_dir: str | Path | None,
    text_field: str,
    label_field: str,
) -> TextClassificationDataset:
    dataset = load_dataset(spec.hf_path, cache_dir=str(cache_dir) if cache_dir else None)
    label_names = _extract_label_names(dataset, split_name=spec.train_split, label_field=label_field)

    train_split = _limit_split_by_label(
        dataset[spec.train_split],
        max_train_samples,
        label_field=label_field,
        label_names=label_names,
    )
    test_split = _limit_split_by_label(
        dataset[spec.test_split],
        max_test_samples,
        label_field=label_field,
        label_names=label_names,
    )

    train_texts, train_labels = _split_to_lists(train_split, text_field, label_field, label_names)
    test_texts, test_labels = _split_to_lists(test_split, text_field, label_field, label_names)

    return TextClassificationDataset(
        dataset_name=spec.dataset_name,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
        text_field=text_field,
        label_field=label_field,
        hf_path=spec.hf_path,
    )


def _limit_split_by_label(
    split: Any,
    max_samples: int | None,
    label_field: str,
    label_names: list[str],
) -> Any:
    if max_samples is None or max_samples >= len(split):
        return split
    if max_samples < 0:
        msg = "max_samples must be non-negative."
        raise ValueError(msg)
    if max_samples == 0:
        return split.select([])

    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(split[label_field]):
        label_to_indices[_coerce_label(label, label_names)].append(index)

    ordered_labels = sorted(label_to_indices)
    if not ordered_labels:
        return split.select([])

    base_quota, remainder = divmod(max_samples, len(ordered_labels))
    selected_indices: list[int] = []
    for label_position, label in enumerate(ordered_labels):
        quota = base_quota + int(label_position < remainder)
        selected_indices.extend(label_to_indices[label][:quota])

    return split.select(sorted(selected_indices))


def _extract_label_names(dataset: Any, split_name: str, label_field: str) -> list[str]:
    label_feature = dataset[split_name].features.get(label_field)
    label_names = getattr(label_feature, "names", None)
    if label_names:
        return [str(name) for name in label_names]

    labels: set[int] = set()
    for current_split_name in dataset:
        labels.update(_coerce_label(label, []) for label in dataset[current_split_name][label_field])
    return [str(label) for label in sorted(labels)]


def _split_to_lists(
    split: Any,
    text_field: str,
    label_field: str,
    label_names: list[str],
) -> tuple[list[str], list[int]]:
    texts = [str(text) for text in split[text_field]]
    labels = [_coerce_label(label, label_names) for label in split[label_field]]
    return texts, labels


def _coerce_label(label: Any, label_names: list[str]) -> int:
    try:
        return int(label)
    except (TypeError, ValueError):
        label_text = str(label)
        if label_text in label_names:
            return label_names.index(label_text)
        msg = f"Could not convert label '{label_text}' to an integer class id."
        raise ValueError(msg) from None
