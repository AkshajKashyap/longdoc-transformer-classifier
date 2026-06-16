from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)


def load_tokenizer(model_name: str, cache_dir: str | None = None) -> Any:
    if _is_plain_bert_model_name(model_name):
        return _load_bert_tokenizer(model_name, cache_dir)
    try:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    except ValueError:
        if _is_plain_bert_model_name(model_name):
            return _load_bert_tokenizer(model_name, cache_dir)
        raise


def load_sequence_classifier(
    model_name: str,
    num_labels: int,
    cache_dir: str | None = None,
) -> torch.nn.Module:
    if _is_plain_bert_model_name(model_name):
        return _load_bert_sequence_classifier(model_name, num_labels, cache_dir)
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )
    except ValueError:
        if _is_plain_bert_model_name(model_name):
            return _load_bert_sequence_classifier(model_name, num_labels, cache_dir)
        raise


def tokenize_texts(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    if max_length <= 0:
        msg = "max_length must be greater than 0."
        raise ValueError(msg)

    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": _as_tensor(encoded["input_ids"]),
        "attention_mask": _as_tensor(encoded["attention_mask"]),
    }


def _load_bert_sequence_classifier(
    model_name: str,
    num_labels: int,
    cache_dir: str | None,
) -> BertForSequenceClassification:
    try:
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return BertForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    except OSError:
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )
        return BertForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )


def _load_bert_tokenizer(model_name: str, cache_dir: str | None) -> BertTokenizer:
    try:
        return BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except OSError:
        return BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def _is_plain_bert_model_name(model_name: str) -> bool:
    lower_name = model_name.lower()
    return "bert" in lower_name and "distilbert" not in lower_name


class TruncatedTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            msg = "texts and labels must have the same length."
            raise ValueError(msg)
        self.encodings = tokenize_texts(texts, tokenizer, max_length)
        self.labels = torch.tensor([int(label) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)
