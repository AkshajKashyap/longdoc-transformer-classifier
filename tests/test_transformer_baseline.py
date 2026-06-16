import json

import torch

from longdoc_transformer_classifier.models.transformer_baseline import (
    TruncatedTextDataset,
    tokenize_texts,
)
from longdoc_transformer_classifier.training.train_truncated_transformer import (
    build_truncation_diagnostics,
    write_reports,
)


def test_tokenizer_preprocessing_returns_required_tensors():
    encoded = tokenize_texts(["one two", "three four five"], FakeTokenizer(), max_length=4)

    assert set(encoded) == {"input_ids", "attention_mask"}
    assert isinstance(encoded["input_ids"], torch.Tensor)
    assert isinstance(encoded["attention_mask"], torch.Tensor)
    assert encoded["input_ids"].shape == (2, 4)
    assert encoded["attention_mask"].shape == (2, 4)


def test_tokenizer_preprocessing_respects_max_length():
    encoded = tokenize_texts(["one two three four five"], FakeTokenizer(), max_length=3)

    assert encoded["input_ids"].shape == (1, 3)
    assert encoded["input_ids"][0].tolist() == [1, 2, 3]
    assert encoded["attention_mask"][0].tolist() == [1, 1, 1]


def test_tiny_transformer_dataset_wrapper_returns_model_inputs():
    dataset = TruncatedTextDataset(
        texts=["alpha beta", "gamma"],
        labels=[1, 0],
        tokenizer=FakeTokenizer(),
        max_length=3,
    )

    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].tolist() == [1, 2, 0]
    assert item["attention_mask"].tolist() == [1, 1, 0]
    assert item["labels"].item() == 1


def test_report_writer_creates_expected_keys_and_content(tmp_path):
    report = _fake_report()

    markdown_path, json_path = write_reports(report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["dataset_name"] == "toy"
    assert saved["model_name"] == "fake-model"
    assert "truncation_diagnostics" in saved
    assert "truncation baseline" in markdown
    assert "Confusion Matrix" in markdown


def test_truncation_diagnostics_use_whitespace_counts():
    diagnostics = build_truncation_diagnostics(
        ["one two", " ".join(["token"] * 10)],
        max_length=4,
    )

    assert diagnostics["document_count"] == 2
    assert diagnostics["documents_above_max_length"] == 1
    assert diagnostics["percent_above_max_length"] == 50.0
    assert diagnostics["p95_whitespace_tokens"] > 9


class FakeTokenizer:
    def __call__(
        self,
        texts,
        truncation,
        max_length,
        padding,
        return_tensors,
    ):
        assert truncation is True
        assert padding == "max_length"
        assert return_tensors == "pt"

        input_ids = []
        attention_masks = []
        for text in texts:
            token_count = len(text.split())
            tokens = list(range(1, token_count + 1))[:max_length]
            padding_length = max_length - len(tokens)
            input_ids.append(tokens + [0] * padding_length)
            attention_masks.append([1] * len(tokens) + [0] * padding_length)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
        }


def _fake_report():
    return {
        "dataset_name": "toy",
        "hf_path": "toy/path",
        "text_field": "text",
        "label_field": "label",
        "model_name": "fake-model",
        "max_length": 8,
        "max_train_samples": 4,
        "max_test_samples": 2,
        "train_size": 4,
        "test_size": 2,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "random_state": 42,
        "device": "cpu",
        "truncation_note": "fake truncation note",
        "truncation_diagnostics": {
            "max_length": 8,
            "document_count": 2,
            "mean_whitespace_tokens": 7.5,
            "median_whitespace_tokens": 7.5,
            "p95_whitespace_tokens": 10.0,
            "documents_above_max_length": 1,
            "percent_above_max_length": 50.0,
        },
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "per_class_f1": {"a": 0.5, "b": 0.5},
        "confusion_matrix": [[1, 0], [1, 0]],
        "labels": [0, 1],
        "label_names": ["a", "b"],
    }
