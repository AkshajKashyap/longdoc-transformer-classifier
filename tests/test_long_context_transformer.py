import json
from types import SimpleNamespace

import torch

from longdoc_transformer_classifier.training.train_long_context_transformer import (
    build_global_attention_mask,
    build_model_inputs,
    count_parameters,
    freeze_base_encoder,
    write_reports,
)


class TinySequenceClassifier(torch.nn.Module):
    base_model_prefix = "encoder"

    def __init__(self, model_type="longformer"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, **kwargs):
        return kwargs


def test_global_attention_mask_sets_first_token_only():
    input_ids = torch.tensor([[101, 5, 6], [101, 7, 0]])

    mask = build_global_attention_mask(input_ids)

    assert mask.tolist() == [[1, 0, 0], [1, 0, 0]]


def test_global_attention_is_only_added_for_longformer_models():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([0]),
    }

    longformer_inputs = build_model_inputs(
        batch,
        TinySequenceClassifier(model_type="longformer"),
        include_labels=True,
    )
    bert_inputs = build_model_inputs(
        batch,
        TinySequenceClassifier(model_type="bert"),
        include_labels=True,
    )

    assert longformer_inputs["global_attention_mask"].tolist() == [[1, 0, 0]]
    assert "global_attention_mask" not in bert_inputs


def test_freeze_encoder_changes_trainable_parameter_count():
    model = TinySequenceClassifier()
    before = count_parameters(model)["trainable_parameter_count"]

    freeze_base_encoder(model)
    after = count_parameters(model)["trainable_parameter_count"]

    assert after < before
    assert after == sum(parameter.numel() for parameter in model.classifier.parameters())


def test_long_context_report_writer_includes_key_settings(tmp_path):
    report = _fake_report()

    markdown_path, json_path = write_reports(report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["freeze_encoder"] is True
    assert saved["max_length"] == 1024
    assert saved["trainable_parameter_count"] == 10
    assert "Freeze encoder: True" in markdown
    assert "Trainable parameters: 10" in markdown


def _fake_report():
    return {
        "method": "long_context_transformer_toy",
        "dataset": "toy",
        "dataset_name": "toy",
        "hf_path": "toy/path",
        "text_field": "text",
        "label_field": "label",
        "model_name": "fake-longformer",
        "max_length": 1024,
        "max_train_samples": 4,
        "max_test_samples": 2,
        "train_size": 4,
        "test_size": 2,
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "freeze_encoder": True,
        "trainable_parameter_count": 10,
        "total_parameter_count": 100,
        "random_state": 42,
        "device": "cpu",
        "diagnostics": {
            "max_length": 1024,
            "document_count": 2,
            "mean_whitespace_tokens": 100.0,
            "median_whitespace_tokens": 100.0,
            "p95_whitespace_tokens": 120.0,
            "documents_above_max_length": 0,
            "percent_above_max_length": 0.0,
        },
        "limitations": ["fake limitation"],
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "per_class_f1": {"a": 0.5, "b": 0.5},
        "confusion_matrix": [[1, 0], [1, 0]],
        "labels": [0, 1],
        "label_names": ["a", "b"],
    }

