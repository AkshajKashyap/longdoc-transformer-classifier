import json

import torch

from longdoc_transformer_classifier.training.train_chunked_transformer import (
    aggregate_chunk_probabilities,
    build_chunk_diagnostics,
    expand_documents_to_chunks,
    write_reports,
)


def test_documents_expand_into_chunks_with_parent_document_ids():
    examples = expand_documents_to_chunks(
        texts=[" ".join(str(index) for index in range(6)), "alpha beta"],
        labels=[1, 0],
        chunk_size=3,
        chunk_overlap=1,
        max_chunks_per_doc=10,
    )

    assert examples.chunk_texts == ["0 1 2", "2 3 4", "4 5", "alpha beta"]
    assert examples.chunk_labels == [1, 1, 1, 0]
    assert examples.document_ids == [0, 0, 0, 1]
    assert examples.chunks_per_document_before_cap == [3, 1]
    assert examples.chunks_per_document_after_cap == [3, 1]


def test_max_chunks_per_doc_cap_works():
    examples = expand_documents_to_chunks(
        texts=[" ".join(str(index) for index in range(10))],
        labels=[2],
        chunk_size=3,
        chunk_overlap=1,
        max_chunks_per_doc=2,
    )

    assert examples.chunk_texts == ["0 1 2", "2 3 4"]
    assert examples.document_ids == [0, 0]
    assert examples.chunks_per_document_before_cap == [5]
    assert examples.chunks_per_document_after_cap == [2]


def test_mean_proba_aggregation_returns_one_prediction_per_document():
    predictions = aggregate_chunk_probabilities(
        chunk_probabilities=torch.tensor(
                [
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.1, 0.9],
                ]
        ),
        document_ids=[0, 0, 1],
        aggregation="mean_proba",
        num_documents=2,
    )

    assert predictions == [0, 1]


def test_majority_vote_aggregation_handles_multiple_chunks_per_document():
    predictions = aggregate_chunk_probabilities(
        chunk_probabilities=torch.tensor(
            [
                [0.1, 0.9, 0.0],
                [0.2, 0.7, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        document_ids=[0, 0, 1, 1],
        aggregation="majority_vote",
        num_documents=2,
    )

    assert predictions == [1, 0]


def test_report_writer_includes_chunk_diagnostics(tmp_path):
    train_examples = expand_documents_to_chunks(
        texts=["one two three four"],
        labels=[0],
        chunk_size=2,
        chunk_overlap=0,
        max_chunks_per_doc=4,
    )
    test_examples = expand_documents_to_chunks(
        texts=["alpha beta gamma"],
        labels=[1],
        chunk_size=2,
        chunk_overlap=0,
        max_chunks_per_doc=4,
    )
    report = _fake_report(build_chunk_diagnostics(train_examples, test_examples, 4, "mean_proba"))

    markdown_path, json_path = write_reports(report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["chunk_diagnostics"]["train"]["generated_chunks"] == 2
    assert saved["chunk_diagnostics"]["aggregation"] == "mean_proba"
    assert "Chunk Diagnostics" in markdown
    assert "weak labels inherited from the parent document" in markdown


def _fake_report(chunk_diagnostics):
    return {
        "dataset_name": "toy",
        "hf_path": "toy/path",
        "text_field": "text",
        "label_field": "label",
        "model_name": "fake-model",
        "chunk_size": 2,
        "chunk_overlap": 0,
        "max_length": 8,
        "max_chunks_per_doc": 4,
        "aggregation": "mean_proba",
        "max_train_samples": 4,
        "max_test_samples": 2,
        "train_size": 4,
        "test_size": 2,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "random_state": 42,
        "device": "cpu",
        "limitation_note": "fake limitation note",
        "chunk_diagnostics": chunk_diagnostics,
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "per_class_f1": {"a": 0.5, "b": 0.5},
        "confusion_matrix": [[1, 0], [1, 0]],
        "labels": [0, 1],
        "label_names": ["a", "b"],
    }
