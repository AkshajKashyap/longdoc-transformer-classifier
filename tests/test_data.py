from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import (
    ARXIV_HF_PATH,
    load_ag_news,
    load_text_classification_dataset,
)
import longdoc_transformer_classifier.data as data_module


def test_load_ag_news_with_tiny_sample_sizes(monkeypatch):
    calls = []

    def fake_load_dataset(name, cache_dir=None):
        calls.append(name)
        return _fake_ag_news()

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    dataset = load_ag_news(max_train_samples=4, max_test_samples=4)

    assert calls == [data_module.AG_NEWS_HF_PATH]
    assert dataset.dataset_name == "ag_news"
    assert dataset.name == "ag_news"
    assert dataset.text_field == "text"
    assert dataset.label_field == "label"
    assert dataset.label_names == ["World", "Sports", "Business", "Sci/Tech"]
    assert len(dataset.train_texts) == 4
    assert len(dataset.test_texts) == 4
    assert dataset.train_labels == [0, 1, 2, 3]
    assert dataset.test_labels == [0, 1, 2, 3]


def test_load_arxiv_dataset_by_name(monkeypatch):
    calls = []

    def fake_load_dataset(name, cache_dir=None):
        calls.append(name)
        return _fake_arxiv()

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    dataset = load_text_classification_dataset(
        DatasetConfig(name="arxiv", max_train_samples=4, max_test_samples=2)
    )

    assert calls == [ARXIV_HF_PATH]
    assert dataset.dataset_name == "arxiv"
    assert dataset.hf_path == ARXIV_HF_PATH
    assert dataset.text_field == "text"
    assert dataset.label_field == "label"
    assert dataset.label_names == ["cs.CV", "math.ST"]
    assert dataset.train_labels == [0, 0, 1, 1]
    assert dataset.test_labels == [0, 1]


def test_load_text_classification_dataset_rejects_unknown_dataset():
    config = DatasetConfig(name="not_supported")

    try:
        load_text_classification_dataset(config)
    except ValueError as error:
        assert "Unsupported dataset" in str(error)
    else:
        raise AssertionError("Expected unsupported datasets to raise ValueError.")


def _fake_ag_news() -> DatasetDict:
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["World", "Sports", "Business", "Sci/Tech"]),
        }
    )
    train = Dataset.from_dict(
        {
            "text": [
                "world one",
                "world two",
                "sports one",
                "sports two",
                "business one",
                "business two",
                "science one",
                "science two",
            ],
            "label": [0, 0, 1, 1, 2, 2, 3, 3],
        },
        features=features,
    )
    test = Dataset.from_dict(
        {
            "text": [
                "world test",
                "world holdout",
                "sports test",
                "sports holdout",
                "business test",
                "business holdout",
                "science test",
                "science holdout",
            ],
            "label": [0, 0, 1, 1, 2, 2, 3, 3],
        },
        features=features,
    )
    return DatasetDict({"train": train, "test": test})


def _fake_arxiv() -> DatasetDict:
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["cs.CV", "math.ST"]),
        }
    )
    train = Dataset.from_dict(
        {
            "text": [
                "vision paper one",
                "vision paper two",
                "statistics paper one",
                "statistics paper two",
            ],
            "label": [0, 0, 1, 1],
        },
        features=features,
    )
    test = Dataset.from_dict(
        {
            "text": ["vision test", "statistics test"],
            "label": [0, 1],
        },
        features=features,
    )
    return DatasetDict({"train": train, "test": test})
