from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from longdoc_transformer_classifier.config import DatasetConfig
from longdoc_transformer_classifier.data import load_ag_news, load_text_classification_dataset
import longdoc_transformer_classifier.data as data_module


def test_load_ag_news_with_tiny_sample_sizes(monkeypatch):
    calls = []

    def fake_load_dataset(name):
        calls.append(name)
        return _fake_ag_news()

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    dataset = load_ag_news(max_train_samples=4, max_test_samples=4)

    assert calls == [data_module.AG_NEWS_HF_PATH]
    assert dataset.name == "ag_news"
    assert dataset.label_names == ["World", "Sports", "Business", "Sci/Tech"]
    assert len(dataset.train_texts) == 4
    assert len(dataset.test_texts) == 4
    assert dataset.train_labels == [0, 1, 2, 3]
    assert dataset.test_labels == [0, 1, 2, 3]


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
