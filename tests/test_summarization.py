from pathlib import Path

from longdoc_transformer_classifier.summarization import (
    SummaryGenerationConfig,
    SummaryRecord,
    build_summary_cache_key,
    build_summary_cache_path,
    calculate_summary_statistics,
    load_summary_records,
    save_summary_records,
)


def test_summary_cache_path_and_key_are_deterministic():
    config = SummaryGenerationConfig(
        max_input_tokens=1024,
        max_new_tokens=120,
        min_new_tokens=30,
        num_beams=2,
    )

    key_one = build_summary_cache_key("arxiv", "train", "sshleifer/distilbart-cnn-12-6", 30, config)
    key_two = build_summary_cache_key("arxiv", "train", "sshleifer/distilbart-cnn-12-6", 30, config)
    path_one = build_summary_cache_path(
        Path("data/processed/summaries"),
        "arxiv",
        "train",
        "sshleifer/distilbart-cnn-12-6",
        30,
        config,
    )
    path_two = build_summary_cache_path(
        Path("data/processed/summaries"),
        "arxiv",
        "train",
        "sshleifer/distilbart-cnn-12-6",
        30,
        config,
    )

    assert key_one == key_two
    assert path_one == path_two
    assert path_one.name.startswith("arxiv_train_sshleifer_distilbart-cnn-12-6_")
    assert path_one.suffix == ".jsonl"


def test_summary_jsonl_save_load_roundtrip(tmp_path):
    records = [
        SummaryRecord(
            doc_id=0,
            label=1,
            original_word_count=100,
            summarizer_input_token_count=80,
            summarizer_input_word_count=70,
            summary_word_count=20,
            compression_ratio=0.2,
            summary_text="short summary",
        )
    ]
    path = tmp_path / "summaries.jsonl"

    save_summary_records(records, path)
    loaded = load_summary_records(path)

    assert loaded == records


def test_compression_statistics_calculation():
    records = [
        SummaryRecord(
            doc_id=0,
            label=0,
            original_word_count=100,
            summarizer_input_token_count=90,
            summarizer_input_word_count=80,
            summary_word_count=20,
            compression_ratio=0.2,
            summary_text="summary one",
        ),
        SummaryRecord(
            doc_id=1,
            label=1,
            original_word_count=50,
            summarizer_input_token_count=55,
            summarizer_input_word_count=50,
            summary_word_count=10,
            compression_ratio=0.2,
            summary_text="summary two",
        ),
    ]

    stats = calculate_summary_statistics(records)

    assert stats["document_count"] == 2
    assert stats["average_original_word_count"] == 75.0
    assert stats["average_summarizer_input_word_count"] == 65.0
    assert stats["average_summary_word_count"] == 15.0
    assert stats["average_compression_ratio"] == 0.2
    assert stats["percent_summarizer_input_shorter_than_original"] == 50.0
