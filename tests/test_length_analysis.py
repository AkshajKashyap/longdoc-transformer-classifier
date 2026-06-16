from longdoc_transformer_classifier.length_analysis import summarize_document_lengths


def test_summarize_document_lengths_reports_bert_limit_rate():
    summary = summarize_document_lengths(["one two", " ".join(["token"] * 600)])

    assert summary["document_count"] == 2
    assert summary["character_lengths"]["count"] == 2
    assert summary["word_counts"]["max"] == 600
    assert summary["whitespace_token_counts"]["p95"] > 500
    assert summary["documents_above_bert_limit"] == 1
    assert summary["percent_above_bert_limit"] == 50.0
