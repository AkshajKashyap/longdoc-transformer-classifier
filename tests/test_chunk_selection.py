import pytest

from longdoc_transformer_classifier.chunk_selection import (
    IDFChunkScorer,
    select_chunks_by_strategy,
    select_first_k,
    select_idf_top_k,
    select_uniform_k,
)
from longdoc_transformer_classifier.chunking import DocumentChunk


def test_first_k_selects_first_n_chunks():
    chunks = _chunks_for_document(0, 5)

    selected = select_first_k(chunks, 3)

    assert [chunk.chunk_id for chunk in selected] == [0, 1, 2]


def test_uniform_k_selects_beginning_middle_and_end():
    chunks = _chunks_for_document(0, 10)

    selected = select_uniform_k(chunks, 3)

    assert [chunk.chunk_id for chunk in selected] == [0, 4, 9]


def test_idf_scorer_fits_training_chunks_and_scores_deterministically():
    training_chunks = [
        DocumentChunk(0, 0, "common alpha", 0, 2),
        DocumentChunk(0, 1, "common beta", 2, 4),
        DocumentChunk(1, 0, "common beta gamma", 0, 3),
    ]
    scorer = IDFChunkScorer.fit(training_chunks)
    test_chunk = DocumentChunk(2, 0, "alpha gamma unknown", 0, 3)

    first_score = scorer.score_chunk(test_chunk)
    second_score = scorer.score_chunk(test_chunk)

    assert first_score == second_score
    assert scorer.metadata() == {
        "vocabulary_size": 4,
        "scoring_mode": "average_idf",
        "fitted_on_num_chunks": 3,
    }


def test_idf_top_k_uses_training_fitted_scorer():
    training_chunks = [
        DocumentChunk(0, 0, "common common", 0, 2),
        DocumentChunk(0, 1, "common rareterm", 2, 4),
    ]
    candidate_chunks = [
        DocumentChunk(1, 0, "common common", 0, 2),
        DocumentChunk(1, 1, "rareterm", 2, 3),
    ]
    scorer = IDFChunkScorer.fit(training_chunks)

    selected = select_idf_top_k(candidate_chunks, 1, scorer)

    assert [chunk.chunk_id for chunk in selected] == [1]


def test_select_chunks_by_strategy_preserves_document_ids():
    chunks = _chunks_for_document("doc-a", 6)

    selected = select_chunks_by_strategy(chunks, max_chunks=3, strategy="uniform_k")

    assert {chunk.document_id for chunk in selected} == {"doc-a"}


def test_invalid_chunk_selection_strategy_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported chunk selection strategy"):
        select_chunks_by_strategy([], max_chunks=2, strategy="not-real")


def _chunks_for_document(document_id, count):
    return [
        DocumentChunk(
            document_id=document_id,
            chunk_id=index,
            text=f"chunk {index}",
            start_word=index * 2,
            end_word=index * 2 + 2,
        )
        for index in range(count)
    ]
