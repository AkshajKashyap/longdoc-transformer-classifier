import pytest

from longdoc_transformer_classifier.chunking import chunk_document, chunk_documents


def test_short_document_returns_one_chunk():
    chunks = chunk_document("one two three", chunk_size=10)

    assert chunks == ["one two three"]


def test_long_document_returns_multiple_chunks():
    text = " ".join(str(index) for index in range(10))

    chunks = chunk_document(text, chunk_size=4)

    assert chunks == ["0 1 2 3", "4 5 6 7", "8 9"]


def test_overlap_works_correctly():
    text = " ".join(str(index) for index in range(8))

    chunks = chunk_document(text, chunk_size=4, overlap=2)

    assert chunks == ["0 1 2 3", "2 3 4 5", "4 5 6 7"]


def test_document_ids_are_preserved():
    chunks = chunk_documents(
        ["alpha beta gamma delta", "one two three four"],
        chunk_size=2,
        document_ids=["doc-a", "doc-b"],
    )

    assert [chunk.document_id for chunk in chunks] == ["doc-a", "doc-a", "doc-b", "doc-b"]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 0, 1]


def test_invalid_overlap_raises_clear_error():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_document("one two three", chunk_size=3, overlap=3)
