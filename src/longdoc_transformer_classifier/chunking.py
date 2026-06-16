from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    document_id: int | str
    chunk_id: int
    text: str
    start_word: int
    end_word: int


def chunk_document(text: str, chunk_size: int = 512, overlap: int = 0) -> list[str]:
    validate_chunking_args(chunk_size, overlap)
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def chunk_documents(
    texts: Sequence[str],
    chunk_size: int = 512,
    overlap: int = 0,
    document_ids: Sequence[int | str] | None = None,
) -> list[DocumentChunk]:
    validate_chunking_args(chunk_size, overlap)
    resolved_document_ids = list(document_ids) if document_ids is not None else list(range(len(texts)))
    if len(resolved_document_ids) != len(texts):
        msg = "document_ids must have the same length as texts."
        raise ValueError(msg)

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for document_id, text in zip(resolved_document_ids, texts, strict=True):
        words = text.split()
        if not words:
            continue
        for chunk_id, start in enumerate(range(0, len(words), step)):
            end = min(start + chunk_size, len(words))
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=" ".join(words[start:end]),
                    start_word=start,
                    end_word=end,
                )
            )
            if end == len(words):
                break
    return chunks


def validate_chunking_args(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        msg = "chunk_size must be greater than 0."
        raise ValueError(msg)
    if overlap < 0:
        msg = "overlap must be greater than or equal to 0."
        raise ValueError(msg)
    if overlap >= chunk_size:
        msg = "overlap must be smaller than chunk_size."
        raise ValueError(msg)
