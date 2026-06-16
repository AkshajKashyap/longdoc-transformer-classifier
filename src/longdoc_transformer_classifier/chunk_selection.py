from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from longdoc_transformer_classifier.chunking import DocumentChunk

CHUNK_SELECTION_STRATEGIES = {"first_k", "uniform_k", "idf_top_k", "longest_k"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class IDFChunkScorer:
    idf_by_token: dict[str, float]
    fitted_on_num_chunks: int
    scoring_mode: str = "average_idf"

    @classmethod
    def fit(
        cls,
        chunks: Sequence[DocumentChunk],
        scoring_mode: str = "average_idf",
    ) -> IDFChunkScorer:
        validate_scoring_mode(scoring_mode)
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            document_frequency.update(set(tokenize_for_idf(chunk.text)))

        num_chunks = len(chunks)
        idf_by_token = {
            token: math.log((1 + num_chunks) / (1 + frequency)) + 1.0
            for token, frequency in sorted(document_frequency.items())
        }
        return cls(
            idf_by_token=idf_by_token,
            fitted_on_num_chunks=num_chunks,
            scoring_mode=scoring_mode,
        )

    def score_chunk(self, chunk: DocumentChunk) -> float:
        token_scores = [
            self.idf_by_token[token]
            for token in tokenize_for_idf(chunk.text)
            if token in self.idf_by_token
        ]
        if not token_scores:
            return 0.0
        total = sum(token_scores)
        if self.scoring_mode == "sum_idf":
            return total
        return total / len(token_scores)

    def metadata(self) -> dict[str, Any]:
        return {
            "vocabulary_size": len(self.idf_by_token),
            "scoring_mode": self.scoring_mode,
            "fitted_on_num_chunks": self.fitted_on_num_chunks,
        }


def tokenize_for_idf(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def select_first_k(chunks: Sequence[DocumentChunk], max_chunks: int) -> list[DocumentChunk]:
    validate_max_chunks(max_chunks)
    return list(chunks[:max_chunks])


def select_uniform_k(chunks: Sequence[DocumentChunk], max_chunks: int) -> list[DocumentChunk]:
    validate_max_chunks(max_chunks)
    if len(chunks) <= max_chunks:
        return list(chunks)
    if max_chunks == 1:
        return [chunks[0]]

    indices = _evenly_spaced_indices(len(chunks), max_chunks)
    return [chunks[index] for index in indices]


def select_idf_top_k(
    chunks: Sequence[DocumentChunk],
    max_chunks: int,
    scorer: IDFChunkScorer,
) -> list[DocumentChunk]:
    validate_max_chunks(max_chunks)
    if len(chunks) <= max_chunks:
        return list(chunks)

    ranked = sorted(
        chunks,
        key=lambda chunk: (-scorer.score_chunk(chunk), chunk.chunk_id),
    )
    selected = ranked[:max_chunks]
    return sorted(selected, key=lambda chunk: chunk.chunk_id)


def select_longest_k(chunks: Sequence[DocumentChunk], max_chunks: int) -> list[DocumentChunk]:
    validate_max_chunks(max_chunks)
    if len(chunks) <= max_chunks:
        return list(chunks)

    ranked = sorted(
        chunks,
        key=lambda chunk: (-len(chunk.text.split()), chunk.chunk_id),
    )
    selected = ranked[:max_chunks]
    return sorted(selected, key=lambda chunk: chunk.chunk_id)


def select_chunks_by_strategy(
    chunks: Sequence[DocumentChunk],
    max_chunks: int,
    strategy: str,
    idf_scorer: IDFChunkScorer | None = None,
) -> list[DocumentChunk]:
    validate_strategy(strategy)
    if strategy == "first_k":
        return select_first_k(chunks, max_chunks)
    if strategy == "uniform_k":
        return select_uniform_k(chunks, max_chunks)
    if strategy == "longest_k":
        return select_longest_k(chunks, max_chunks)
    if idf_scorer is None:
        msg = "idf_top_k requires a training-fitted IDFChunkScorer."
        raise ValueError(msg)
    return select_idf_top_k(chunks, max_chunks, idf_scorer)


def validate_max_chunks(max_chunks: int) -> None:
    if max_chunks <= 0:
        msg = "max_chunks must be greater than 0."
        raise ValueError(msg)


def validate_strategy(strategy: str) -> None:
    if strategy not in CHUNK_SELECTION_STRATEGIES:
        supported = ", ".join(sorted(CHUNK_SELECTION_STRATEGIES))
        msg = f"Unsupported chunk selection strategy '{strategy}'. Supported values: {supported}."
        raise ValueError(msg)


def validate_scoring_mode(scoring_mode: str) -> None:
    if scoring_mode not in {"average_idf", "sum_idf"}:
        msg = "scoring_mode must be 'average_idf' or 'sum_idf'."
        raise ValueError(msg)


def _evenly_spaced_indices(item_count: int, selected_count: int) -> list[int]:
    if selected_count >= item_count:
        return list(range(item_count))

    last_index = item_count - 1
    indices: list[int] = []
    for position in range(selected_count):
        raw_index = position * last_index / (selected_count - 1)
        index = int(round(raw_index))
        if indices and index <= indices[-1]:
            index = indices[-1] + 1
        remaining_slots = selected_count - len(indices) - 1
        max_allowed = last_index - remaining_slots
        indices.append(min(index, max_allowed))
    return indices

