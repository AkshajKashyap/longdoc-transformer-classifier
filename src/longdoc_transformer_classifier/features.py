from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from longdoc_transformer_classifier.config import TfidfConfig


def build_tfidf_vectorizer(config: TfidfConfig | None = None) -> TfidfVectorizer:
    config = config or TfidfConfig()
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
    )


def fit_tfidf(
    texts: Sequence[str], config: TfidfConfig | None = None
) -> tuple[TfidfVectorizer, Any]:
    vectorizer = build_tfidf_vectorizer(config)
    features = vectorizer.fit_transform(texts)
    return vectorizer, features


def transform_tfidf(vectorizer: TfidfVectorizer, texts: Sequence[str]) -> Any:
    return vectorizer.transform(texts)
