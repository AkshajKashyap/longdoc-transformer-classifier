from longdoc_transformer_classifier.config import TfidfConfig
from longdoc_transformer_classifier.features import fit_tfidf, transform_tfidf


def test_tfidf_vectorizer_shape_sanity():
    texts = ["apple banana", "banana carrot", "apple carrot"]
    config = TfidfConfig(max_features=3, ngram_range=(1, 1))

    vectorizer, features = fit_tfidf(texts, config)
    transformed = transform_tfidf(vectorizer, ["apple banana"])

    assert features.shape == (3, 3)
    assert transformed.shape == (1, 3)
