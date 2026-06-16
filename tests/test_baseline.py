from longdoc_transformer_classifier.config import BaselineModelConfig, TfidfConfig
from longdoc_transformer_classifier.features import fit_tfidf
from longdoc_transformer_classifier.models.baseline import predict, predict_proba, train


def test_baseline_can_fit_predict_and_predict_proba_on_toy_data():
    texts = [
        "space mission launch",
        "rocket reaches orbit",
        "market stocks rally",
        "shares and bonds fall",
    ]
    labels = [0, 0, 1, 1]
    vectorizer, features = fit_tfidf(texts, TfidfConfig(max_features=20, ngram_range=(1, 1)))

    classifier = train(features, labels, BaselineModelConfig(max_iter=200))
    predictions = predict(classifier, features)
    probabilities = predict_proba(classifier, features)

    assert len(predictions) == len(labels)
    assert set(predictions) <= {0, 1}
    assert len(probabilities) == len(labels)
    assert len(probabilities[0]) == 2
    assert vectorizer.vocabulary_
