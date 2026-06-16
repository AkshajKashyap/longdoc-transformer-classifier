# Model Comparison

> Smoke runs with tiny models and tiny sample sizes are useful engineering checks, not final model rankings.

## Results

| Method | Dataset | Accuracy | Macro-F1 | Key Settings |
| --- | --- | ---: | ---: | --- |
| `baseline_ag_news` | `ag_news` | 0.7400 | 0.7376 | TF-IDF + Logistic Regression |
| `baseline_arxiv` | `arxiv` | 0.7880 | 0.7826 | TF-IDF + Logistic Regression |
| `truncated_transformer_ag_news` | `ag_news` | 0.2800 | 0.2605 | model=prajjwal1/bert-tiny, max_length=128 |
| `truncated_transformer_arxiv` | `arxiv` | 0.1600 | 0.0782 | model=prajjwal1/bert-tiny, max_length=512 |
| `chunked_transformer_ag_news` | `ag_news` | 0.2000 | 0.0847 | model=prajjwal1/bert-tiny, chunk_size=100, overlap=20, cap=4, aggregation=mean_proba |
| `chunked_transformer_arxiv` | `arxiv` | 0.2600 | 0.1663 | model=prajjwal1/bert-tiny, chunk_size=220, overlap=40, cap=8, aggregation=mean_proba |
| `summary_classifier_arxiv` | `arxiv` | 0.2000 | 0.1515 | summarizer=sshleifer/distilbart-cnn-12-6, classifier=tfidf, classifier_model=n/a |

## Best By Dataset

- `ag_news`: `baseline_ag_news` by macro-F1 (0.7376)
- `arxiv`: `baseline_arxiv` by macro-F1 (0.7826)

## Interpretation

- TF-IDF proves the classical lexical baseline and often remains strong on small samples.
- Truncated transformers show what happens when only the first fixed window is visible.
- Chunked transformers make the model structurally long-document aware by aggregating chunk predictions.
- Summary-first classification tests compression before classification, but can lose discriminative details.
