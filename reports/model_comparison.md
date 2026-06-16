# Model Comparison

> Smoke runs with tiny models and tiny sample sizes are useful engineering checks, not final model rankings.

## Results

| Method | Dataset | Accuracy | Macro-F1 | Key Settings |
| --- | --- | ---: | ---: | --- |
| `baseline_ag_news` | `ag_news` | 0.7400 | 0.7376 | TF-IDF + Logistic Regression |
| `baseline_arxiv` | `arxiv` | 0.5600 | 0.4972 | TF-IDF + Logistic Regression |
| `truncated_transformer_ag_news` | `ag_news` | 0.2800 | 0.2605 | model=prajjwal1/bert-tiny, max_length=128 |
| `truncated_transformer_arxiv` | `arxiv` | 0.1600 | 0.0782 | model=prajjwal1/bert-tiny, max_length=512 |
| `chunked_transformer_ag_news` | `ag_news` | 0.2000 | 0.0847 | model=prajjwal1/bert-tiny, chunk_size=100, overlap=20, cap=4, aggregation=mean_proba |
| `chunked_transformer_arxiv` | `arxiv` | 0.2600 | 0.1663 | model=prajjwal1/bert-tiny, chunk_size=220, overlap=40, cap=8, aggregation=mean_proba |
| `summary_classifier_arxiv` | `arxiv` | 0.2000 | 0.1515 | summarizer=sshleifer/distilbart-cnn-12-6, classifier=tfidf, classifier_model=n/a |

## Benchmark View

| Method | Family | Dataset | Accuracy | Macro-F1 | Train Samples | Test Samples | Model | Key Limitation | Structural Takeaway |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `baseline_ag_news` | TF-IDF + Logistic Regression | `ag_news` | 0.7400 | 0.7376 | 1000 | 500 | TF-IDF + Logistic Regression | No neural long-context reasoning; relies on lexical signals. | Strong lexical baseline, no neural long-context reasoning. |
| `baseline_arxiv` | TF-IDF + Logistic Regression | `arxiv` | 0.5600 | 0.4972 | 100 | 50 | TF-IDF + Logistic Regression | No neural long-context reasoning; relies on lexical features. | Strong lexical baseline, no neural long-context reasoning. |
| `truncated_transformer_ag_news` | Truncated Transformer | `ag_news` | 0.2800 | 0.2605 | 100 | 50 | prajjwal1/bert-tiny | This is a truncation baseline: each document is tokenized once and only the first max_length tokens are available to the classifier. | Shows what happens when long documents are clipped. |
| `truncated_transformer_arxiv` | Truncated Transformer | `arxiv` | 0.1600 | 0.0782 | 100 | 50 | prajjwal1/bert-tiny | This is a truncation baseline: each document is tokenized once and only the first max_length tokens are available to the classifier. | Shows what happens when long documents are clipped. |
| `chunked_transformer_ag_news` | Chunked Transformer | `ag_news` | 0.2000 | 0.0847 | 100 | 50 | prajjwal1/bert-tiny | Chunk labels are weak labels inherited from the parent document; not every chunk necessarily contains evidence for the document label. | Structurally sees more text, but uses weak inherited chunk labels. |
| `chunked_transformer_arxiv` | Chunked Transformer | `arxiv` | 0.2600 | 0.1663 | 100 | 50 | prajjwal1/bert-tiny | Chunk labels are weak labels inherited from the parent document; not every chunk necessarily contains evidence for the document label. | Structurally sees more text, but uses weak inherited chunk labels. |
| `summary_classifier_arxiv` | Summary-First Classifier | `arxiv` | 0.2000 | 0.1515 | 30 | 15 | sshleifer/distilbart-cnn-12-6 + tfidf | This is not full summarizer fine-tuning. | Compresses long documents, but may discard class evidence. |

## Best By Dataset

- `ag_news`: `baseline_ag_news` by macro-F1 (0.7376)
- `arxiv`: `baseline_arxiv` by macro-F1 (0.4972)

## Interpretation

- TF-IDF proves the classical lexical baseline and often remains strong on small samples.
- Truncated transformers show what happens when only the first fixed window is visible.
- Chunked transformers make the model structurally long-document aware by aggregating chunk predictions.
- Summary-first classification tests compression before classification, but can lose discriminative details.
