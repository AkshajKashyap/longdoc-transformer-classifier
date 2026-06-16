# ag_news Baseline Report

## Model

TF-IDF features with Logistic Regression.

## Dataset

- Dataset name: `ag_news`
- Hugging Face dataset: `fancyzhx/ag_news`
- Text field: `text`
- Label field: `label`

## Configuration

- Train samples: 1000
- Test samples: 500
- Max TF-IDF features: 50000
- N-gram range: 1-2
- Random state: 42

## Metrics

- Accuracy: 0.7400
- Macro-F1: 0.7376

## Per-Class F1

| Class | F1 |
| --- | ---: |
| World | 0.7915 |
| Sports | 0.8667 |
| Business | 0.6667 |
| Sci/Tech | 0.6255 |

## Confusion Matrix

| true/pred | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| World | 93 | 14 | 8 | 10 |
| Sports | 1 | 117 | 0 | 7 |
| Business | 13 | 3 | 84 | 25 |
| Sci/Tech | 3 | 11 | 35 | 76 |
