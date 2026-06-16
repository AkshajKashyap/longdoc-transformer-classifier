# AG News Baseline Report

## Model

TF-IDF features with Logistic Regression.

## Configuration

- Train samples: 500
- Test samples: 200
- Max TF-IDF features: 50000
- N-gram range: 1-2
- Random state: 42

## Metrics

- Accuracy: 0.7250
- Macro-F1: 0.7251

## Per-Class F1

| Class | F1 |
| --- | ---: |
| World | 0.7473 |
| Sports | 0.8350 |
| Business | 0.6981 |
| Sci/Tech | 0.6200 |

## Confusion Matrix

| true/pred | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| World | 34 | 6 | 4 | 6 |
| Sports | 1 | 43 | 0 | 6 |
| Business | 4 | 2 | 37 | 7 |
| Sci/Tech | 2 | 2 | 15 | 31 |
