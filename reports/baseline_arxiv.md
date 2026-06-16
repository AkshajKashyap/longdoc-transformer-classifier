# arxiv Baseline Report

## Model

TF-IDF features with Logistic Regression.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Train samples: 5000
- Test samples: 1000
- Max TF-IDF features: 100000
- N-gram range: 1-2
- Min document frequency: 2
- Max document frequency: 0.95
- Sublinear TF: True
- Class weight: `balanced`
- Solver: `lbfgs`
- Max iterations: 1000
- Random state: 42

## Metrics

- Accuracy: 0.8340
- Macro-F1: 0.8320

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.9674 |
| cs.CV | 0.8235 |
| cs.AI | 0.6211 |
| cs.SY | 0.8114 |
| math.GR | 0.9457 |
| cs.CE | 0.8021 |
| cs.PL | 0.8842 |
| cs.IT | 0.8605 |
| cs.DS | 0.8508 |
| cs.NE | 0.7188 |
| math.ST | 0.8663 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 89 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 77 | 3 | 1 | 0 | 2 | 0 | 0 | 0 | 7 | 1 |
| cs.AI | 0 | 7 | 50 | 2 | 0 | 1 | 10 | 0 | 4 | 15 | 2 |
| cs.SY | 1 | 1 | 4 | 71 | 0 | 5 | 0 | 2 | 2 | 3 | 2 |
| math.GR | 2 | 0 | 0 | 0 | 87 | 1 | 1 | 0 | 0 | 0 | 0 |
| cs.CE | 0 | 0 | 2 | 4 | 0 | 75 | 1 | 2 | 3 | 4 | 0 |
| cs.PL | 1 | 0 | 2 | 0 | 0 | 4 | 84 | 0 | 0 | 0 | 0 |
| cs.IT | 0 | 2 | 1 | 3 | 3 | 1 | 1 | 74 | 0 | 1 | 5 |
| cs.DS | 0 | 0 | 3 | 0 | 1 | 1 | 2 | 2 | 77 | 1 | 4 |
| cs.NE | 0 | 8 | 5 | 1 | 0 | 4 | 0 | 1 | 1 | 69 | 2 |
| math.ST | 0 | 1 | 0 | 2 | 0 | 2 | 0 | 0 | 3 | 1 | 81 |
