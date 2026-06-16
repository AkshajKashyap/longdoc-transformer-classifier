# arxiv Baseline Report

## Model

TF-IDF features with Logistic Regression.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Train samples: 1000
- Test samples: 500
- Max TF-IDF features: 50000
- N-gram range: 1-2
- Random state: 42

## Metrics

- Accuracy: 0.7880
- Macro-F1: 0.7826

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.9684 |
| cs.CV | 0.8478 |
| cs.AI | 0.4935 |
| cs.SY | 0.7619 |
| math.GR | 0.8889 |
| cs.CE | 0.7755 |
| cs.PL | 0.8800 |
| cs.IT | 0.8193 |
| cs.DS | 0.7640 |
| cs.NE | 0.6374 |
| math.ST | 0.7723 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 46 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 39 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 1 |
| cs.AI | 0 | 2 | 19 | 1 | 0 | 1 | 6 | 0 | 3 | 12 | 2 |
| cs.SY | 1 | 1 | 2 | 32 | 0 | 2 | 1 | 1 | 2 | 2 | 2 |
| math.GR | 2 | 0 | 0 | 0 | 40 | 2 | 1 | 0 | 0 | 0 | 1 |
| cs.CE | 0 | 0 | 2 | 0 | 1 | 38 | 1 | 1 | 0 | 0 | 2 |
| cs.PL | 0 | 0 | 0 | 0 | 0 | 1 | 44 | 0 | 0 | 0 | 0 |
| cs.IT | 0 | 0 | 0 | 3 | 3 | 1 | 0 | 34 | 0 | 1 | 3 |
| cs.DS | 0 | 0 | 1 | 0 | 0 | 2 | 2 | 2 | 34 | 0 | 4 |
| cs.NE | 0 | 4 | 5 | 0 | 0 | 3 | 0 | 0 | 2 | 29 | 2 |
| math.ST | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 3 | 1 | 39 |
