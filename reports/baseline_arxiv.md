# arxiv Baseline Report

## Model

TF-IDF features with Logistic Regression.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Train samples: 100
- Test samples: 50
- Max TF-IDF features: 50000
- N-gram range: 1-2
- Random state: 42

## Metrics

- Accuracy: 0.5600
- Macro-F1: 0.4972

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.7692 |
| cs.CV | 0.6667 |
| cs.AI | 0.0000 |
| cs.SY | 0.6000 |
| math.GR | 0.7692 |
| cs.CE | 0.5000 |
| cs.PL | 0.6667 |
| cs.IT | 0.0000 |
| cs.DS | 0.2857 |
| cs.NE | 0.6667 |
| math.ST | 0.5455 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| cs.AI | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 2 | 0 |
| cs.SY | 0 | 0 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| math.GR | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CE | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 2 |
| cs.PL | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| cs.IT | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.DS | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 2 |
| cs.NE | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| math.ST | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 |
