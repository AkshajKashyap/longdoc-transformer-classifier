# arxiv Truncated Transformer Report

> This is a truncation baseline, not a long-document solution. Each document is tokenized once and only the first `max_length` tokens can affect the prediction.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Model name: `prajjwal1/bert-tiny`
- Max length: 512
- Max train samples: 100
- Max test samples: 50
- Actual train size: 100
- Actual test size: 50
- Epochs: 1
- Batch size: 4
- Learning rate: 5e-05
- Device: cpu

## Truncation Diagnostics

- Mean whitespace tokens: 8909.1
- Median whitespace tokens: 7137.5
- P95 whitespace tokens: 20540.2
- Documents above max length: 50 (100.00%)

## Metrics

- Accuracy: 0.1600
- Macro-F1: 0.0782

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.0000 |
| cs.CV | 0.0000 |
| cs.AI | 0.2857 |
| cs.SY | 0.0000 |
| math.GR | 0.3750 |
| cs.CE | 0.0000 |
| cs.PL | 0.0000 |
| cs.IT | 0.0000 |
| cs.DS | 0.0000 |
| cs.NE | 0.2000 |
| math.ST | 0.0000 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| cs.AI | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| cs.SY | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 0 |
| math.GR | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 2 | 0 |
| cs.CE | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| cs.PL | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| cs.IT | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| cs.DS | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| cs.NE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| math.ST | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
