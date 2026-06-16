# ag_news Truncated Transformer Report

> This is a truncation baseline, not a long-document solution. Each document is tokenized once and only the first `max_length` tokens can affect the prediction.

## Dataset

- Dataset name: `ag_news`
- Hugging Face dataset: `fancyzhx/ag_news`
- Text field: `text`
- Label field: `label`

## Configuration

- Model name: `prajjwal1/bert-tiny`
- Max length: 128
- Max train samples: 100
- Max test samples: 50
- Actual train size: 100
- Actual test size: 50
- Epochs: 1
- Batch size: 8
- Learning rate: 5e-05
- Device: cpu

## Truncation Diagnostics

- Mean whitespace tokens: 47.3
- Median whitespace tokens: 44.0
- P95 whitespace tokens: 88.1
- Documents above max length: 1 (2.00%)

## Metrics

- Accuracy: 0.2800
- Macro-F1: 0.2605

## Per-Class F1

| Class | F1 |
| --- | ---: |
| World | 0.0000 |
| Sports | 0.3871 |
| Business | 0.2105 |
| Sci/Tech | 0.4444 |

## Confusion Matrix

| true/pred | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| World | 0 | 3 | 10 | 0 |
| Sports | 0 | 6 | 7 | 0 |
| Business | 0 | 6 | 4 | 2 |
| Sci/Tech | 0 | 3 | 5 | 4 |
