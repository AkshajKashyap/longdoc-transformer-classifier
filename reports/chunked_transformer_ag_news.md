# ag_news Chunked Transformer Report

> Chunk labels are weak labels inherited from the parent document. A chunk may not contain the evidence that justifies the document label.

## Dataset

- Dataset name: `ag_news`
- Hugging Face dataset: `fancyzhx/ag_news`
- Text field: `text`
- Label field: `label`

## Configuration

- Model name: `prajjwal1/bert-tiny`
- Chunk size: 100
- Chunk overlap: 20
- Max length: 128
- Max chunks per document: 4
- Aggregation: `mean_proba`
- Max train samples: 100
- Max test samples: 50
- Actual train size: 100
- Actual test size: 50
- Epochs: 1
- Batch size: 8
- Learning rate: 5e-05
- Device: cpu

## Chunk Diagnostics

| Split | Documents | Chunks | Avg Chunks/Doc | Max Before Cap | >1 Chunk | Hit Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 100 | 1.00 | 1 | 0.00% | 0.00% |
| test | 50 | 53 | 1.06 | 2 | 6.00% | 0.00% |

## Document-Level Metrics

- Accuracy: 0.2000
- Macro-F1: 0.0847

## Per-Class F1

| Class | F1 |
| --- | ---: |
| World | 0.0000 |
| Sports | 0.3390 |
| Business | 0.0000 |
| Sci/Tech | 0.0000 |

## Confusion Matrix

| true/pred | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| World | 0 | 12 | 1 | 0 |
| Sports | 0 | 10 | 3 | 0 |
| Business | 0 | 12 | 0 | 0 |
| Sci/Tech | 0 | 12 | 0 | 0 |
