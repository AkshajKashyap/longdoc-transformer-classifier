# arxiv Chunked Transformer Report

> Chunk labels are weak labels inherited from the parent document. A chunk may not contain the evidence that justifies the document label.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Model name: `prajjwal1/bert-tiny`
- Chunk size: 220
- Chunk overlap: 40
- Max length: 256
- Max chunks per document: 8
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
| train | 100 | 800 | 8.00 | 196 | 100.00% | 100.00% |
| test | 50 | 398 | 7.96 | 190 | 100.00% | 98.00% |

## Document-Level Metrics

- Accuracy: 0.2600
- Macro-F1: 0.1663

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.3636 |
| cs.CV | 0.2500 |
| cs.AI | 0.0000 |
| cs.SY | 0.0000 |
| math.GR | 0.5000 |
| cs.CE | 0.0000 |
| cs.PL | 0.0000 |
| cs.IT | 0.0000 |
| cs.DS | 0.4000 |
| cs.NE | 0.3158 |
| math.ST | 0.0000 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 |
| cs.CV | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 |
| cs.AI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| cs.SY | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| math.GR | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CE | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| cs.PL | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| cs.IT | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| cs.DS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| cs.NE | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| math.ST | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
