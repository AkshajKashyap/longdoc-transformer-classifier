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
- Chunk selection: `idf_top_k`
- Max train samples: 100
- Max test samples: 50
- Actual train size: 100
- Actual test size: 50
- Epochs: 1
- Batch size: 8
- Learning rate: 5e-05
- Device: cpu

## Chunk Diagnostics

- Selection note: Ranks chunks by training-fitted lexical IDF scores, without using test labels.

| Split | Documents | Chunks Before | Chunks After | Avg Before | Avg After | Retained | Docs Capped | Avg Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 5420 | 800 | 54.20 | 8.00 | 14.76% | 100.00% | 22.96% |
| test | 50 | 2490 | 398 | 49.80 | 7.96 | 15.98% | 98.00% | 27.22% |

## IDF Scorer

- Vocabulary size: 28350
- Scoring mode: `average_idf`
- Fitted on chunks: 5420

## Document-Level Metrics

- Accuracy: 0.1800
- Macro-F1: 0.0848

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.4444 |
| cs.CV | 0.0000 |
| cs.AI | 0.0000 |
| cs.SY | 0.0000 |
| math.GR | 0.0000 |
| cs.CE | 0.0000 |
| cs.PL | 0.0000 |
| cs.IT | 0.0000 |
| cs.DS | 0.2222 |
| cs.NE | 0.2667 |
| math.ST | 0.0000 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| cs.CV | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 3 | 0 |
| cs.AI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 |
| cs.SY | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 0 | 0 |
| math.GR | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| cs.CE | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| cs.PL | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| cs.IT | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| cs.DS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| cs.NE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| math.ST | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
