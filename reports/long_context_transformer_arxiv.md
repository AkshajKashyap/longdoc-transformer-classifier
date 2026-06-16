# arxiv Long-Context Transformer Report

> This baseline uses a sequence classifier capable of longer context windows. It is still bounded by `max_length` and available compute.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Model name: `allenai/longformer-base-4096`
- Max length: 1024
- Max train samples: 20
- Max test samples: 10
- Actual train size: 20
- Actual test size: 10
- Epochs: 1
- Batch size: 1
- Gradient accumulation steps: 1
- Freeze encoder: True
- Trainable parameters: 599051
- Total parameters: 148667915
- Learning rate: 5e-05
- Device: cuda

## Length Diagnostics

- Mean whitespace tokens: 13779.4
- Median whitespace tokens: 10702.0
- P95 whitespace tokens: 33787.6
- Documents above max length: 10 (100.00%)

## Metrics

- Accuracy: 0.1000
- Macro-F1: 0.0202

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.0000 |
| cs.CV | 0.0000 |
| cs.AI | 0.0000 |
| cs.SY | 0.0000 |
| math.GR | 0.0000 |
| cs.CE | 0.2222 |
| cs.PL | 0.0000 |
| cs.IT | 0.0000 |
| cs.DS | 0.0000 |
| cs.NE | 0.0000 |
| math.ST | 0.0000 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.AI | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.SY | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| math.GR | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.CE | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.PL | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.IT | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.DS | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.NE | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| math.ST | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Limitations

- Long-context transformers still have a maximum context window.
- max_length=1024 smoke runs do not use the full 4096-token Longformer capacity.
- Freezing the encoder limits adaptation.
- Tiny sample results are not final model rankings.
- Long-context models are more expensive than TF-IDF or chunk heuristics.
