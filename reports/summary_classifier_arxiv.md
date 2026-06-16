# arxiv Summary-First Classifier Report

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`

## Configuration

- Summarizer model: `sshleifer/distilbart-cnn-12-6`
- Classifier: `tfidf`
- Classifier model: `n/a`
- Max train samples: 30
- Max test samples: 15
- Actual train size: 30
- Actual test size: 15
- Summary max input tokens: 1024
- Summary max new tokens: 120
- Summary min new tokens: 30
- Summary num beams: 2
- Summary batch size: 2
- Classifier device: cpu

## Cache

- Train cache path: `data/processed/summaries/arxiv_train_sshleifer_distilbart-cnn-12-6_848e2f2c61f6e3d1.jsonl`
- Test cache path: `data/processed/summaries/arxiv_test_sshleifer_distilbart-cnn-12-6_914a0f00ae0d436c.jsonl`
- Train cache status: `loaded`
- Test cache status: `loaded`

## Summary Statistics

- Average original word count: 10282.2
- Average summarizer input tokens: 1024.0
- Average summarizer input words: 646.8
- Average summary word count: 49.0
- Average compression ratio: 0.0079
- Documents where summarizer input was shorter than original: 45 (100.00%)

## Metrics

- Accuracy: 0.2000
- Macro-F1: 0.1515

## Per-Class F1

| Class | F1 |
| --- | ---: |
| math.AC | 0.0000 |
| cs.CV | 0.6667 |
| cs.AI | 0.0000 |
| cs.SY | 0.5000 |
| math.GR | 0.5000 |
| cs.CE | 0.0000 |
| cs.PL | 0.0000 |
| cs.IT | 0.0000 |
| cs.DS | 0.0000 |
| cs.NE | 0.0000 |
| math.ST | 0.0000 |

## Confusion Matrix

| true/pred | math.AC | cs.CV | cs.AI | cs.SY | math.GR | cs.CE | cs.PL | cs.IT | cs.DS | cs.NE | math.ST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math.AC | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| cs.CV | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| cs.AI | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| cs.SY | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| math.GR | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.CE | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.PL | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.IT | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.DS | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cs.NE | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| math.ST | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Limitations

- This is not full summarizer fine-tuning.
- The summarizer may only see the first part of very long documents.
- Summaries may remove class-discriminative details.
- Summary-first classification trades recall for compression.
