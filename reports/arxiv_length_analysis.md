# arxiv Length Analysis

- Hugging Face dataset: `ccdv/arxiv-classification`
- Text field: `text`
- Label field: `label`
- BERT limit approximation: 512 whitespace tokens

## Split Summary

### Train

- Documents: 1000
- Characters mean / median / p95 / max: 58162.9 / 46038.0 / 120421.9 / 2344406
- Whitespace tokens mean / median / p95 / max: 10908.3 / 8257.0 / 24305.6 / 521384
- Documents above BERT limit: 1000 (100.00%)

### Test

- Documents: 500
- Characters mean / median / p95 / max: 53778.5 / 44694.0 / 113638.1 / 395898
- Whitespace tokens mean / median / p95 / max: 9973.1 / 8184.0 / 21094.3 / 71155
- Documents above BERT limit: 500 (100.00%)

### Combined

- Documents: 1500
- Characters mean / median / p95 / max: 56701.4 / 45653.0 / 118934.0 / 2344406
- Whitespace tokens mean / median / p95 / max: 10596.6 / 8233.5 / 23190.9 / 521384
- Documents above BERT limit: 1500 (100.00%)

