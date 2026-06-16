# ag_news Length Analysis

- Hugging Face dataset: `fancyzhx/ag_news`
- Text field: `text`
- Label field: `label`
- BERT limit approximation: 512 whitespace tokens

## Split Summary

### Train

- Documents: 1000
- Characters mean / median / p95 / max: 255.1 / 247.0 / 430.0 / 959
- Whitespace tokens mean / median / p95 / max: 40.9 / 39.0 / 69.0 / 134
- Documents above BERT limit: 0 (0.00%)

### Test

- Documents: 500
- Characters mean / median / p95 / max: 247.4 / 243.0 / 382.0 / 844
- Whitespace tokens mean / median / p95 / max: 39.6 / 39.0 / 60.0 / 136
- Documents above BERT limit: 0 (0.00%)

### Combined

- Documents: 1500
- Characters mean / median / p95 / max: 252.5 / 245.5 / 416.1 / 959
- Whitespace tokens mean / median / p95 / max: 40.5 / 39.0 / 67.0 / 136
- Documents above BERT limit: 0 (0.00%)

