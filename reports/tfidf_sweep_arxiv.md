# arxiv TF-IDF Sweep

> This is still a classical lexical baseline, not neural long-context reasoning.

## Dataset

- Dataset name: `arxiv`
- Hugging Face dataset: `ccdv/arxiv-classification`
- Train samples: 3000
- Test samples: 1000

## Fixed Settings

- Min document frequency: 2
- Max document frequency: 0.95
- Solver: `lbfgs`
- Max iterations: 1000

## Best Setting By Macro-F1

- Accuracy: 0.8180
- Macro-F1: 0.8155
- Settings: `{'max_features': 100000, 'ngram_range': [1, 2], 'sublinear_tf': True, 'class_weight': None}`

## Results

| Max Features | N-gram Range | Sublinear TF | Class Weight | Accuracy | Macro-F1 |
| ---: | --- | --- | --- | ---: | ---: |
| 50000 | 1-1 | False | none | 0.7940 | 0.7917 |
| 50000 | 1-2 | False | none | 0.8000 | 0.7971 |
| 100000 | 1-2 | True | none | 0.8180 | 0.8155 |
| 100000 | 1-2 | True | balanced | 0.8180 | 0.8155 |
