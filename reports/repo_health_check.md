# Repo Health Check

## Tests Status

- Verified command: `pytest -q`

## Lint Status

- Verified command: `ruff check .`

## CI Status

- GitHub Actions workflow: `.github/workflows/ci.yml`
- CI runs lint and tests only; it does not download datasets or models.

## Artifact Hygiene

- Source, tests, docs, and compact reports are committed.
- Large datasets, model caches, generated summaries, and virtualenvs are ignored.

## Ignored Cache/Model/Data Directories

- `.venv/`
- `data/hf_cache/`
- `data/processed/`
- `data/raw/`
- Python and test cache directories

## Commands Verified

```bash
ruff check .
pytest -q
python -m longdoc_transformer_classifier.training.compare_reports
python -m longdoc_transformer_classifier.training.plot_comparison
```
