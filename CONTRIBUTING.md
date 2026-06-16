# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Checks

```bash
make test
make lint
```

## Artifact Hygiene

Do not commit datasets, Hugging Face caches, generated summaries, checkpoints, model weights, or other
large artifacts. The intended cache and artifact directories are ignored in `.gitignore`.
