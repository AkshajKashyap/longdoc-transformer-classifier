from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "ag_news"
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    cache_dir: Path = Path("data/hf_cache")
    text_field: str | None = None
    label_field: str | None = None


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class BaselineModelConfig:
    max_iter: int = 1_000
    random_state: int = DEFAULT_RANDOM_STATE
    solver: str = "lbfgs"


@dataclass(frozen=True)
class ReportConfig:
    reports_dir: Path = Path("reports")
