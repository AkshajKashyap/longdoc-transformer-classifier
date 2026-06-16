from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_BENCHMARK_DATASET = "arxiv"
DEFAULT_REPORTS_DIR = Path("reports")
TINY_TRANSFORMER_MODEL = "prajjwal1/bert-tiny"
DEFAULT_CHUNK_SELECTION = "first_k"


@dataclass(frozen=True)
class BenchmarkSampleConfig:
    analysis_train_samples: int
    analysis_test_samples: int
    baseline_train_samples: int
    baseline_test_samples: int
    transformer_train_samples: int
    transformer_test_samples: int
    summary_train_samples: int
    summary_test_samples: int


QUICK_SAMPLE_CONFIG = BenchmarkSampleConfig(
    analysis_train_samples=100,
    analysis_test_samples=50,
    baseline_train_samples=100,
    baseline_test_samples=50,
    transformer_train_samples=100,
    transformer_test_samples=50,
    summary_train_samples=30,
    summary_test_samples=15,
)

STANDARD_SAMPLE_CONFIG = BenchmarkSampleConfig(
    analysis_train_samples=1_000,
    analysis_test_samples=500,
    baseline_train_samples=1_000,
    baseline_test_samples=500,
    transformer_train_samples=100,
    transformer_test_samples=50,
    summary_train_samples=30,
    summary_test_samples=15,
)

REPORT_METHOD_ORDER = [
    "baseline_ag_news",
    "baseline_arxiv",
    "truncated_transformer_ag_news",
    "truncated_transformer_arxiv",
    "chunked_transformer_ag_news",
    "chunked_transformer_arxiv",
    "summary_classifier_ag_news",
    "summary_classifier_arxiv",
]

METHOD_FAMILY_LABELS = {
    "tfidf_baseline": "TF-IDF + Logistic Regression",
    "truncated_transformer": "Truncated Transformer",
    "chunked_transformer": "Chunked Transformer",
    "summary_classifier": "Summary-First Classifier",
    "unknown": "Unknown",
}

METHOD_LIMITATIONS = {
    "tfidf_baseline": "No neural long-context reasoning; relies on lexical signals.",
    "truncated_transformer": "Only the first fixed token window can affect predictions.",
    "chunked_transformer": "Chunks inherit weak document labels and may lack class evidence.",
    "summary_classifier": "Summaries may discard details needed for classification.",
    "unknown": "No method-specific limitation is registered.",
}

METHOD_STRUCTURAL_TAKEAWAYS = {
    "tfidf_baseline": "Strong lexical baseline, no neural long-context reasoning.",
    "truncated_transformer": "Shows what happens when long documents are clipped.",
    "chunked_transformer": "Structurally sees more text, but uses weak inherited chunk labels.",
    "summary_classifier": "Compresses long documents, but may discard class evidence.",
    "unknown": "Useful only as an unclassified report artifact.",
}
