from longdoc_transformer_classifier.benchmark_config import (
    METHOD_STRUCTURAL_TAKEAWAYS,
    QUICK_SAMPLE_CONFIG,
    REPORT_METHOD_ORDER,
    TINY_TRANSFORMER_MODEL,
)


def test_benchmark_config_has_expected_methods_and_settings():
    assert QUICK_SAMPLE_CONFIG.baseline_train_samples == 100
    assert QUICK_SAMPLE_CONFIG.baseline_test_samples == 50
    assert QUICK_SAMPLE_CONFIG.summary_train_samples == 30
    assert TINY_TRANSFORMER_MODEL == "prajjwal1/bert-tiny"
    assert "baseline_arxiv" in REPORT_METHOD_ORDER
    assert "chunked_transformer_arxiv" in REPORT_METHOD_ORDER
    assert "summary_classifier" in METHOD_STRUCTURAL_TAKEAWAYS

