from __future__ import annotations

from cm_en_pipeline.config import FilterConfig
from cm_en_pipeline.filtering import evaluate_record
from cm_en_pipeline.metrics import MetricsScorer


def test_metrics_score_identical_strings_without_embeddings():
    scorer = MetricsScorer(FilterConfig(enable_embeddings=False))
    result = scorer.score_pair("天下大乱", "天下大乱")

    assert result.chrf is not None and result.chrf > 90
    assert result.edit_similarity == 1.0
    assert result.length_ratio == 1.0


def test_filter_rejects_meta_text():
    config = FilterConfig(enable_embeddings=False)
    record = {
        "overall_status": "success",
        "forward_status": "success",
        "backward_status": "success",
        "target_modern_zh": "天下大乱",
        "translation_en": "Here is the translation: chaos under heaven.",
        "back_translation_modern_zh": "天下大乱",
        "chrf": 100.0,
        "bleu": 100.0,
        "edit_similarity": 1.0,
        "length_ratio": 1.0,
        "embedding_similarity": None,
        "metric_error": None,
    }

    passed, reasons = evaluate_record(record, config)

    assert not passed
    assert "translation_en_contains_meta_or_refusal" in reasons

