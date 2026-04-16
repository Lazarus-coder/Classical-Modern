from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from collections import Counter
from typing import Any

from .config import FilterConfig

try:  # pragma: no cover - optional dependency path
    from rapidfuzz.distance import Levenshtein as RapidFuzzLevenshtein
except ImportError:  # pragma: no cover - exercised via fallback logic
    RapidFuzzLevenshtein = None

try:  # pragma: no cover - optional dependency path
    from sacrebleu.metrics import BLEU as SacreBLEU
    from sacrebleu.metrics import CHRF as SacreCHRF
except ImportError:  # pragma: no cover - exercised via fallback logic
    SacreBLEU = None
    SacreCHRF = None


@dataclass
class MetricResult:
    chrf: float | None
    bleu: float | None
    edit_similarity: float | None
    length_ratio: float | None
    embedding_similarity: float | None
    metric_error: str | None = None


@dataclass
class _SimpleScore:
    score: float


class _FallbackBLEU:
    def sentence_score(self, hypothesis: str, references: list[str]) -> _SimpleScore:
        reference = references[0] if references else ""
        if not hypothesis and not reference:
            return _SimpleScore(100.0)
        hyp_tokens = list(hypothesis)
        ref_tokens = list(reference)
        overlap = sum((Counter(hyp_tokens) & Counter(ref_tokens)).values())
        precision = overlap / max(len(hyp_tokens), 1)
        brevity_penalty = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
        return _SimpleScore(100.0 * precision * brevity_penalty)


class _FallbackCHRF:
    def sentence_score(self, hypothesis: str, references: list[str]) -> _SimpleScore:
        reference = references[0] if references else ""
        if not hypothesis and not reference:
            return _SimpleScore(100.0)
        hyp_chars = Counter(hypothesis)
        ref_chars = Counter(reference)
        overlap = sum((hyp_chars & ref_chars).values())
        precision = overlap / max(len(hypothesis), 1)
        recall = overlap / max(len(reference), 1)
        if precision + recall == 0:
            return _SimpleScore(0.0)
        f1 = 2 * precision * recall / (precision + recall)
        return _SimpleScore(100.0 * f1)


def _normalized_edit_similarity(left: str, right: str) -> float:
    if RapidFuzzLevenshtein is not None:
        return float(RapidFuzzLevenshtein.normalized_similarity(left, right))
    if left == right:
        return 1.0
    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current_row[right_index - 1] + 1
            deletion = previous_row[right_index] + 1
            substitution = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    distance = previous_row[-1]
    return 1.0 - (distance / max(len(left), len(right), 1))


class EmbeddingScorer:
    def __init__(self, config: FilterConfig, logger: logging.Logger | None = None) -> None:
        self.enabled = False
        self.model = None
        self.logger = logger or logging.getLogger(__name__)
        if not config.enable_embeddings:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.logger.warning(
                "Embedding similarity disabled because sentence-transformers is not installed."
            )
            return
        try:
            self.model = SentenceTransformer(
                config.embedding_model_name,
                local_files_only=config.embedding_local_files_only,
            )
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on local model availability
            self.logger.warning("Embedding similarity disabled: %s", exc)

    def similarity(self, left: str, right: str) -> float | None:
        if not self.enabled or self.model is None:
            return None
        embeddings = self.model.encode([left, right], normalize_embeddings=True)
        return float((embeddings[0] * embeddings[1]).sum())


class MetricsScorer:
    def __init__(
        self,
        config: FilterConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.bleu_metric = (
            SacreBLEU(tokenize="zh", effective_order=True)
            if SacreBLEU is not None
            else _FallbackBLEU()
        )
        self.chrf_metric = SacreCHRF() if SacreCHRF is not None else _FallbackCHRF()
        self.embedding_scorer = EmbeddingScorer(config, logger=logger)

    def score_pair(self, reference_text: str, hypothesis_text: str) -> MetricResult:
        reference = reference_text or ""
        hypothesis = hypothesis_text or ""
        try:
            chrf = float(self.chrf_metric.sentence_score(hypothesis, [reference]).score)
            bleu = float(self.bleu_metric.sentence_score(hypothesis, [reference]).score)
            edit_similarity = _normalized_edit_similarity(reference, hypothesis)
            length_ratio = len(hypothesis) / max(len(reference), 1)
            embedding_similarity = self.embedding_scorer.similarity(reference, hypothesis)
            return MetricResult(
                chrf=chrf,
                bleu=bleu,
                edit_similarity=edit_similarity,
                length_ratio=length_ratio,
                embedding_similarity=embedding_similarity,
            )
        except Exception as exc:
            return MetricResult(
                chrf=None,
                bleu=None,
                edit_similarity=None,
                length_ratio=None,
                embedding_similarity=None,
                metric_error=str(exc),
            )

    def score_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        metric_result = self.score_pair(
            str(raw_record.get("target_modern_zh") or ""),
            str(raw_record.get("back_translation_modern_zh") or ""),
        )
        return {**raw_record, **asdict(metric_result)}
