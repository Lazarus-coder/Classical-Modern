from __future__ import annotations

import re
from typing import Any

from .config import FilterConfig

REFUSAL_MARKERS = (
    "sorry",
    "i cannot",
    "i can't",
    "i’m unable",
    "i am unable",
    "cannot assist",
    "抱歉",
    "无法",
    "不能帮助",
)
META_MARKERS = (
    "here is the translation",
    "translation:",
    "back-translation:",
    "以下是翻译",
    "翻译如下",
    "返回json",
    "translation_en",
    "back_translation_modern_zh",
    "```",
)
REPEATED_FRAGMENT_RE = re.compile(r"(.{1,8})\1{4,}")
REPEATED_CHAR_RE = re.compile(r"(.)\1{7,}")


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _looks_repetitive(text: str) -> bool:
    if not text:
        return False
    if REPEATED_CHAR_RE.search(text):
        return True
    if REPEATED_FRAGMENT_RE.search(text):
        return True
    if len(text) >= 30 and len(set(text)) / len(text) < 0.1:
        return True
    return False


def evaluate_record(record: dict[str, Any], config: FilterConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    target_modern_zh = str(record.get("target_modern_zh") or "")
    translation_en = str(record.get("translation_en") or "")
    back_translation_modern_zh = str(record.get("back_translation_modern_zh") or "")

    if record.get("overall_status") != "success":
        reasons.append(f"overall_status:{record.get('overall_status', 'missing')}")
    if record.get("forward_status") != "success":
        reasons.append(f"forward_status:{record.get('forward_status', 'missing')}")
    if record.get("backward_status") != "success":
        reasons.append(f"backward_status:{record.get('backward_status', 'missing')}")

    if not translation_en.strip():
        reasons.append("empty_translation_en")
    if not back_translation_modern_zh.strip():
        reasons.append("empty_back_translation_modern_zh")

    if translation_en and _contains_marker(translation_en, REFUSAL_MARKERS + META_MARKERS):
        reasons.append("translation_en_contains_meta_or_refusal")
    if back_translation_modern_zh and _contains_marker(
        back_translation_modern_zh, REFUSAL_MARKERS + META_MARKERS
    ):
        reasons.append("back_translation_contains_meta_or_refusal")

    if _looks_repetitive(translation_en):
        reasons.append("translation_en_repetitive")
    if _looks_repetitive(back_translation_modern_zh):
        reasons.append("back_translation_repetitive")

    if target_modern_zh and translation_en:
        forward_ratio = len(translation_en) / max(len(target_modern_zh), 1)
        if forward_ratio > config.max_forward_expansion_ratio:
            reasons.append("translation_en_suspiciously_long")

    if target_modern_zh and back_translation_modern_zh:
        backward_ratio = len(back_translation_modern_zh) / max(len(target_modern_zh), 1)
        if backward_ratio > config.max_backward_expansion_ratio:
            reasons.append("back_translation_suspiciously_long")

    if record.get("metric_error"):
        reasons.append("metric_error")

    chrf = record.get("chrf")
    bleu = record.get("bleu")
    edit_similarity = record.get("edit_similarity")
    length_ratio = record.get("length_ratio")
    embedding_similarity = record.get("embedding_similarity")

    if chrf is None or float(chrf) < config.min_chrf:
        reasons.append("chrf_below_threshold")
    if bleu is None or float(bleu) < config.min_bleu:
        reasons.append("bleu_below_threshold")
    if edit_similarity is None or float(edit_similarity) < config.min_edit_similarity:
        reasons.append("edit_similarity_below_threshold")
    if length_ratio is None or not (
        config.min_length_ratio <= float(length_ratio) <= config.max_length_ratio
    ):
        reasons.append("length_ratio_out_of_range")
    if config.enable_embeddings:
        if embedding_similarity is None:
            reasons.append("embedding_similarity_missing")
        elif float(embedding_similarity) < config.min_embedding_similarity:
            reasons.append("embedding_similarity_below_threshold")

    seen: set[str] = set()
    deduped_reasons = [reason for reason in reasons if not (reason in seen or seen.add(reason))]
    return not deduped_reasons, deduped_reasons


def build_filtered_export(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "source_classical_zh": record.get("source_classical_zh"),
        "target_modern_zh": record.get("target_modern_zh"),
        "translation_en": record.get("translation_en"),
        "back_translation_modern_zh": record.get("back_translation_modern_zh"),
        "book": record.get("book"),
        "chapter_path": record.get("chapter_path"),
        "line_index": record.get("line_index"),
        "chrf": record.get("chrf"),
        "bleu": record.get("bleu"),
        "edit_similarity": record.get("edit_similarity"),
        "length_ratio": record.get("length_ratio"),
        "embedding_similarity": record.get("embedding_similarity"),
        "filter_reason": "; ".join(record.get("filter_reason") or []),
    }

