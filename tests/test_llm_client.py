from __future__ import annotations

from cm_en_pipeline.llm_client import build_cache_key, normalize_text_for_cache, parse_json_text


def test_parse_json_text_handles_fenced_payload():
    parsed, error = parse_json_text(
        """```json
        {"translation_en": "The white horse is not a horse.", "notes": "", "uncertainty": "low"}
        ```"""
    )

    assert error is None
    assert parsed is not None
    assert parsed["translation_en"] == "The white horse is not a horse."


def test_cache_key_is_stable_for_normalized_input():
    left = build_cache_key(
        provider="deepseek",
        model="deepseek-chat",
        task_type="forward",
        normalized_input=normalize_text_for_cache("  天下  大乱 "),
        prompt_version="v1",
    )
    right = build_cache_key(
        provider="deepseek",
        model="deepseek-chat",
        task_type="forward",
        normalized_input=normalize_text_for_cache("天下 大乱"),
        prompt_version="v1",
    )

    assert left == right

