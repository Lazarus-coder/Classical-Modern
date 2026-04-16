from __future__ import annotations

from typing import Any

from .dataset import DatasetRecord
from .io_utils import utc_now_iso
from .llm_client import LLMCallResult, OpenAICompatibleChatClient
from .prompts import (
    BACKWARD_PROMPT_VERSION,
    BACKWARD_SYSTEM_PROMPT,
    FORWARD_PROMPT_VERSION,
    FORWARD_SYSTEM_PROMPT,
    build_backward_user_prompt,
    build_forward_user_prompt,
)

PIPELINE_PROMPT_VERSION = f"{FORWARD_PROMPT_VERSION}|{BACKWARD_PROMPT_VERSION}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_uncertainty(value: Any) -> tuple[str, str | None]:
    uncertainty = _clean_text(value).lower()
    if uncertainty in {"low", "medium", "high"}:
        return uncertainty, None
    if not uncertainty:
        return "", "Missing uncertainty value."
    return uncertainty, f"Unexpected uncertainty value: {uncertainty}"


def _skipped_result(reason: str) -> LLMCallResult:
    return LLMCallResult(
        provider="",
        model="",
        task_type="backward",
        prompt_version=BACKWARD_PROMPT_VERSION,
        cache_key="",
        status="skipped",
        raw_text="",
        parsed_json=None,
        usage=None,
        error_message=reason,
        parse_error=None,
        from_cache=False,
        response_id=None,
    )


def translate_record(
    record: DatasetRecord,
    client: OpenAICompatibleChatClient,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    forward = client.request_json(
        task_type="forward",
        prompt_version=FORWARD_PROMPT_VERSION,
        source_text=record.target_modern_zh,
        system_prompt=FORWARD_SYSTEM_PROMPT,
        user_prompt=build_forward_user_prompt(record.target_modern_zh),
        expected_keys=("translation_en", "notes", "uncertainty"),
        max_tokens=256,
    )

    translation_en = ""
    forward_notes = ""
    forward_uncertainty = ""
    forward_validation_error = None
    if forward.parsed_json:
        translation_en = _clean_text(forward.parsed_json.get("translation_en"))
        forward_notes = _clean_text(forward.parsed_json.get("notes"))
        forward_uncertainty, uncertainty_error = _normalize_uncertainty(
            forward.parsed_json.get("uncertainty")
        )
        if uncertainty_error:
            forward_validation_error = uncertainty_error

    if forward.status == "success" and not translation_en:
        forward_validation_error = "translation_en is empty."
    if forward_validation_error:
        forward = LLMCallResult(
            provider=forward.provider,
            model=forward.model,
            task_type=forward.task_type,
            prompt_version=forward.prompt_version,
            cache_key=forward.cache_key,
            status="validation_error",
            raw_text=forward.raw_text,
            parsed_json=forward.parsed_json,
            usage=forward.usage,
            error_message=forward_validation_error,
            parse_error=forward.parse_error,
            from_cache=forward.from_cache,
            response_id=forward.response_id,
        )

    if forward.status == "success":
        backward = client.request_json(
            task_type="backward",
            prompt_version=BACKWARD_PROMPT_VERSION,
            source_text=translation_en,
            system_prompt=BACKWARD_SYSTEM_PROMPT,
            user_prompt=build_backward_user_prompt(translation_en),
            expected_keys=("back_translation_modern_zh",),
            max_tokens=160,
        )
    else:
        backward = _skipped_result("Forward translation did not succeed.")

    back_translation_modern_zh = ""
    if backward.parsed_json:
        back_translation_modern_zh = _clean_text(
            backward.parsed_json.get("back_translation_modern_zh")
        )
        if backward.status == "success" and not back_translation_modern_zh:
            backward = LLMCallResult(
                provider=backward.provider,
                model=backward.model,
                task_type=backward.task_type,
                prompt_version=backward.prompt_version,
                cache_key=backward.cache_key,
                status="validation_error",
                raw_text=backward.raw_text,
                parsed_json=backward.parsed_json,
                usage=backward.usage,
                error_message="back_translation_modern_zh is empty.",
                parse_error=backward.parse_error,
                from_cache=backward.from_cache,
                response_id=backward.response_id,
            )

    overall_status = (
        "success" if forward.status == "success" and backward.status == "success" else "failed"
    )
    return {
        "record_id": record.record_id,
        "book": record.book,
        "chapter_path": record.chapter_path,
        "line_index": record.line_index,
        "source_classical_zh": record.source_classical_zh,
        "target_modern_zh": record.target_modern_zh,
        "translation_en": translation_en,
        "back_translation_modern_zh": back_translation_modern_zh,
        "forward_notes": forward_notes,
        "forward_uncertainty": forward_uncertainty,
        "provider": client.config.provider,
        "model": client.config.model,
        "prompt_version": PIPELINE_PROMPT_VERSION,
        "forward_prompt_version": FORWARD_PROMPT_VERSION,
        "backward_prompt_version": BACKWARD_PROMPT_VERSION,
        "created_at_utc": started_at,
        "updated_at_utc": utc_now_iso(),
        "forward_status": forward.status,
        "backward_status": backward.status,
        "overall_status": overall_status,
        "forward_error": forward.error_message,
        "backward_error": backward.error_message,
        "forward_parse_error": forward.parse_error,
        "backward_parse_error": backward.parse_error,
        "forward_cache_hit": forward.from_cache,
        "backward_cache_hit": backward.from_cache,
        "forward_usage": forward.usage,
        "backward_usage": backward.usage,
        "forward_response_text": forward.raw_text,
        "backward_response_text": backward.raw_text,
        "forward_response_id": forward.response_id,
        "backward_response_id": backward.response_id,
    }
