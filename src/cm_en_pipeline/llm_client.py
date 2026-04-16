from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from .config import ProviderConfig
from .io_utils import SQLiteCache

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_text_for_cache(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_cache_key(
    *,
    provider: str,
    model: str,
    task_type: str,
    normalized_input: str,
    prompt_version: str,
    request_settings: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "normalized_input": normalized_input,
            "prompt_version": prompt_version,
            "request_settings": request_settings or {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_json_text(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = raw_text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        fenced = re.sub(r"^```(?:json)?\s*", "", stripped)
        fenced = re.sub(r"\s*```$", "", fenced)
        candidates.append(fenced.strip())

    match = JSON_OBJECT_RE.search(stripped)
    if match:
        candidates.append(match.group(0).strip())

    seen: set[str] = set()
    last_error = "JSON payload missing."
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
            last_error = "Parsed JSON is not an object."
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error


def _extract_response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _validate_expected_keys(
    parsed_json: dict[str, Any] | None,
    expected_keys: Iterable[str],
) -> tuple[str, str | None]:
    if parsed_json is None:
        return "parse_error", "Model output could not be parsed as JSON."
    missing_keys = [key for key in expected_keys if key not in parsed_json]
    if missing_keys:
        return "validation_error", f"Missing expected keys: {', '.join(missing_keys)}"
    return "success", None


@dataclass
class LLMCallResult:
    provider: str
    model: str
    task_type: str
    prompt_version: str
    cache_key: str
    status: str
    raw_text: str
    parsed_json: dict[str, Any] | None
    usage: dict[str, Any] | None
    error_message: str | None
    parse_error: str | None
    from_cache: bool
    response_id: str | None


class OpenAICompatibleChatClient:
    def __init__(self, config: ProviderConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache
        self._rate_limit_lock = threading.Lock()
        self._last_request_at = 0.0

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _wait_for_rate_limit_window(self) -> None:
        if self.config.request_interval_sec <= 0:
            return
        with self._rate_limit_lock:
            now = time.monotonic()
            remaining = self.config.request_interval_sec - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def request_json(
        self,
        *,
        task_type: str,
        prompt_version: str,
        source_text: str,
        system_prompt: str,
        user_prompt: str,
        expected_keys: Iterable[str],
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        normalized_input = normalize_text_for_cache(source_text)
        request_settings = {
            "max_tokens": max_tokens,
            "use_json_mode": self.config.use_json_mode,
            "temperature": 0.0,
        }
        cache_key = build_cache_key(
            provider=self.config.provider,
            model=self.config.model,
            task_type=task_type,
            normalized_input=normalized_input,
            prompt_version=prompt_version,
            request_settings=request_settings,
        )
        cached = self.cache.get(cache_key)
        if cached and cached.get("status") == "success":
            return LLMCallResult(
                provider=self.config.provider,
                model=self.config.model,
                task_type=task_type,
                prompt_version=prompt_version,
                cache_key=cache_key,
                status=str(cached["status"]),
                raw_text=str(cached.get("response_text") or ""),
                parsed_json=cached.get("parsed_json"),
                usage=cached.get("usage_json"),
                error_message=cached.get("error_message"),
                parse_error=None,
                from_cache=True,
                response_id=None,
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.config.use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        attempts = max(self.config.retry_max_attempts, 1)
        last_error = "Unknown API error."
        for attempt in range(1, attempts + 1):
            try:
                self._wait_for_rate_limit_window()
                response = requests.post(
                    self._endpoint(),
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout_sec,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self._compute_retry_delay(attempt)
                    )
                    last_error = f"Retryable HTTP {response.status_code}: {response.text[:500]}"
                    if attempt == attempts:
                        return self._build_failure_result(
                            task_type=task_type,
                            prompt_version=prompt_version,
                            cache_key=cache_key,
                            raw_text=response.text[:2000],
                            error_message=last_error,
                            status="api_error",
                        )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                response_json = response.json()
                raw_text = _extract_response_text(response_json)
                parsed_json, parse_error = parse_json_text(raw_text)
                status, validation_error = _validate_expected_keys(parsed_json, expected_keys)
                usage = response_json.get("usage")
                error_message = parse_error or validation_error

                self.cache.put(
                    cache_key=cache_key,
                    provider=self.config.provider,
                    model=self.config.model,
                    task_type=task_type,
                    prompt_version=prompt_version,
                    normalized_input=normalized_input,
                    request_json=payload,
                    response_text=raw_text,
                    response_json=response_json,
                    parsed_json=parsed_json,
                    usage_json=usage if isinstance(usage, dict) else None,
                    status=status,
                    error_message=error_message,
                )
                return LLMCallResult(
                    provider=self.config.provider,
                    model=self.config.model,
                    task_type=task_type,
                    prompt_version=prompt_version,
                    cache_key=cache_key,
                    status=status,
                    raw_text=raw_text,
                    parsed_json=parsed_json,
                    usage=usage if isinstance(usage, dict) else None,
                    error_message=error_message,
                    parse_error=parse_error,
                    from_cache=False,
                    response_id=response_json.get("id"),
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt == attempts:
                    return self._build_failure_result(
                        task_type=task_type,
                        prompt_version=prompt_version,
                        cache_key=cache_key,
                        raw_text="",
                        error_message=last_error,
                        status="api_error",
                    )
                time.sleep(self._compute_retry_delay(attempt))

        return self._build_failure_result(
            task_type=task_type,
            prompt_version=prompt_version,
            cache_key=cache_key,
            raw_text="",
            error_message=last_error,
            status="api_error",
        )

    def _compute_retry_delay(self, attempt: int) -> float:
        base = self.config.retry_base_delay_sec * (2 ** (attempt - 1))
        jitter = random.uniform(0.0, self.config.retry_base_delay_sec)
        return base + jitter

    def _build_failure_result(
        self,
        *,
        task_type: str,
        prompt_version: str,
        cache_key: str,
        raw_text: str,
        error_message: str,
        status: str,
    ) -> LLMCallResult:
        return LLMCallResult(
            provider=self.config.provider,
            model=self.config.model,
            task_type=task_type,
            prompt_version=prompt_version,
            cache_key=cache_key,
            status=status,
            raw_text=raw_text,
            parsed_json=None,
            usage=None,
            error_message=error_message,
            parse_error=error_message if status == "parse_error" else None,
            from_cache=False,
            response_id=None,
        )
