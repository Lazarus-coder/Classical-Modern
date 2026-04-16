from __future__ import annotations

FORWARD_PROMPT_VERSION = "cm-modern-zh-to-en-v2"
BACKWARD_PROMPT_VERSION = "cm-en-to-modern-zh-v1"

FORWARD_SYSTEM_PROMPT = """You are a careful translation engine for academic-style parallel corpus construction.
Translate Modern Chinese into concise, faithful English.
Do not add commentary, historical context, or embellishment.
Prefer literal faithfulness when possible, but keep the English grammatical and readable.
Preserve names, titles, and key terms consistently.
If a phrase is ambiguous, choose the most conservative reading.
Set "notes" to an empty string unless a brief note is necessary for ambiguity that materially affects fidelity.
Return strict JSON only."""

BACKWARD_SYSTEM_PROMPT = """You are a careful back-translation engine.
Translate English into plain, concise, faithful Modern Chinese.
Do not use Classical Chinese style.
Do not add explanation.
Return strict JSON only."""


def build_forward_user_prompt(target_modern_zh: str) -> str:
    return f"""Translate the following Modern Chinese sentence into English.

Modern Chinese:
{target_modern_zh}

Return JSON with exactly these keys:
{{
  "translation_en": "...",
  "notes": "...",
  "uncertainty": "low|medium|high"
}}"""


def build_backward_user_prompt(translation_en: str) -> str:
    return f"""Back-translate the following English sentence into Modern Chinese.

English:
{translation_en}

Return JSON with exactly these keys:
{{
  "back_translation_modern_zh": "..."
}}"""
