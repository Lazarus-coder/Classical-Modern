from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_CACHE_FILENAME = "translation_cache.sqlite3"


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ProviderConfig:
    provider: str = field(default_factory=lambda: _env_str("TRANSLATION_PROVIDER", "deepseek") or "deepseek")
    api_key: str | None = field(default_factory=lambda: _env_str("DEEPSEEK_API_KEY"))
    base_url: str = field(default_factory=lambda: _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com")
    model: str = field(default_factory=lambda: _env_str("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat")
    max_concurrency: int = field(default_factory=lambda: _env_int("MAX_CONCURRENCY", 4))
    request_timeout_sec: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT_SEC", 60.0))
    retry_max_attempts: int = field(default_factory=lambda: _env_int("RETRY_MAX_ATTEMPTS", 5))
    retry_base_delay_sec: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY_SEC", 1.5))
    request_interval_sec: float = field(default_factory=lambda: _env_float("REQUEST_INTERVAL_SEC", 0.0))
    use_json_mode: bool = field(default_factory=lambda: _env_bool("USE_JSON_MODE", True))

    def validate_for_translation(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set DEEPSEEK_API_KEY in the environment before running translation."
            )


@dataclass
class SelectionConfig:
    book: str | None = None
    path_keyword: str | None = None
    start_offset: int = 0
    end_offset: int | None = None
    max_records: int | None = None
    test_mode: bool = False
    test_sample_size: int = 20

    def normalized_max_records(self) -> int | None:
        if not self.test_mode:
            return self.max_records
        if self.max_records is None:
            return self.test_sample_size
        return min(self.max_records, self.test_sample_size)


@dataclass
class FilterConfig:
    min_chrf: float = field(default_factory=lambda: _env_float("FILTER_MIN_CHRF", 45.0))
    min_bleu: float = field(default_factory=lambda: _env_float("FILTER_MIN_BLEU", 10.0))
    min_edit_similarity: float = field(default_factory=lambda: _env_float("FILTER_MIN_EDIT_SIMILARITY", 0.45))
    min_length_ratio: float = field(default_factory=lambda: _env_float("FILTER_MIN_LENGTH_RATIO", 0.5))
    max_length_ratio: float = field(default_factory=lambda: _env_float("FILTER_MAX_LENGTH_RATIO", 1.8))
    min_embedding_similarity: float = field(default_factory=lambda: _env_float("FILTER_MIN_EMBEDDING_SIMILARITY", 0.75))
    enable_embeddings: bool = field(default_factory=lambda: _env_bool("ENABLE_EMBEDDINGS", False))
    embedding_model_name: str = field(
        default_factory=lambda: _env_str(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_local_files_only: bool = field(default_factory=lambda: _env_bool("EMBEDDING_LOCAL_FILES_ONLY", True))
    max_forward_expansion_ratio: float = field(default_factory=lambda: _env_float("FILTER_MAX_FORWARD_EXPANSION_RATIO", 8.0))
    max_backward_expansion_ratio: float = field(default_factory=lambda: _env_float("FILTER_MAX_BACKWARD_EXPANSION_RATIO", 3.0))


@dataclass
class PipelineConfig:
    dataset_root: Path
    output_dir: Path
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    prompt_version: str = DEFAULT_PROMPT_VERSION
    overwrite: bool = False
    batch_size: int = field(default_factory=lambda: _env_int("TRANSLATION_BATCH_SIZE", 16))
    progress_every: int = field(default_factory=lambda: _env_int("PROGRESS_EVERY", 50))
    cache_filename: str = DEFAULT_CACHE_FILENAME

    @property
    def raw_output_path(self) -> Path:
        return self.output_dir / "raw" / "translations.jsonl"

    @property
    def scored_output_path(self) -> Path:
        return self.output_dir / "scored" / "translations_scored.jsonl"

    @property
    def filtered_output_path(self) -> Path:
        return self.output_dir / "filtered" / "high_quality_parallel.jsonl"

    @property
    def filtered_csv_path(self) -> Path:
        return self.output_dir / "filtered" / "high_quality_parallel.csv"

    @property
    def cache_path(self) -> Path:
        return self.output_dir / "cache" / self.cache_filename
