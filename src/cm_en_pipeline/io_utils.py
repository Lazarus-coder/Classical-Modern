from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    paths = {
        "root": root,
        "raw": root / "raw",
        "scored": root / "scored",
        "filtered": root / "filtered",
        "logs": root / "logs",
        "cache": root / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(output_dir: str | Path, command_name: str) -> tuple[logging.Logger, Path]:
    paths = ensure_output_dirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths["logs"] / f"{command_name}_{timestamp}.log"
    logger = logging.getLogger(f"cm_en_pipeline.{command_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def coerce_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: coerce_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(item) for item in value]
    return value


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(coerce_jsonable(row), ensure_ascii=False) + "\n")
        handle.flush()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}: {exc}") from exc


def load_success_record_ids(path: str | Path, status_key: str = "overall_status") -> set[str]:
    record_ids: set[str] = set()
    for row in iter_jsonl(path) or []:
        if row.get(status_key) == "success":
            record_id = row.get("record_id")
            if record_id:
                record_ids.add(str(record_id))
    return record_ids


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class SQLiteCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    normalized_input TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_text TEXT,
                    response_json TEXT,
                    parsed_json TEXT,
                    usage_json TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        for field in ("request_json", "response_json", "parsed_json", "usage_json"):
            value = payload.get(field)
            if value:
                payload[field] = json.loads(value)
            else:
                payload[field] = None
        return payload

    def put(
        self,
        *,
        cache_key: str,
        provider: str,
        model: str,
        task_type: str,
        prompt_version: str,
        normalized_input: str,
        request_json: dict[str, Any],
        response_text: str | None,
        response_json: dict[str, Any] | None,
        parsed_json: dict[str, Any] | None,
        usage_json: dict[str, Any] | None,
        status: str,
        error_message: str | None,
    ) -> None:
        now = utc_now_iso()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO llm_cache (
                    cache_key, provider, model, task_type, prompt_version, normalized_input,
                    request_json, response_text, response_json, parsed_json, usage_json,
                    status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    task_type = excluded.task_type,
                    prompt_version = excluded.prompt_version,
                    normalized_input = excluded.normalized_input,
                    request_json = excluded.request_json,
                    response_text = excluded.response_text,
                    response_json = excluded.response_json,
                    parsed_json = excluded.parsed_json,
                    usage_json = excluded.usage_json,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    provider,
                    model,
                    task_type,
                    prompt_version,
                    normalized_input,
                    json.dumps(coerce_jsonable(request_json), ensure_ascii=False),
                    response_text,
                    json.dumps(coerce_jsonable(response_json), ensure_ascii=False) if response_json else None,
                    json.dumps(coerce_jsonable(parsed_json), ensure_ascii=False) if parsed_json else None,
                    json.dumps(coerce_jsonable(usage_json), ensure_ascii=False) if usage_json else None,
                    status,
                    error_message,
                    now,
                    now,
                ),
            )

