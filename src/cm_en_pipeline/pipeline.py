from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import FilterConfig, PipelineConfig
from .dataset import DatasetRecord, iter_dataset_records, scan_dataset
from .filtering import build_filtered_export, evaluate_record
from .io_utils import (
    SQLiteCache,
    append_jsonl,
    ensure_output_dirs,
    iter_jsonl,
    load_success_record_ids,
    utc_now_iso,
    write_csv,
)
from .llm_client import OpenAICompatibleChatClient
from .translate import translate_record


def _chunked(records: Iterable[DatasetRecord], batch_size: int) -> Iterator[list[DatasetRecord]]:
    batch: list[DatasetRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_all_record_ids(path: str | Path) -> set[str]:
    record_ids: set[str] = set()
    for row in iter_jsonl(path):
        record_id = row.get("record_id")
        if record_id:
            record_ids.add(str(record_id))
    return record_ids


def build_scan_report(dataset_root: str | Path) -> dict[str, Any]:
    summary = scan_dataset(dataset_root)
    return {
        "dataset_root": str(summary.dataset_root),
        "folder_count": summary.folder_count,
        "total_records": summary.total_records,
        "skipped_issue_count": len(summary.skipped_issues),
        "skipped_issues": [
            {
                "issue_type": issue.issue_type,
                "folder_path": issue.folder_path,
                "message": issue.message,
            }
            for issue in summary.skipped_issues[:100]
        ],
    }


def run_translation(config: PipelineConfig, logger: logging.Logger) -> dict[str, Any]:
    ensure_output_dirs(config.output_dir)
    config.provider.validate_for_translation()

    scan_summary = scan_dataset(config.dataset_root)
    logger.info(
        "Discovered %s aligned folders with %s total records.",
        scan_summary.folder_count,
        scan_summary.total_records,
    )
    if scan_summary.skipped_issues:
        logger.warning("Skipped %s folders during scanning.", len(scan_summary.skipped_issues))
        for issue in scan_summary.skipped_issues[:20]:
            logger.warning("%s | %s | %s", issue.issue_type, issue.folder_path, issue.message)

    completed_ids = set() if config.overwrite else load_success_record_ids(config.raw_output_path)
    cache = SQLiteCache(config.cache_path)
    client = OpenAICompatibleChatClient(config.provider, cache)

    translated = 0
    successful = 0
    failed = 0
    skipped_completed = 0
    batch_size = max(config.batch_size, 1)
    worker_count = max(config.provider.max_concurrency, 1)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        batch: list[DatasetRecord] = []
        for record in iter_dataset_records(scan_summary, config.selection):
            if record.record_id in completed_ids:
                skipped_completed += 1
                continue
            batch.append(record)
            if len(batch) < batch_size:
                continue
            futures = {
                executor.submit(translate_record, batch_record, client): batch_record.record_id
                for batch_record in batch
            }
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - safety net
                    row = {
                        "record_id": record_id,
                        "provider": config.provider.provider,
                        "model": config.provider.model,
                        "prompt_version": "unknown",
                        "created_at_utc": utc_now_iso(),
                        "updated_at_utc": utc_now_iso(),
                        "forward_status": "internal_error",
                        "backward_status": "internal_error",
                        "overall_status": "failed",
                        "forward_error": str(exc),
                        "backward_error": str(exc),
                    }
                append_jsonl(config.raw_output_path, row)
                translated += 1
                if row.get("overall_status") == "success":
                    successful += 1
                else:
                    failed += 1
                if translated % max(config.progress_every, 1) == 0:
                    logger.info(
                        "Translated %s records so far (%s success, %s failed, %s skipped-complete).",
                        translated,
                        successful,
                        failed,
                        skipped_completed,
                    )
            batch = []

        if batch:
            futures = {
                executor.submit(translate_record, batch_record, client): batch_record.record_id
                for batch_record in batch
            }
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - safety net
                    row = {
                        "record_id": record_id,
                        "provider": config.provider.provider,
                        "model": config.provider.model,
                        "prompt_version": "unknown",
                        "created_at_utc": utc_now_iso(),
                        "updated_at_utc": utc_now_iso(),
                        "forward_status": "internal_error",
                        "backward_status": "internal_error",
                        "overall_status": "failed",
                        "forward_error": str(exc),
                        "backward_error": str(exc),
                    }
                append_jsonl(config.raw_output_path, row)
                translated += 1
                if row.get("overall_status") == "success":
                    successful += 1
                else:
                    failed += 1
                if translated % max(config.progress_every, 1) == 0:
                    logger.info(
                        "Translated %s records so far (%s success, %s failed, %s skipped-complete).",
                        translated,
                        successful,
                        failed,
                        skipped_completed,
                    )

    summary = {
        "raw_output_path": str(config.raw_output_path),
        "translated_records": translated,
        "successful_records": successful,
        "failed_records": failed,
        "skipped_completed_records": skipped_completed,
    }
    logger.info("Translation stage complete: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def run_scoring(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    filter_config: FilterConfig,
    overwrite: bool,
    progress_every: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    from .metrics import MetricsScorer

    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    if overwrite and output_path.exists():
        output_path.unlink()

    existing_ids = set() if overwrite else _load_all_record_ids(output_path)
    scorer = MetricsScorer(filter_config, logger=logger)

    processed = 0
    skipped = 0
    passed = 0
    for row in iter_jsonl(input_path):
        record_id = row.get("record_id")
        if not record_id:
            logger.warning("Skipping row without record_id in %s", input_path)
            continue
        if record_id in existing_ids:
            skipped += 1
            continue
        scored_row = scorer.score_record(row)
        pass_filter, filter_reason = evaluate_record(scored_row, filter_config)
        scored_row["pass_filter"] = pass_filter
        scored_row["filter_reason"] = filter_reason
        append_jsonl(output_path, scored_row)
        processed += 1
        if pass_filter:
            passed += 1
        if processed % max(progress_every, 1) == 0:
            logger.info(
                "Scored %s records so far (%s passed filter, %s skipped existing).",
                processed,
                passed,
                skipped,
            )

    summary = {
        "scored_output_path": str(output_path),
        "processed_records": processed,
        "passed_records": passed,
        "skipped_existing_records": skipped,
    }
    logger.info("Scoring stage complete: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def run_filtering(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_json_path = Path(output_jsonl)
    output_csv_path = Path(output_csv)

    if overwrite:
        if output_json_path.exists():
            output_json_path.unlink()
        if output_csv_path.exists():
            output_csv_path.unlink()

    existing_ids = set() if overwrite else _load_all_record_ids(output_json_path)
    passed = 0
    skipped = 0
    for row in iter_jsonl(input_path):
        record_id = row.get("record_id")
        if not record_id:
            continue
        if record_id in existing_ids:
            skipped += 1
            continue
        if row.get("pass_filter"):
            append_jsonl(output_json_path, build_filtered_export(row))
            passed += 1

    filtered_rows = list(iter_jsonl(output_json_path))
    csv_fieldnames = [
        "record_id",
        "source_classical_zh",
        "target_modern_zh",
        "translation_en",
        "back_translation_modern_zh",
        "book",
        "chapter_path",
        "line_index",
        "chrf",
        "bleu",
        "edit_similarity",
        "length_ratio",
        "embedding_similarity",
        "filter_reason",
    ]
    write_csv(output_csv_path, filtered_rows, csv_fieldnames)

    summary = {
        "filtered_output_path": str(output_json_path),
        "filtered_csv_path": str(output_csv_path),
        "passed_records_written": passed,
        "skipped_existing_records": skipped,
        "total_filtered_records": len(filtered_rows),
    }
    logger.info("Filtering stage complete: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def run_all(config: PipelineConfig, logger: logging.Logger) -> dict[str, Any]:
    translate_summary = run_translation(config, logger)
    score_summary = run_scoring(
        input_jsonl=config.raw_output_path,
        output_jsonl=config.scored_output_path,
        filter_config=config.filters,
        overwrite=config.overwrite,
        progress_every=config.progress_every,
        logger=logger,
    )
    filter_summary = run_filtering(
        input_jsonl=config.scored_output_path,
        output_jsonl=config.filtered_output_path,
        output_csv=config.filtered_csv_path,
        overwrite=config.overwrite,
        logger=logger,
    )
    return {
        "translate": translate_summary,
        "score": score_summary,
        "filter": filter_summary,
    }
