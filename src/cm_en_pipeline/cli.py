from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import FilterConfig, PipelineConfig, ProviderConfig, SelectionConfig
from .io_utils import ensure_output_dirs, setup_logging
from .pipeline import build_scan_report, run_all, run_filtering, run_scoring, run_translation


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book", help="Only process records from a single book.")
    parser.add_argument(
        "--path-keyword",
        help="Only process chapter paths containing this substring.",
    )
    parser.add_argument("--start-offset", type=int, default=0, help="Start offset after selection filters.")
    parser.add_argument("--end-offset", type=int, help="Exclusive end offset after selection filters.")
    parser.add_argument("--max-records", type=int, help="Maximum number of records to process.")
    parser.add_argument(
        "--test-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Limit processing to a small sample for smoke tests.",
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=20,
        help="Sample size used when --test-mode is enabled.",
    )


def _selection_from_args(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        book=args.book,
        path_keyword=args.path_keyword,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        max_records=args.max_records,
        test_mode=args.test_mode,
        test_sample_size=args.test_sample_size,
    )


def _filter_config_from_args(args: argparse.Namespace) -> FilterConfig:
    return FilterConfig(
        min_chrf=args.min_chrf,
        min_bleu=args.min_bleu,
        min_edit_similarity=args.min_edit_similarity,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        min_embedding_similarity=args.min_embedding_similarity,
        enable_embeddings=args.enable_embeddings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical-Modern English pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan dataset folders and report aligned pairs.")
    scan_parser.add_argument("--dataset-root", required=True)
    scan_parser.add_argument("--output-dir", default="./outputs")

    translate_parser = subparsers.add_parser("translate", help="Translate and back-translate dataset records.")
    translate_parser.add_argument("--dataset-root", required=True)
    translate_parser.add_argument("--output-dir", default="./outputs")
    translate_parser.add_argument("--overwrite", action="store_true")
    translate_parser.add_argument("--batch-size", type=int, default=16)
    translate_parser.add_argument("--progress-every", type=int, default=50)
    _add_selection_arguments(translate_parser)

    score_parser = subparsers.add_parser("score", help="Score raw translation outputs.")
    score_parser.add_argument("--input-jsonl", required=True)
    score_parser.add_argument("--output-jsonl", required=True)
    score_parser.add_argument("--output-dir", default="./outputs")
    score_parser.add_argument("--overwrite", action="store_true")
    score_parser.add_argument("--progress-every", type=int, default=100)
    score_parser.add_argument("--min-chrf", type=float, default=45.0)
    score_parser.add_argument("--min-bleu", type=float, default=10.0)
    score_parser.add_argument("--min-edit-similarity", type=float, default=0.45)
    score_parser.add_argument("--min-length-ratio", type=float, default=0.5)
    score_parser.add_argument("--max-length-ratio", type=float, default=1.8)
    score_parser.add_argument("--min-embedding-similarity", type=float, default=0.75)
    score_parser.add_argument(
        "--enable-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    filter_parser = subparsers.add_parser("filter", help="Export high-quality filtered records.")
    filter_parser.add_argument("--input-jsonl", required=True)
    filter_parser.add_argument("--output-jsonl", required=True)
    filter_parser.add_argument("--output-csv", required=True)
    filter_parser.add_argument("--output-dir", default="./outputs")
    filter_parser.add_argument("--overwrite", action="store_true")

    run_all_parser = subparsers.add_parser("run-all", help="Run translation, scoring, and filtering.")
    run_all_parser.add_argument("--dataset-root", required=True)
    run_all_parser.add_argument("--output-dir", default="./outputs")
    run_all_parser.add_argument("--overwrite", action="store_true")
    run_all_parser.add_argument("--batch-size", type=int, default=16)
    run_all_parser.add_argument("--progress-every", type=int, default=50)
    run_all_parser.add_argument("--min-chrf", type=float, default=45.0)
    run_all_parser.add_argument("--min-bleu", type=float, default=10.0)
    run_all_parser.add_argument("--min-edit-similarity", type=float, default=0.45)
    run_all_parser.add_argument("--min-length-ratio", type=float, default=0.5)
    run_all_parser.add_argument("--max-length-ratio", type=float, default=1.8)
    run_all_parser.add_argument("--min-embedding-similarity", type=float, default=0.75)
    run_all_parser.add_argument(
        "--enable-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    _add_selection_arguments(run_all_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_output_dirs(args.output_dir)
    logger, log_path = setup_logging(args.output_dir, args.command.replace("-", "_"))
    logger.info("Log file: %s", log_path)

    if args.command == "scan":
        report = build_scan_report(args.dataset_root)
        logger.info("Scan report: %s", json.dumps(report, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "translate":
        config = PipelineConfig(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            provider=ProviderConfig(),
            selection=_selection_from_args(args),
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        summary = run_translation(config, logger)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score":
        filter_config = _filter_config_from_args(args)
        summary = run_scoring(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            filter_config=filter_config,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            logger=logger,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "filter":
        summary = run_filtering(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            output_csv=args.output_csv,
            overwrite=args.overwrite,
            logger=logger,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-all":
        config = PipelineConfig(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            provider=ProviderConfig(),
            filters=_filter_config_from_args(args),
            selection=_selection_from_args(args),
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        summary = run_all(config, logger)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
