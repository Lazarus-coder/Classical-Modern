from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from .config import SelectionConfig


@dataclass
class DatasetRecord:
    record_id: str
    book: str
    chapter_path: str
    line_index: int
    source_classical_zh: str
    target_modern_zh: str


@dataclass
class ScanIssue:
    issue_type: str
    folder_path: str
    message: str


@dataclass
class FolderPair:
    folder_path: Path
    relative_folder_path: str
    book: str
    chapter_path: str
    source_path: Path
    target_path: Path
    line_count: int


@dataclass
class ScanSummary:
    dataset_root: Path
    valid_folders: list[FolderPair]
    skipped_issues: list[ScanIssue]
    total_records: int

    @property
    def folder_count(self) -> int:
        return len(self.valid_folders)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def scan_dataset(dataset_root: str | Path) -> ScanSummary:
    root = Path(dataset_root).expanduser().resolve()
    valid_folders: list[FolderPair] = []
    skipped_issues: list[ScanIssue] = []
    total_records = 0

    for source_path in sorted(root.rglob("source.txt")):
        folder = source_path.parent
        target_path = folder / "target.txt"
        if not target_path.exists():
            skipped_issues.append(
                ScanIssue(
                    issue_type="missing_target",
                    folder_path=str(folder),
                    message="Folder contains source.txt but not target.txt.",
                )
            )
            continue

        source_lines = _read_lines(source_path)
        target_lines = _read_lines(target_path)
        if len(source_lines) != len(target_lines):
            skipped_issues.append(
                ScanIssue(
                    issue_type="line_count_mismatch",
                    folder_path=str(folder),
                    message=(
                        f"source.txt has {len(source_lines)} lines while target.txt has "
                        f"{len(target_lines)} lines."
                    ),
                )
            )
            continue

        rel_folder = folder.relative_to(root)
        rel_posix = PurePosixPath(rel_folder.as_posix()).as_posix()
        parts = rel_folder.parts
        book = parts[0] if parts else folder.name
        valid_folders.append(
            FolderPair(
                folder_path=folder,
                relative_folder_path=rel_posix,
                book=book,
                chapter_path=rel_posix,
                source_path=source_path,
                target_path=target_path,
                line_count=len(source_lines),
            )
        )
        total_records += len(source_lines)

    return ScanSummary(
        dataset_root=root,
        valid_folders=valid_folders,
        skipped_issues=skipped_issues,
        total_records=total_records,
    )


def iter_dataset_records(
    scan_summary: ScanSummary,
    selection: SelectionConfig | None = None,
) -> Iterator[DatasetRecord]:
    selection = selection or SelectionConfig()
    selected_count = 0
    flattened_offset = 0
    max_records = selection.normalized_max_records()

    for folder in scan_summary.valid_folders:
        if selection.book and folder.book != selection.book:
            continue
        if selection.path_keyword and selection.path_keyword not in folder.chapter_path:
            continue

        source_lines = _read_lines(folder.source_path)
        target_lines = _read_lines(folder.target_path)
        for line_index, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
            if flattened_offset < selection.start_offset:
                flattened_offset += 1
                continue
            if selection.end_offset is not None and flattened_offset >= selection.end_offset:
                return
            if max_records is not None and selected_count >= max_records:
                return

            record_id = f"{folder.relative_folder_path}::{line_index}"
            yield DatasetRecord(
                record_id=record_id,
                book=folder.book,
                chapter_path=folder.chapter_path,
                line_index=line_index,
                source_classical_zh=source_line.strip(),
                target_modern_zh=target_line.strip(),
            )
            selected_count += 1
            flattened_offset += 1
