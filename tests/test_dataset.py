from __future__ import annotations

from cm_en_pipeline.config import SelectionConfig
from cm_en_pipeline.dataset import iter_dataset_records, scan_dataset


def test_scan_dataset_discovers_nested_utf8_paths(tmp_path):
    dataset_root = tmp_path / "双语数据"
    chapter_dir = dataset_root / "史记" / "本纪" / "卷一"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "source.txt").write_text("古文甲\n古文乙\n", encoding="utf-8")
    (chapter_dir / "target.txt").write_text("现代甲\n现代乙\n", encoding="utf-8")

    summary = scan_dataset(dataset_root)
    records = list(iter_dataset_records(summary, SelectionConfig()))

    assert summary.folder_count == 1
    assert summary.total_records == 2
    assert records[0].record_id == "史记/本纪/卷一::0"
    assert records[1].target_modern_zh == "现代乙"


def test_scan_dataset_skips_mismatched_line_counts(tmp_path):
    dataset_root = tmp_path / "双语数据"
    chapter_dir = dataset_root / "老子" / "德经" / "第一章"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "source.txt").write_text("甲\n乙\n", encoding="utf-8")
    (chapter_dir / "target.txt").write_text("现代甲\n", encoding="utf-8")

    summary = scan_dataset(dataset_root)

    assert summary.folder_count == 0
    assert len(summary.skipped_issues) == 1
    assert summary.skipped_issues[0].issue_type == "line_count_mismatch"

