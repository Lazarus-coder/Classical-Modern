# CM-EN Pipeline

This pipeline turns the aligned Modern Chinese side of the NiuTrans Classical-Modern dataset into English, then back-translates the English into Modern Chinese for round-trip filtering.

Important caveat: the English is generated from `target.txt` (Modern Chinese), not directly from the Classical Chinese in `source.txt`. The original Classical Chinese sentence is still preserved in every exported record so the final dataset remains grounded in the aligned source pair.

## What It Does

The package scans the dataset root recursively for folders that contain both `source.txt` and `target.txt`, validates line alignment, and creates one normalized record per line:

- `record_id`
- `book`
- `chapter_path`
- `line_index`
- `source_classical_zh`
- `target_modern_zh`

Then it runs this two-step flow:

1. Modern Chinese `target.txt` sentence -> English translation
2. English translation -> Modern Chinese back-translation

The pipeline caches API responses on disk, writes append-only JSONL outputs for inspection, scores round-trip quality, and exports a filtered high-quality subset.

## Layout

```text
project_root/
  src/cm_en_pipeline/
  outputs/
    raw/
    scored/
    filtered/
    logs/
    cache/
  tests/
  requirements.txt
  README_pipeline.md
```

## Why `target.txt` Instead Of Direct Classical Chinese Translation

This version is intentionally an engineering-oriented corpus construction pipeline, not a direct Classical Chinese to English translation system. Using the existing Modern Chinese alignment gives us:

- a cleaner and better-controlled source sentence for English generation
- a straightforward round-trip validation path back into Modern Chinese
- a practical first-stage filter before attempting harder direct Classical Chinese workflows

## Environment Setup

Python 3.10+ is required.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Optional embedding similarity support:

- install `sentence-transformers`
- make sure a multilingual model is already available locally
- enable it with `ENABLE_EMBEDDINGS=true`

By default, the embedding loader uses `local_files_only=True` so the scorer can stay offline and predictable.

## API Configuration

Set environment variables before running the translation step:

```bash
export TRANSLATION_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_key_here
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
export MAX_CONCURRENCY=4
export REQUEST_TIMEOUT_SEC=60
export RETRY_MAX_ATTEMPTS=5
export RETRY_BASE_DELAY_SEC=1.5
```

Optional runtime knobs:

```bash
export REQUEST_INTERVAL_SEC=0.0
export TRANSLATION_BATCH_SIZE=16
export PROGRESS_EVERY=50
export ENABLE_EMBEDDINGS=false
export EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export EMBEDDING_LOCAL_FILES_ONLY=true
```

Secrets are never hardcoded in the package.

## Recommended First Run

Start with a small sample to verify credentials, prompts, and output shape:

```bash
python3 -m cm_en_pipeline.cli scan \
  --dataset-root "./双语数据"

python3 -m cm_en_pipeline.cli translate \
  --dataset-root "./双语数据" \
  --output-dir "./outputs" \
  --test-mode \
  --test-sample-size 20
```

You can also narrow the run:

- `--book "老子"`
- `--path-keyword "德经"`
- `--start-offset 0 --end-offset 100`
- `--max-records 1000`

## CLI

Scan dataset folders:

```bash
python3 -m cm_en_pipeline.cli scan \
  --dataset-root "./双语数据"
```

Translate and back-translate:

```bash
python3 -m cm_en_pipeline.cli translate \
  --dataset-root "./双语数据" \
  --output-dir "./outputs" \
  --max-records 1000
```

Score raw output:

```bash
python3 -m cm_en_pipeline.cli score \
  --input-jsonl "./outputs/raw/translations.jsonl" \
  --output-jsonl "./outputs/scored/translations_scored.jsonl"
```

Filter scored output:

```bash
python3 -m cm_en_pipeline.cli filter \
  --input-jsonl "./outputs/scored/translations_scored.jsonl" \
  --output-jsonl "./outputs/filtered/high_quality_parallel.jsonl" \
  --output-csv "./outputs/filtered/high_quality_parallel.csv"
```

Run all stages:

```bash
python3 -m cm_en_pipeline.cli run-all \
  --dataset-root "./双语数据" \
  --output-dir "./outputs" \
  --max-records 1000
```

## Outputs

### `outputs/raw/translations.jsonl`

One line per processed record with:

- original Classical Chinese and Modern Chinese
- English translation
- back-translation
- prompt versions
- provider/model metadata
- token usage if the API returns it
- response text and stage status fields for inspection

### `outputs/scored/translations_scored.jsonl`

Adds:

- `chrf`
- `bleu`
- `edit_similarity`
- `length_ratio`
- `embedding_similarity`
- `pass_filter`
- `filter_reason`

### `outputs/filtered/high_quality_parallel.jsonl`

Filtered high-quality export for downstream use.

### `outputs/filtered/high_quality_parallel.csv`

Spreadsheet-friendly version of the filtered export.

### `outputs/cache/translation_cache.sqlite3`

Persistent cache keyed by:

- provider
- model
- task type
- normalized input text
- prompt version

### `outputs/logs/*.log`

Stage-specific logs with progress, scan issues, and failures.

## Resumability

The translation stage skips records that already have a successful raw output unless `--overwrite` is passed.

Scoring and filtering also support append-style resume, but if you change scoring thresholds or re-translate previously failed items, the cleanest approach is to rerun those downstream stages with `--overwrite`.

## Filtering Rules

The scorer compares original `target_modern_zh` against `back_translation_modern_zh` using:

- chrF
- BLEU
- normalized edit similarity
- length ratio
- optional embedding cosine similarity

Filtering also rejects records for heuristic issues such as:

- refusal text
- meta wrappers like `Here is the translation`
- empty outputs
- suspiciously long outputs
- repeated garbage or formatting artifacts
- JSON/validation failures

Thresholds can be adjusted from the CLI for `score` and `run-all`.

## Notes And Caveats

- Paths and file IO are UTF-8 throughout.
- Folder depth is not assumed; the scanner only requires co-located `source.txt` and `target.txt`.
- If a folder has mismatched line counts, it is logged and skipped.
- Embedding similarity is optional by design so the pipeline can run in lighter environments.
