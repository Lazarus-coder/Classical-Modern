# Classical-Modern English Pipeline

This project provides a production-minded pipeline for constructing an English translation dataset from the [NiuTrans Classical-Modern dataset](https://github.com/NiuTrans/Classical-Modern).

The pipeline processes aligned Classical Chinese and Modern Chinese text pairs, translates the Modern Chinese side to English, and performs round-trip validation by back-translating to Modern Chinese for quality filtering.

## Key Features

- **Dataset Processing**: Scans and normalizes the bilingual dataset structure
- **Translation Pipeline**: Modern Chinese → English → Modern Chinese round-trip
- **Quality Filtering**: Multiple scoring metrics (BLEU, chrF, edit similarity, embeddings)
- **Caching & Resumability**: Efficient API usage with persistent caching
- **CLI Interface**: Modular commands for scanning, translating, scoring, and filtering

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. Configure API credentials (see detailed setup below)

3. Run the full pipeline:
   ```bash
   python -m cm_en_pipeline.cli run-all --dataset-root "./双语数据" --output-dir "./outputs"
   ```

## Project Structure

- `src/cm_en_pipeline/`: Core pipeline code
- `双语数据/`: Bilingual Classical-Modern Chinese dataset
- `古文原文/`: Original Classical Chinese texts
- `outputs/`: Pipeline outputs (raw, scored, filtered)
- `tests/`: Test suite

## Documentation

For detailed pipeline documentation, API configuration, CLI usage, and advanced options, see [README_pipeline.md](README_pipeline.md).

## Original Dataset

This pipeline is based on the [NiuTrans Classical-Modern dataset](https://github.com/NiuTrans/Classical-Modern), which provides aligned Classical and Modern Chinese text pairs from historical works.

## License

Copyright (c) 2022 NiuTrans Open Source - See [LICENSE](LICENSE) for details.