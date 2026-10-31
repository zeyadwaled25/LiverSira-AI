# LiverSira-AI

> NLP data pipeline for the HCC Surveillance AI Platform — extracting structured liver-lesion features from radiology reports.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](tests/)

## Overview

**LiverSira-AI** is the data-processing backbone of the **HCC Surveillance AI Platform** — a medical decision-support system for monitoring liver cirrhosis patients and predicting hepatocellular carcinoma (HCC) risk.

This repository handles the **NLP data pipeline**: transforming raw radiology reports from the [AbdomenAtlas 3.0 Mini](https://github.com/MrGiovanni/AbdomenAtlas) dataset into structured, machine-learning-ready features. These features feed downstream models for lesion classification, risk scoring, and follow-up prediction.

> **System boundaries:** The platform does **not** independently diagnose cancer, replace confirmatory CT/MRI examinations, or override physician decisions. All AI outputs are advisory.

### Pipeline & Architecture

| Tier | Component | Purpose |
|------|-----------|---------|
| **Schemas** | [`schemas/master_schema.py`](schemas/master_schema.py) | Master Record schema preserving all 37 columns & multi-organ data |
| | [`schemas/ground_truth_schema.py`](schemas/ground_truth_schema.py) | Benchmark target schema with consistency validators & audit status |
| | [`schemas/inference_schema.py`](schemas/inference_schema.py) | Model prediction schema with evidence text spans & ternary logic |
| **Data Ops** | [`src/generate_master_jsonl.py`](src/generate_master_jsonl.py) | Generates `master_records.jsonl` (9,262 records, 0 rows dropped) |
| | [`src/generate_ground_truth_jsonl.py`](src/generate_ground_truth_jsonl.py) | Generates source-derived `ground_truth.jsonl` (pending verification) |
| | [`src/audit_target_consistency.py`](src/audit_target_consistency.py) | Automated text vs label consistency audit across all 9,262 studies |
| | [`src/split_dataset.py`](src/split_dataset.py) | Stratified 70/15/15 dataset partitioning into Train/Val/Test |
| | [`src/baseline_extractor.py`](src/baseline_extractor.py) | Rule-based clinical NLP baseline extracting 4 core targets |
| **Evaluation** | [`evaluation/evaluate_baseline.py`](evaluation/evaluate_baseline.py) | Benchmark evaluation against Ground Truth across dataset splits |
| | [`src/audit_dataset.py`](src/audit_dataset.py) | Profile the raw CSV — missing values, duplicates, target consistency |
| | [`src/check_pdf_csv_match.py`](src/check_pdf_csv_match.py) | Cross-check BDMAP IDs between CSV metadata and 9,262 PDF reports |
| | [`src/prepare_nlp_dataset.py`](src/prepare_nlp_dataset.py) | Clean report text, produce `liver_nlp_clean.csv` |

## Project Structure

```
LiverSira-AI/
├── data/
│   ├── raw/                        # Source data (git-ignored — see data/README.md)
│   │   ├── AbdomenAtlas3.0MiniWithMeta.csv
│   │   └── pdf_reports/            # 9,262 radiology report PDFs
│   └── processed/                  # Pipeline outputs (git-ignored)
│       ├── master_records.jsonl    # 9,262 full study records
│       ├── ground_truth.jsonl      # 9,262 benchmark records
│       └── liver_nlp_clean.csv
├── schemas/                        # Pydantic v2 data models
│   ├── __init__.py
│   ├── master_schema.py
│   ├── ground_truth_schema.py
│   └── inference_schema.py
├── src/
│   ├── __init__.py
│   ├── audit_dataset.py            # Data-quality profiling
│   ├── check_pdf_csv_match.py      # PDF ↔ CSV ID matching
│   ├── generate_ground_truth_jsonl.py
│   ├── generate_master_jsonl.py
│   ├── prepare_nlp_dataset.py      # NLP dataset builder
│   └── validate_targets.py         # Human adjudication sampler
├── tests/                          # Pytest test suite
│   └── test_schemas.py
├── docs/                           # Documentation
├── evaluation/                     # Model evaluation outputs
├── pyproject.toml                  # Pytest configuration
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.12 or later
- The [AbdomenAtlas 3.0 Mini](https://github.com/MrGiovanni/AbdomenAtlas) dataset (CSV + PDF reports)

### Installation

```bash
# Clone the repository
git clone https://github.com/zeyadwaled25/LiverSira-AI.git
cd LiverSira-AI

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Place `AbdomenAtlas3.0MiniWithMeta.csv` in `data/raw/`.
2. Extract PDF reports into `data/raw/pdf_reports/`.
3. See [`data/README.md`](data/README.md) for the expected directory layout.

### Running the Pipeline

```bash
# 1. Audit the raw dataset
python -m src.audit_dataset

# 2. Verify PDF ↔ CSV matching
python -m src.check_pdf_csv_match

# 3. Build the clean NLP dataset
python -m src.prepare_nlp_dataset

# 4. Generate Master Dataset JSONL (all 9,262 records)
python -m src.generate_master_jsonl

# 5. Generate Ground Truth JSONL (pending verification)
python -m src.generate_ground_truth_jsonl

# 6. Extract human validation review sample (79 cases)
python -m src.validate_targets

# 7. Run test suite
pytest
```

## Dataset Details

| Property | Value |
|----------|-------|
| Source | [AbdomenAtlas 3.0 Mini](https://github.com/MrGiovanni/AbdomenAtlas) |
| Total Records | 9,262 |
| Positive Liver Lesion Records | 1,470 |
| PDF reports | 9,262 |
| Master JSONL Records | 9,262 |
| Ground Truth JSONL Records | 9,262 |

### Missing-Value Placeholders

The pipeline standardizes missing liver-lesion values with explicit sentinels:

| Placeholder | Meaning |
|-------------|---------|
| `NO_LESION` | Zero lesions — location field does not apply |
| `NOT_APPLICABLE` | Zero lesions — diameter / attenuation do not apply |
| `UNKNOWN` | Lesions present but the value is missing in the source data |

## Part of the HCC Surveillance AI Platform

This NLP pipeline is one component of a larger clinical system:

| Phase | Component | Status |
|-------|-----------|--------|
| MVP-1 | Ultrasound AI (lesion classification) | Mandatory |
| MVP-1 | **NLP data pipeline (this repo)** | Mandatory |
| Phase 2 | Clinical registry & workflow | Conditional |
| Phase 3 | HCC risk prediction, follow-up prediction, patient/coordinator portals | Conditional |

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m "Add my feature"`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [AbdomenAtlas 3.0](https://github.com/MrGiovanni/AbdomenAtlas) for the radiology dataset.
- Faculty of Computer Science & AI — Menoufia National University.
