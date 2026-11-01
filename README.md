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
| **Data Ops** | [`src/audit_dataset.py`](src/audit_dataset.py) | Profile the raw CSV — missing values, duplicates, target consistency |
| | [`src/check_pdf_csv_match.py`](src/check_pdf_csv_match.py) | Cross-check BDMAP IDs between CSV metadata and 9,262 PDF reports |
| | [`src/prepare_nlp_dataset.py`](src/prepare_nlp_dataset.py) | Clean report text, produce `liver_nlp_clean.csv` |
| | [`src/generate_master_jsonl.py`](src/generate_master_jsonl.py) | Generates `master_records.jsonl` (9,262 records, 0 rows dropped) |
| | [`src/generate_ground_truth_jsonl.py`](src/generate_ground_truth_jsonl.py) | Generates source-derived `ground_truth.jsonl` (pending verification) |
| | [`src/audit_target_consistency.py`](src/audit_target_consistency.py) | Automated text vs label consistency audit across all 9,262 studies |
| | [`src/validate_targets.py`](src/validate_targets.py) | Extracts the 79-case human adjudication sample |
| | [`src/adjudicate_validation_sample.py`](src/adjudicate_validation_sample.py) | Records adjudication and promotes verified labels in `ground_truth.jsonl` |
| | [`src/split_dataset.py`](src/split_dataset.py) | Stratified 70/15/15 dataset partitioning into Train/Val/Test |
| | [`src/baseline_extractor.py`](src/baseline_extractor.py) | Rule-based clinical NLP baseline extracting 4 core targets |
| **Evaluation** | [`evaluation/evaluate_baseline.py`](evaluation/evaluate_baseline.py) | Benchmark evaluation against Ground Truth across dataset splits |

## Project Structure

```
LiverSira-AI/
├── data/
│   ├── raw/                        # Source data (git-ignored — see data/README.md)
│   │   ├── AbdomenAtlas3.0MiniWithMeta.csv
│   │   └── pdf_reports/            # 9,262 radiology report PDFs
│   └── processed/                  # Pipeline outputs (git-ignored)
│       ├── liver_nlp_clean.csv
│       ├── master_records.jsonl
│       ├── ground_truth.jsonl
│       ├── target_consistency_audit.csv
│       ├── target_validation_review.csv
│       ├── splits.json
│       └── dataset_splits.csv
├── schemas/                        # Pydantic v2 data models
│   ├── __init__.py
│   ├── master_schema.py
│   ├── ground_truth_schema.py
│   └── inference_schema.py
├── src/
│   ├── __init__.py
│   ├── audit_dataset.py            # Data-quality profiling
│   ├── check_pdf_csv_match.py      # PDF ↔ CSV ID matching
│   ├── prepare_nlp_dataset.py      # NLP dataset builder
│   ├── validate_targets.py         # Human adjudication sampler
│   ├── adjudicate_validation_sample.py
│   ├── generate_master_jsonl.py
│   ├── generate_ground_truth_jsonl.py
│   ├── audit_target_consistency.py
│   ├── split_dataset.py
│   └── baseline_extractor.py       # Rule-based NLP baseline
├── evaluation/
│   ├── evaluate_baseline.py        # Baseline metrics
│   └── baseline_inferences.jsonl   # Baseline predictions (9,262 records)
├── scripts/
│   ├── __init__.py
│   └── run_pipeline.py             # One-shot end-to-end pipeline runner
├── tests/                          # Pytest test suite
│   ├── test_schemas.py
│   ├── test_audit.py
│   └── test_baseline.py
├── pyproject.toml                  # Pytest configuration
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.12 or later (tested on 3.14; dependencies are pinned in `requirements.txt` / `pyproject.toml`)
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

Run the steps in order, or execute the whole pipeline at once:

```bash
python -m scripts.run_pipeline
```

Individual steps (equivalent to the one-shot command above):

```bash
# 1. Audit the raw dataset
python -m src.audit_dataset

# 2. Verify PDF <-> CSV matching
python -m src.check_pdf_csv_match

# 3. Build the clean NLP dataset
python -m src.prepare_nlp_dataset

# 4. Generate Master Dataset JSONL (all 9,262 records)
python -m src.generate_master_jsonl

# 5. Generate Ground Truth JSONL (pending verification)
python -m src.generate_ground_truth_jsonl

# 6. Audit label <-> report consistency
python -m src.audit_target_consistency

# 7. Extract human validation review sample (79 cases)
python -m src.validate_targets

# 8. Adjudicate the sample and promote verified labels
python -m src.adjudicate_validation_sample

# 9. Create stratified 70/15/15 splits
python -m src.split_dataset

# 10. Run the rule-based NLP baseline
python -m src.baseline_extractor

# 11. Evaluate the baseline on the validation and test splits
python -m evaluation.evaluate_baseline

# 12. Run test suite
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
