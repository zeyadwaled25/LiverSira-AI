# LiverSira-AI

> NLP data pipeline for the HCC Surveillance AI Platform — extracting structured liver-lesion features from radiology reports.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**LiverSira-AI** is the data-processing backbone of the **HCC Surveillance AI Platform** — a medical decision-support system for monitoring liver cirrhosis patients and predicting hepatocellular carcinoma (HCC) risk.

This repository handles the **NLP data pipeline**: transforming raw radiology reports from the [AbdomenAtlas 3.0 Mini](https://github.com/MrGiovanni/AbdomenAtlas) dataset into structured, machine-learning-ready features. These features feed downstream models for lesion classification, risk scoring, and follow-up prediction.

> **System boundaries:** The platform does **not** independently diagnose cancer, replace confirmatory CT/MRI examinations, or override physician decisions. All AI outputs are advisory.

### Pipeline Scripts

| Script | Purpose |
|--------|---------|
| [`audit_dataset.py`](src/audit_dataset.py) | Profile the raw CSV — missing values, duplicates, target-column consistency |
| [`check_pdf_csv_match.py`](src/check_pdf_csv_match.py) | Cross-check BDMAP IDs between the CSV metadata and 9,262 PDF report files |
| [`prepare_nlp_dataset.py`](src/prepare_nlp_dataset.py) | Clean report text, standardize missing-value placeholders, produce `liver_nlp_clean.csv` |

## Project Structure

```
LiverSira-AI/
├── data/
│   ├── raw/                        # Source data (git-ignored — see data/README.md)
│   │   ├── AbdomenAtlas3.0MiniWithMeta.csv
│   │   └── pdf_reports/            # 9,262 radiology report PDFs
│   └── processed/                  # Pipeline outputs (git-ignored)
│       └── liver_nlp_clean.csv
├── src/
│   ├── __init__.py
│   ├── audit_dataset.py            # Data-quality audit
│   ├── check_pdf_csv_match.py      # PDF ↔ CSV ID matching
│   └── prepare_nlp_dataset.py      # NLP dataset builder
├── tests/                          # Test suite
├── docs/                           # Documentation
├── evaluation/                     # Model evaluation outputs
├── schemas/                        # Data schemas
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

### Usage

```bash
# 1. Audit the raw dataset
python -m src.audit_dataset

# 2. Verify PDF ↔ CSV matching
python -m src.check_pdf_csv_match

# 3. Build the clean NLP dataset
python -m src.prepare_nlp_dataset
```

## Dataset Details

| Property | Value |
|----------|-------|
| Source | [AbdomenAtlas 3.0 Mini](https://github.com/MrGiovanni/AbdomenAtlas) |
| Records | 9,262 |
| PDF reports | 9,262 |
| Key columns | BDMAP ID, narrative report, lesion count, location (1–8), diameter (cm), attenuation |

### Missing-Value Placeholders

The pipeline standardizes missing liver-lesion values with explicit sentinels:

| Placeholder | Meaning |
|-------------|---------|
| `NO_LESION` | Zero lesions — location field does not apply |
| `NOT_APPLICABLE` | Zero lesions — diameter / attenuation do not apply |
| `UNKNOWN` | Lesions present but the value is missing in the source data |

## Part of the HCC Surveillance AI Platform

This NLP pipeline is one component of a larger clinical system (see the [Project Scope Document](docs/) for full details):

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
