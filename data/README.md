# Data Directory

This directory contains the project's datasets.

> **⚠️ Data files are excluded from version control** via `.gitignore` because
> they are too large for GitHub.

## Expected Structure

```
data/
├── raw/
│   ├── AbdomenAtlas3.0MiniWithMeta.csv    # Source metadata (9,262 rows)
│   └── pdf_reports/                       # One PDF per BDMAP ID (9,262 files)
│       ├── BDMAP_00000001.pdf
│       ├── BDMAP_00000002.pdf
│       └── ...
└── processed/
    └── liver_nlp_clean.csv                # Output of prepare_nlp_dataset.py
```

## How to Obtain the Data

1. Download the AbdomenAtlas 3.0 Mini dataset.
2. Place `AbdomenAtlas3.0MiniWithMeta.csv` in `data/raw/`.
3. Extract the PDF reports into `data/raw/pdf_reports/`.
4. Run `python -m src.prepare_nlp_dataset` to generate the processed dataset.

