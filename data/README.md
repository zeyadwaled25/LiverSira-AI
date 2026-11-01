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
    ├── liver_nlp_clean.csv                # Output of prepare_nlp_dataset.py
    ├── master_records.jsonl               # Output of generate_master_jsonl.py
    ├── ground_truth.jsonl                 # Output of generate_ground_truth_jsonl.py
    ├── target_consistency_audit.csv       # Output of audit_target_consistency.py
    ├── target_validation_review.csv       # Output of validate_targets.py
    ├── splits.json                        # Output of split_dataset.py
    └── dataset_splits.csv                 # Output of split_dataset.py
```

## Data Versioning / External Storage (optional)

The raw and processed datasets are intentionally excluded from Git (see
`.gitignore`) because they are too large for GitHub. Recommended options
for reproducibility:

- **DVC** — version the `data/` directory alongside the code without
  storing binaries in Git. Typical setup:

  ```bash
  pip install dvc
  dvc init
  dvc add data/raw data/processed
  dvc remote add -d storage <your-remote-url>
  dvc push
  ```

- **External storage** — keep the datasets on cloud/object storage
  (S3, GCS, Azure Blob, or a shared drive) and document the download
  location in this README.

Either approach keeps the repository small while keeping the pipeline
reproducible. The code itself is pinned (see `requirements.txt` and
`pyproject.toml`) so the same versions are used across machines.

1. Download the AbdomenAtlas 3.0 Mini dataset.
2. Place `AbdomenAtlas3.0MiniWithMeta.csv` in `data/raw/`.
3. Extract the PDF reports into `data/raw/pdf_reports/`.
4. Run the pipeline steps in the main [README](../README.md#running-the-pipeline)
   to generate the `data/processed/` outputs.

