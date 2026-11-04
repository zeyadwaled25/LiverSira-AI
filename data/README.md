# Data Directory

This directory contains the project's datasets.

> **⚠️ Data files are excluded from version control** via `.gitignore` because
> they are too large for GitHub.

## Expected Structure

```
data/
├── raw/
│   ├── cv/SMC-LUD/                        # Original SMC-LUD ultrasound images
│   └── nlp/
│       ├── AbdomenAtlas3.0MiniWithMeta.csv
│       └── pdf_reports/                   # One PDF per BDMAP ID
├── manifests/
│   └── cv/SMC-LUD/                        # CV audit and patient-level manifests
└── processed/
    ├── cv/SMC-LUD/patient_split_v1/       # Physical CV split
    └── nlp/                               # NLP pipeline outputs
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
2. Place `AbdomenAtlas3.0MiniWithMeta.csv` in `data/raw/nlp/`.
3. Extract the PDF reports into `data/raw/nlp/pdf_reports/`.
4. Run the pipeline steps in the main [README](../README.md#running-the-pipeline)
   to generate the `data/processed/` outputs.
