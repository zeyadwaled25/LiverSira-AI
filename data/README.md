# Data Directory

Everything under `data/` except this file is **excluded from Git** (see
`.gitignore`): the datasets are large and are distributed by their owners
under their own terms.

## Expected structure

```text
data/
├── raw/
│   ├── cv/SMC-LUD/                        # original SMC-LUD ultrasound images
│   └── nlp/
│       ├── AbdomenAtlas3.0MiniWithMeta.csv
│       └── pdf_reports/                   # one PDF per BDMAP ID
├── manifests/
│   └── cv/SMC-LUD/
│       ├── patient_level_manifest.csv     # source of truth for the CV split
│       └── dataset_audit_report.md
└── processed/
    ├── cv/SMC-LUD/patient_split_v1/       # physical CV split (materialized)
    └── nlp/                               # NLP pipeline outputs
```

## NLP data

1. Download the AbdomenAtlas 3.0 Mini dataset
   (CC BY-NC-SA 4.0; see the licence notice in the main
   [README](../README.md#licence-and-data-notice)).
2. Place `AbdomenAtlas3.0MiniWithMeta.csv` in `data/raw/nlp/`.
3. Extract the PDF reports into `data/raw/nlp/pdf_reports/`.
4. From the repository root, generate `data/processed/nlp/`:

   ```powershell
   python scripts/nlp/prepare_dataset.py
   ```

   This writes the cleaned CSV, `master_records.jsonl`, `ground_truth.jsonl`,
   the target-consistency audit and the frozen study-level split
   (`splits.json`). See [NLP workflows](../README.md#nlp-workflows).

The result manifests under `results/nlp/` record SHA-256 values of these
input files, so a regenerated copy can be checked against them (note the
line-ending caveat in the main README).

## CV data

1. Download the SMC-LUD dataset from its Figshare record and follow its
   terms. Place it under `data/raw/cv/SMC-LUD/`.
2. Obtain `data/manifests/cv/SMC-LUD/patient_level_manifest.csv` (see below).
3. From the repository root, materialize the physical split:

   ```powershell
   python scripts/cv/materialize_cv_split.py
   ```

   It copies the images listed in the manifest into
   `data/processed/cv/SMC-LUD/patient_split_v1/`, verifies every checksum, and
   writes `patient_split_metadata.csv` next to them. The original images are
   never moved or modified.

### The patient-level manifest is not tracked

`data/manifests/cv/SMC-LUD/patient_level_manifest.csv` is **gitignored and
is not in this repository**. It lists every clean SMC-LUD image with its
label, patient group, MD5 checksum and assigned split, so it is derived
patient-level data and is not published here.

Known limitation: **the current code cannot regenerate it.** The CV pipeline
only reads the manifest. The script that originally produced it,
`src/create_patient_level_split.py`, was removed from the tree; its last
version is in the Git history and can be read with:

```powershell
git show 77f5f5b:src/create_patient_level_split.py
```

That script was written for an earlier package layout and has not been re-run
or checked against the manifest in use, so treat it as a record of the
method, not as a supported tool. To reproduce the CV split exactly, obtain
the manifest file from the project authors.

## Storage

The raw and processed datasets stay outside Git. Keep them on external or
cloud storage, or version them with a tool such as DVC, and keep the
dependency pins in `pyproject.toml` so the same code versions are used across
machines.
