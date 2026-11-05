# LiverSira AI

LiverSira AI contains two independent data pipelines:

- **NLP:** prepares radiology reports and structured liver-lesion features.
- **CV:** materializes the verified patient-level SMC-LUD ultrasound split.

The repository does not include model training or redundant active modules.
Historical evaluation code is kept under `evaluation/`, with its results under
`results/`.

## Structure

```text
LiverSira AI/
├── data/
│   ├── raw/
│   │   ├── cv/SMC-LUD/
│   │   └── nlp/
│   ├── manifests/cv/SMC-LUD/
│   │   └── patient_level_manifest.csv
│   └── processed/
│       ├── cv/SMC-LUD/patient_split_v1/
│       └── nlp/
├── src/
│   ├── liversira_ai/
│   │   ├── common/
│   │   ├── cv/data/
│   │   └── nlp/
├── scripts/
│   ├── cv/materialize_cv_split.py
│   └── nlp/
├── evaluation/
│   └── tests/                 # Historical evaluation tests; not active pytest tests
├── results/
├── docs/
└── tests/
```

## Data locations

Place the AbdomenAtlas CSV and PDF reports under `data/raw/nlp/`.
The original SMC-LUD dataset belongs under `data/raw/cv/SMC-LUD/`.

The CV manifest at
`data/manifests/cv/SMC-LUD/patient_level_manifest.csv` is the source of truth
for the physical split. The original images are never moved or modified.

## Commands

Run the physical CV split from the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/cv/materialize_cv_split.py
```

Prepare the NLP dataset and frozen study-level splits:

```powershell
$env:PYTHONPATH = "src"
python scripts/nlp/prepare_dataset.py
```

Run tests:

```powershell
pytest
```

The reusable PyTorch dataset and DataLoader are in
`src/liversira_ai/cv/data/preprocessing.py`. Interactive CV and NLP workflows
are in `notebooks/`; they display tables, figures, and metrics inline and do
not generate reports or visualization files.

Notebook workflow:

- `notebooks/cv/04_image_preprocessing.ipynb` — CV image validation, resize
  comparison, tensor/normalization checks, and DataLoader inspection.
- `notebooks/cv/05_baseline_training.ipynb` — controlled ResNet18 transfer
  learning at 512x512 and 224x224 with validation-only comparison; the Test
  split is intentionally untouched.
- `notebooks/nlp/01_dataset_audit_and_cleaning.ipynb` — raw AbdomenAtlas
  inspection and pure text-cleaning checks.
- `notebooks/nlp/02_rule_based_parser_and_inference.ipynb` — parser evidence,
  uncertainty, and structured baseline inference inspection.
- `notebooks/nlp/03_evaluation_and_error_analysis.ipynb` — existing baseline
  and learned-presence metrics, split integrity, and inline error inspection.

The active test suite is only `tests/`. Pytest is configured with
`testpaths = ["tests"]`, so `evaluation/tests/` is not collected by default.
The evaluation directory contains historical evaluation implementations and
tests; it does not represent a completed CV evaluation pipeline. CV evaluation
will be added after the model is built.

Historical evaluation implementations are in `evaluation/`; their metrics,
reports, predictions, and configurations are preserved under `results/nlp/`.
The active runtime package is only `src/liversira_ai/`.

The materializer is idempotent. Existing files are accepted only when their
checksums match the manifest; conflicting files fail instead of being silently
overwritten.
