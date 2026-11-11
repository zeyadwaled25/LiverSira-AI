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

## GPU setup

The project runs in `.venv`. `pip install -r requirements.txt` on Windows
installs the CPU-only PyTorch build. To train on an NVIDIA GPU, replace it with
the CUDA build of the same pinned versions (CUDA 13.2 wheels; the NVIDIA driver
must support CUDA 13.2 or newer):

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The check should print `2.15.0+cu132`, `True`, and the GPU name.

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

NLP reliability (bootstrap 95% CIs, coverage-adjusted presence metrics,
paired learned-vs-rule-based comparison) from saved predictions:

```powershell
python scripts/nlp/reliability_report.py
```

Human gold set (two blinded annotators, agreement, clinician adjudication);
see `docs/nlp/gold_set_annotation_guidelines.md`:

```powershell
python scripts/nlp/gold_set.py sample    # then annotators fill sheets A and B
python scripts/nlp/gold_set.py agree     # then a clinician resolves disagreements
python scripts/nlp/gold_set.py finalize
python scripts/nlp/gold_set.py apply     # writes data/processed/nlp/gold_standard.jsonl
```

Once `gold_standard.jsonl` exists, the reliability report also scores both
systems against it.

Rule-based parser: `liversira_ai.nlp.parser` is v1.1 (organ-scoped clauses,
largest-lesion selection, no count imputation, value-level evidence spans);
`liversira_ai.nlp.parser_v1_0` is the frozen v1.0 used by the audit. Golden
cases are in `tests/fixtures/parser_v1_1_golden.json`. Validation-only
comparison of the two versions:

```powershell
python scripts/nlp/validate_parser.py
```

Transformer experiments (Phase 6) are in `experiments/transformers/`. Models
are pinned in `models.py`; Test is never read. Presence uses one shared
`presence/config.json` for BERT, BioBERT and ClinicalBERT:

```powershell
.venv\Scripts\python.exe experiments/transformers/token_lengths.py                  # truncation policy
.venv\Scripts\python.exe experiments/transformers/presence/build_challenge_set.py   # from Validation only
.venv\Scripts\python.exe experiments/transformers/presence/train.py --model bert --seed 42
```

Weights go to `models/transformers/` (gitignored, safetensors, hashed in the
run manifest); predictions and metrics go to `results/nlp/transformers/`.

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
