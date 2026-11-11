# LiverSira AI

LiverSira AI contains two independent data pipelines:

- **NLP:** prepares radiology reports and structured liver-lesion features.
- **CV:** materializes the verified patient-level SMC-LUD ultrasound split.

Reusable code lives in the `liversira_ai` package under `src/`; complete
workflows are run from `scripts/`; committed metrics, reports and predictions
are under `results/`.

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
│   └── nlp/                   # data preparation, parser and baseline evaluation
│       ├── learned_presence/  # classical presence model experiments
│       └── transformers/      # transformer experiments
├── notebooks/
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

## Setup

Create `.venv`, then install the project in editable mode together with the
development tools. Dependencies are declared once, in `pyproject.toml`;
`requirements.txt` only points at it.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The editable install makes `liversira_ai` importable everywhere, so no
`PYTHONPATH` is needed. Scripts use paths relative to the repository root, so
run them from there.

### Bytecode

The project runs with `PYTHONDONTWRITEBYTECODE=1`, so Python writes no
`__pycache__/` directories or `.pyc` files:

- `.env` (copied from `.env.example`, gitignored) holds the variable.
- VS Code: `.vscode/settings.json` sets `python.envFile` to that `.env` and
  enables `python.terminal.useEnvFile`, so the Python extension applies it to
  new terminals, test runs, debugging and Jupyter kernels. Reload the window
  after creating `.env`.
- Any other shell: set it once per session before running Python.

  ```powershell
  $env:PYTHONDONTWRITEBYTECODE = "1"
  ```

- Notebooks set `sys.dont_write_bytecode = True` in their first cell, which
  also covers Colab.

`__pycache__/` and `*.pyc` stay in `.gitignore` as a backstop.

## GPU setup

The editable install on Windows resolves to the CPU-only PyTorch build. To train on an NVIDIA GPU, replace it with
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
python scripts/cv/materialize_cv_split.py
```

Prepare the NLP dataset and frozen study-level splits:

```powershell
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

Transformer experiments (Phase 6) are in `scripts/nlp/transformers/`. Models
are pinned in `liversira_ai.nlp.transformer_models`; Test is never read. Presence uses one shared
`presence/config.json` for BERT, BioBERT and ClinicalBERT:

```powershell
.venv\Scripts\python.exe scripts/nlp/transformers/token_lengths.py                  # truncation policy
.venv\Scripts\python.exe scripts/nlp/transformers/presence/build_challenge_set.py   # from Validation only
.venv\Scripts\python.exe scripts/nlp/transformers/presence/train.py --model bert --seed 42
.venv\Scripts\python.exe scripts/nlp/transformers/presence/run_all.py               # all models x seeds, resumable
.venv\Scripts\python.exe scripts/nlp/transformers/presence/compare_models.py        # vs LR and parser v1.1
```

Weights go to `models/transformers/` (gitignored, safetensors, hashed in the
run manifest); predictions and metrics go to `results/nlp/transformers/`.

The frozen learned-presence model
`results/nlp/final_test/learned_presence_word_logreg_final.joblib` is
gitignored. Verify any copy against its SHA-256 before use (also recorded in
`learned_presence_word_logreg_final_model_manifest.json` next to it):

```text
333d4e2aed0c93e3add9934e662b4461ae0db6bf6c8de49eb04f3a61f49ea19a
```

Known limitations of the committed results:

- `results/cv/preprocessing_contact_sheet.png` is generated by
  `notebooks/cv/04_image_preprocessing.ipynb` and shows SMC-LUD ultrasound
  frames, so it is gitignored and not distributed with the repository.
  `results/cv/preprocessing_report.md` links to it; regenerate it locally by
  running the notebook.
- SHA-256 values recorded in result manifests were computed on a Windows
  checkout, where Git stores text files with LF line endings but writes them
  to disk with CRLF. On a checkout with LF line endings (Linux, macOS, or
  `core.autocrlf=false`) the same files hash differently, so verify recorded
  hashes on Windows or after converting line endings.

Run tests:

```powershell
pytest
```

Interactive CV and NLP workflows are in `notebooks/`; they display tables,
figures, and metrics inline and do not generate reports or visualization
files.

### CV preprocessing

Two preprocessing implementations are kept, under distinct names:

- `src/liversira_ai/cv/data/preprocessing.py` is the canonical one: native
  aspect ratio, shorter side resized to 224, a conservative trim of solid
  near-black edge strips, no padding, no crop, no stretching
  (`preprocess_image`, `NativeAspectUltrasoundDataset`). Notebooks 04 and 05
  import it. **It has no recorded results yet**: no run of it has saved
  metrics, cell outputs or a checkpoint in this repository.
- `src/liversira_ai/cv/data/letterbox.py` is the earlier square letterbox
  pipeline (`preprocess_image_letterbox`, `SMCLUDDataset`), kept for
  reference. Its settings are recorded in
  `docs/cv/legacy_letterbox_preprocessing_config.yaml`, which no code reads.

What the existing CV numbers do and do not show:

- The only ResNet18 numbers on record (the "Historical Letterbox baseline"
  table in notebook 05) come from the letterbox pipeline. That pipeline has a
  documented label shortcut: it added about 19.8% black padding to Hemangioma
  images and none to HCC images, because every HCC image is square and no
  Hemangioma image is. Those numbers are therefore not evidence that the
  model learned lesion appearance.
- That baseline table and the shortcut-audit table in notebook 04 were
  entered by hand. The runs and code that produced them are not in this
  repository, so they cannot be reproduced from it.
- `results/cv/preprocessing_report.md` describes the letterbox pipeline and
  was written by a script that has since been removed.
- No model has been trained or evaluated on the CV Test split. Test images
  were decoded and checksummed for inspection only, by an earlier committed
  version of notebook 04 and by the removed report script. The current
  notebooks 04 and 05 read Train and Validation only.

Notebook workflow:

- `notebooks/cv/04_image_preprocessing.ipynb` — native-aspect preprocessing:
  Train/Validation split checks, comparison with the old letterbox, and
  tensor checks.
- `notebooks/cv/05_baseline_training.ipynb` — ResNet18 baseline on the
  native-aspect input (frozen backbone, patient-weighted loss, image-level
  and patient-level Validation metrics). The training loop and model builder
  are defined in the notebook; it saves the best checkpoint under
  `models/cv/` (gitignored). The Test split is not read.
- `notebooks/nlp/01_dataset_audit_and_cleaning.ipynb` — raw AbdomenAtlas
  inspection and pure text-cleaning checks.
- `notebooks/nlp/02_rule_based_parser_and_inference.ipynb` — parser evidence,
  uncertainty, and structured baseline inference inspection.
- `notebooks/nlp/03_evaluation_and_error_analysis.ipynb` — existing baseline
  and learned-presence metrics, split integrity, and inline error inspection.

All tests are in `tests/` and run with `pytest`. There is no CV evaluation
pipeline yet.

### Learned presence scripts

The classical presence experiments are in `scripts/nlp/learned_presence/`.
These scripts import each other, so run them as modules from the repository
root, not as files:

```powershell
python -m scripts.nlp.learned_presence.train_learned_presence
python -m scripts.nlp.learned_presence.compare_learned_presence
python -m scripts.nlp.learned_presence.export_frozen_model
python -m scripts.nlp.learned_presence.regenerate_audit_groups --check   # read-only comparison with results/
```

`final_evaluate_learned_presence` is the single-use Test evaluation of the
frozen model. It has already been run; do not run it again. An untracked
local copy of its predictions,
`results/nlp/final_test/learned_presence_test_predictions.jsonl`, is dated
10 November 2026; its predictions are identical to the tracked
`results/nlp/predictions/learned_presence_test_predictions.jsonl`. The rule-based
baseline scripts (`evaluate_baseline.py`, `error_analysis.py`,
`create_benchmark_manifest.py`) are in `scripts/nlp/`. Their metrics, reports,
predictions and configurations are preserved under `results/nlp/`.

### Old script paths

Scripts were reorganised after the committed results were produced. Result
files are left unchanged, so some still name the old paths (for example a
`created_by` field). Use this mapping:

| Old path | Current path |
|---|---|
| `evaluation/baseline/*.py` | `scripts/nlp/*.py` |
| `evaluation/learned_presence/*.py` | `scripts/nlp/learned_presence/*.py` |
| `evaluation/learned_presence/utils.py` | `src/liversira_ai/nlp/learned_presence.py` |
| `evaluation/regenerate_audit_groups.py` | `scripts/nlp/learned_presence/regenerate_audit_groups.py` |
| `evaluation/tests/` | `tests/` |
| `experiments/transformers/token_lengths.py` | `scripts/nlp/transformers/token_lengths.py` |
| `experiments/transformers/presence/*` | `scripts/nlp/transformers/presence/*` |
| `experiments/transformers/data.py` | `src/liversira_ai/nlp/transformer_data.py` |
| `experiments/transformers/models.py` | `src/liversira_ai/nlp/transformer_models.py` |

The materializer is idempotent. Existing files are accepted only when their
checksums match the manifest; conflicting files fail instead of being silently
overwritten.

## Licence and data notice

The source code in this repository is released under the MIT License (see
`LICENSE`).

The MIT License does not cover data. Some files under `results/nlp/`,
`tests/fixtures/` and the saved outputs of `notebooks/nlp/` contain text
excerpts and study identifiers (`BDMAP_…`) from the radiology reports of
AbdomenAtlas 3.0 (Bassi et al., "RadGPT: Constructing 3D Image-Text Tumor
Datasets", ICCV 2025), distributed as AbdomenAtlas 3.0 Mini under the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International licence
(CC BY-NC-SA 4.0). Those excerpts, and any file that reproduces them, remain
under CC BY-NC-SA 4.0: they may be shared and adapted for non-commercial
purposes only, with attribution to the AbdomenAtlas authors, and under the
same licence. Complete reports are not distributed, but result files contain
many verbatim excerpts of the liver-related sentences.

This repository does not include the SMC-LUD dataset itself; obtain it from
its Figshare record and follow its terms.
