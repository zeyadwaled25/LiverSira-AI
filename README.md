# LiverSira AI

LiverSira AI is a graduation project with two separate pipelines:

- **NLP:** turns radiology reports (AbdomenAtlas 3.0 Mini) into structured
  liver-lesion features, with a rule-based parser, a classical learned
  presence model and transformer presence experiments.
- **CV:** prepares the SMC-LUD liver ultrasound dataset (HCC vs Hemangioma)
  with a patient-level split and preprocessing, and holds a ResNet18 baseline
  notebook.

## Status

Implemented and backed by committed results:

- NLP data preparation, audit, frozen study-level splits.
- Rule-based parser v1.0 and v1.1, with Validation and single-use Test
  evaluations.
- Learned presence model (TF-IDF + logistic regression), with Validation and
  single-use Test evaluations and a reliability report.
- Transformer presence experiments (BERT, BioBERT, ClinicalBERT) on
  Validation and a Challenge Set. They have not been evaluated on Test.
- CV patient-level split materialization and two preprocessing
  implementations.

Not implemented, or without recorded results:

- The canonical native-aspect CV pipeline has **no recorded results yet** (see
  [CV preprocessing](#cv-preprocessing)).
- There is no CV evaluation pipeline and no CV evaluation on Test.
- The human gold standard for NLP is supported by scripts, but
  `gold_standard.jsonl` is produced only once annotators complete it.
- There is no API, service or user interface in this repository.

## Structure

```text
LiverSira AI/
├── src/liversira_ai/          # reusable package (installed in editable mode)
│   ├── common/                # paths, hashing and Git provenance helpers
│   ├── cv/data/               # split materialization, preprocessing, letterbox
│   └── nlp/                   # parser, baseline, metrics, schemas, data, evaluation helpers
├── scripts/                   # complete workflows, run from the repository root
│   ├── cv/
│   └── nlp/                   # data preparation, parser and baseline evaluation
│       ├── learned_presence/  # classical presence model experiments
│       └── transformers/      # transformer experiments
├── notebooks/                 # exploration and visual inspection
│   ├── cv/
│   └── nlp/
├── tests/                     # pytest suite
├── results/                   # committed metrics, reports and predictions
├── docs/
│   ├── cv/                    # legacy letterbox configuration record
│   └── nlp/                   # gold-set annotation guidelines
├── data/                      # datasets (gitignored; see data/README.md)
└── models/                    # model weights (gitignored)
```

Responsibilities:

- `.py` files in `src/` hold reusable code: preprocessing, parsing, metrics,
  schemas and shared helpers.
- `scripts/` holds entry points that run a complete workflow and import from
  `src/`.
- `.ipynb` notebooks are for exploration, visual checks and experiments; they
  import from `src/` and show their output inline.

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

### GPU

The editable install on Windows resolves to the CPU-only PyTorch build. To
train on an NVIDIA GPU, replace it with the CUDA build of the same pinned
versions (CUDA 13.2 wheels; the NVIDIA driver must support CUDA 13.2 or
newer):

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The check should print `2.15.0+cu132`, `True`, and the GPU name.

### Colab

The notebooks also run on Colab from
`/content/drive/MyDrive/LiverSira AI/Training/`, which must contain `data/`,
`src/` and `notebooks/`. The package is not installed there; each notebook
adds `Training/src` to the import path. The notebooks import their
preprocessing from `src/`, so **upload the current `src/` folder to Drive
whenever it changes**; an outdated copy fails on import or silently runs old
code.

## Data

Datasets are not in Git. Place the AbdomenAtlas CSV and PDF reports under
`data/raw/nlp/` and the original SMC-LUD dataset under
`data/raw/cv/SMC-LUD/`. [data/README.md](data/README.md) describes the
expected layout, what is and is not tracked, and how each generated file is
produced.

## NLP workflows

Prepare the dataset and the frozen study-level splits:

```powershell
python scripts/nlp/prepare_dataset.py
```

Reliability report (bootstrap 95% CIs, coverage-adjusted presence metrics,
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

### Rule-based parser

`liversira_ai.nlp.parser` is v1.1 (organ-scoped clauses, largest-lesion
selection, no count imputation, value-level evidence spans);
`liversira_ai.nlp.parser_v1_0` is the frozen v1.0 used by the audit and kept
for reproducibility. Golden cases are in
`tests/fixtures/parser_v1_1_golden.json`. Validation-only comparison of the
two versions:

```powershell
python scripts/nlp/validate_parser.py
```

`scripts/nlp/evaluate_parser_test.py` is the single-use Test evaluation of
parser v1.1. It has already been run and refuses to run again.

The rule-based baseline scripts (`evaluate_baseline.py`, `error_analysis.py`,
`create_benchmark_manifest.py`) are also in `scripts/nlp/`.

### Learned presence

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
`results/nlp/predictions/learned_presence_test_predictions.jsonl`.

The frozen model `results/nlp/final_test/learned_presence_word_logreg_final.joblib`
is gitignored. Verify any copy against its SHA-256 before use (also recorded
in `learned_presence_word_logreg_final_model_manifest.json` next to it):

```text
333d4e2aed0c93e3add9934e662b4461ae0db6bf6c8de49eb04f3a61f49ea19a
```

### Transformers

Transformer experiments are in `scripts/nlp/transformers/`. Models are pinned
in `liversira_ai.nlp.transformer_models`; Test is never read. Presence uses
one shared `presence/config.json` for BERT, BioBERT and ClinicalBERT:

```powershell
.venv\Scripts\python.exe scripts/nlp/transformers/token_lengths.py                  # truncation policy
.venv\Scripts\python.exe scripts/nlp/transformers/presence/build_challenge_set.py   # from Validation only
.venv\Scripts\python.exe scripts/nlp/transformers/presence/train.py --model bert --seed 42
.venv\Scripts\python.exe scripts/nlp/transformers/presence/run_all.py               # all models x seeds, resumable
.venv\Scripts\python.exe scripts/nlp/transformers/presence/compare_models.py        # vs LR and parser v1.1
```

Weights go to `models/transformers/` (gitignored, safetensors, hashed in the
run manifest); predictions and metrics go to `results/nlp/transformers/`.

## CV workflows

Materialize the physical patient-level split from the manifest:

```powershell
python scripts/cv/materialize_cv_split.py
```

The manifest `data/manifests/cv/SMC-LUD/patient_level_manifest.csv` is the
source of truth for the split. The original images are never moved or
modified. The materializer is idempotent: existing files are accepted only
when their checksums match the manifest, and conflicting files fail instead
of being silently overwritten.

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

## Notebooks

Notebooks display tables, figures and metrics inline and write no reports or
figures. Notebook 05 is the one exception: it saves its best checkpoint.

- `notebooks/cv/04_image_preprocessing.ipynb` — native-aspect preprocessing:
  Train/Validation split checks, comparison with the old letterbox, and
  tensor checks.
- `notebooks/cv/05_baseline_training.ipynb` — ResNet18 baseline on the
  native-aspect input (frozen backbone, patient-weighted loss, image-level
  and patient-level Validation metrics). The training loop and model builder
  are defined in the notebook; it saves the best checkpoint under
  `models/cv/` (gitignored). The Test split is not read.
- `notebooks/nlp/01_dataset_audit_and_cleaning.ipynb` — raw AbdomenAtlas
  inspection and text-cleaning checks.
- `notebooks/nlp/02_rule_based_parser_and_inference.ipynb` — parser evidence,
  uncertainty, and structured baseline inference inspection.
- `notebooks/nlp/03_evaluation_and_error_analysis.ipynb` — existing baseline
  and learned-presence metrics, split integrity, and inline error inspection.

## Tests

All tests are in `tests/`:

```powershell
pytest
```

## Results

`results/` holds the metrics, reports, predictions and manifests of the
experiments above. They are kept exactly as produced. Known limitations:

- Scripts were reorganised after the results were produced, so some result
  files name old script paths (see [Old script paths](#old-script-paths)).
- `results/cv/preprocessing_contact_sheet.png` shows SMC-LUD ultrasound
  frames, so it is gitignored and not distributed. It was written by a script
  that has since been removed, so it cannot be regenerated from the current
  code. `results/cv/preprocessing_report.md` links to it.
- SHA-256 values recorded in result manifests were computed on a Windows
  checkout, where Git stores text files with LF line endings but writes them
  to disk with CRLF. On a checkout with LF line endings (Linux, macOS, or
  `core.autocrlf=false`) the same files hash differently, so verify recorded
  hashes on Windows or after converting line endings.

### Old script paths

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
| `configs/cv/preprocessing_config.yaml` | `docs/cv/legacy_letterbox_preprocessing_config.yaml` |
| `src/liversira_ai/cv/data/preprocessing.py` (letterbox) | `src/liversira_ai/cv/data/letterbox.py` |

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

This repository does not include the SMC-LUD dataset, and no ultrasound
images are tracked in the current tree. Older commits in the public history
still contain some (a preprocessing contact sheet and image outputs saved in
an earlier version of notebook 04); the history has not been rewritten.
Obtain the dataset from its Figshare record and follow its terms.
