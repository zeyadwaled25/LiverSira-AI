<div align="center">

<img src="docs/media/hero.svg" alt="LiverSira AI: radiology reports to structured liver-lesion features, and liver ultrasound HCC vs Hemangioma" width="100%">

<br>

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.22-FFD21E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-2.14-E92063?logo=pydantic&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-2ea44f)](LICENSE)
[![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/data%20excerpts-CC%20BY--NC--SA%204.0-lightgrey)](#licence-and-data-notice)

**[Overview](#overview)** ·
**[How it works](#how-it-works)** ·
**[Status](#status)** ·
**[Key results](#key-results)** ·
**[Setup](#setup)** ·
**[NLP](#nlp-workflow)** ·
**[CV](#cv-workflow)** ·
**[Results](#results)** ·
**[Licence](#licence-and-data-notice)**

</div>

---

## Overview

LiverSira AI is a graduation project with two separate pipelines:

| | Pipeline | Input | Output |
|---|---|---|---|
| 🧾 | **NLP** | Radiology reports from AbdomenAtlas 3.0 Mini (CSV + PDF) | Structured liver-lesion features: presence, count, Couinaud segments, largest diameter, attenuation, each backed by an evidence span in the report |
| 🩻 | **CV** | SMC-LUD liver ultrasound images | A patient-level split, native-aspect preprocessing and a ResNet18 baseline notebook for HCC vs Hemangioma |

The NLP side compares a rule-based parser (v1.0 → v1.1 → v1.2), a classical
learned presence model (TF-IDF + logistic regression), transformer models
(BERT, BioBERT, ClinicalBERT) for presence and for field extraction, a hybrid
of parser and model, and API LLMs as references. Every candidate is frozen and
hashed before its single final evaluation.

> [!IMPORTANT]
> Every score in this repository is measured against **source-derived
> labels** or labels set by construction, not a clinical gold standard.
> Nothing here is clinical validation, and nothing here justifies autonomous
> clinical use.

## How it works

### From report text to structured features

<p align="center">
  <img src="docs/media/report_to_json.svg" alt="Animation on a synthetic report: other-organ sentences are dropped, then count, attenuation, segment and diameter are highlighted and appear as JSON fields" width="100%">
</p>

1. **Liver-only text** (`liversira_ai.nlp.liver_text`, v1.0.1) cuts the report
   into sections, units and clauses and keeps only the clauses about the
   liver, with offsets back into the source text.
2. **Extraction** reads that text. Parser v1.2 is deterministic and returns an
   evidence span for every value; the ClinicalBERT extractor (C3) predicts
   count, segments, attenuation and diameter; hybrid **H** takes the parser's
   value and falls back to the model only when the parser leaves a field
   empty (`liversira_ai.nlp.hybrid.hybrid_h`).
3. **Output** follows the Pydantic schemas in `liversira_ai.nlp.schemas`:
   ternary presence (true / false / unknown), the largest-lesion fields, and
   evidence spans with character offsets.

The report in the animation is synthetic; it is not taken from the dataset.

### Pipelines

<p align="center">
  <img src="docs/media/pipeline_flow.svg" alt="Animated diagram of the NLP and CV pipelines, from raw data to frozen candidates and the single Phase 8 evaluation" width="100%">
</p>

| Id | Frozen candidate | Code |
|---|---|---|
| C1 | Parser v1.1 | `liversira_ai.nlp.parser_v1_1` |
| C2 | Parser v1.2 (14 rules, each justified by a Train-aug item) | `liversira_ai.nlp.parser_v1_2`, `parser_v1_2_rules` |
| C3 | First extraction model (ClinicalBERT, seeds 13 / 42 / 87) | `liversira_ai.nlp.extraction_model` |
| C5 | Hybrid H = C2, with C3 filling empty fields | `liversira_ai.nlp.hybrid` |
| P1 | Presence: frozen TF-IDF + LR (default) | `liversira_ai.nlp.learned_presence` |
| P2 | Presence: ClinicalBERT, seeds 13 / 42 / 87 | `liversira_ai.nlp.transformer_models` |
| R1, R2 | Phase 7 API LLMs (Nemotron 3.5 Lightning, Gemini 3.5 Flash-Lite): **references only**, not deployable, not run in Phase 8 | `liversira_ai.nlp.llm_extraction` |

C4 (per-lesion model) and C6 (hybrid H2) were retired by Amendment 3 of the
[Phase 8 analysis plan](docs/nlp/phase8_analysis_plan.md); their results stay
in the repository.

## Status

```mermaid
flowchart LR
    A[Data prep<br/>and audit] --> B[Parser<br/>v1.0 / v1.1]
    B --> C[Learned<br/>presence LR]
    C --> D[Transformer<br/>presence]
    D --> E[6.2 extraction<br/>and parser v1.2]
    E --> F[Phase 7<br/>API LLMs]
    F --> G{{Phase 8<br/>single run}}
    classDef done fill:#0f766e,stroke:#5eead4,color:#fff
    classDef wait fill:#78350f,stroke:#fbbf24,color:#fff
    class A,B,C,D,E,F done
    class G wait
```

Implemented and backed by committed results:

- ✅ NLP data preparation, audit, frozen study-level splits.
- ✅ Rule-based parser v1.0 and v1.1, with Validation and single-use Test
  evaluations.
- ✅ Learned presence model (TF-IDF + logistic regression), with Validation and
  single-use Test evaluations and a reliability report.
- ✅ Transformer presence experiments (BERT, BioBERT, ClinicalBERT) on
  Validation and a Challenge Set. They have not been evaluated on Test.
- ✅ Liver-only text filter (v1.0.1) and its retention and leakage checks.
- ✅ Phase 6.2 extraction: Train-aug data, ClinicalBERT extraction models,
  parser v1.2 tuned on Train-aug only, hybrid H, Validation re-scored on
  verifiable fields only, and Challenge-A diagnostics
  ([status](docs/nlp/phase6_2_status.md)).
- ✅ Phase 7 LLM extraction with a frozen prompt, as a reference
  ([report](docs/nlp/phase7_report.md)).
- ✅ Phase 8 candidates and analysis plan frozen and hashed
  (`results/nlp/phase8/candidates_freeze.json`).
- ✅ CV patient-level split materialization and two preprocessing
  implementations.

Waiting, not implemented, or without recorded results:

- ⏳ **Challenge-B**, the held-out extraction test, is waiting for its author
  (brief and validator are ready: `docs/nlp/challenge_b_brief.md`).
- ⏳ The human gold standard for NLP is supported by scripts, but
  `gold_standard.jsonl` is produced only once annotators complete it.
- ⏳ The **Phase 8** final evaluation has not run. Extraction has not been
  evaluated on Test.
- ❌ The canonical native-aspect CV pipeline has **no recorded results yet** (see
  [CV workflow](#cv-workflow)).
- ❌ There is no CV evaluation pipeline and no CV evaluation on Test.
- ❌ There is no API, service or user interface in this repository.

## Key results

All numbers are against source-derived labels; 95% intervals in brackets.

**Single-use Test runs (1390 studies, run once, never re-run):**

| System | Metric | Test |
|---|---|---|
| Parser v1.1 | Presence selective F1 (on answered studies) | 1.000 [1.000, 1.000] |
| Parser v1.1 | Presence coverage | 0.445 [0.419, 0.471] |
| Parser v1.1 | Diameter within 0.5 cm (where predicted) | 0.991 [0.977, 1.000] |
| Parser v1.1 | Count exact (where predicted) | 0.980 [0.960, 0.995] |
| Frozen TF-IDF + LR | Presence F1 (coverage 1.000) | 0.998 |
| Frozen TF-IDF + LR | Presence specificity | 1.000 |

Sources: [`parser_v1_1/test_evaluation.md`](results/nlp/parser_v1_1/test_evaluation.md),
[`final_test/learned_presence_test_report.md`](results/nlp/final_test/learned_presence_test_report.md).

**Extraction on Validation (220 lesion-positive reports, verifiable fields only, development data):**

| System | count | segments | attenuation | diameter |
|---|---|---|---|---|
| C2 parser v1.2 | 1.000 | 0.990 | 0.977 | 1.000 |
| C3 ClinicalBERT extractor | 1.000 | 0.974 | 1.000 | 1.000 |
| C5 hybrid H | 1.000 | 0.995 | 0.986 | 1.000 |

Source: [`phase8/validation_rescored.md`](results/nlp/phase8/validation_rescored.md).
Which system extracts each field is decided only by the single Phase 8 run.

## Structure

```text
LiverSira AI/
├── src/liversira_ai/          # reusable package (installed in editable mode)
│   ├── common/                # paths, hashing and Git provenance helpers
│   ├── cv/data/               # split materialization, preprocessing, letterbox
│   └── nlp/                   # parsers, liver-only text, models, hybrid, LLM client, metrics
│       ├── data/              # raw CSV, audit, gold set, adjudication
│       └── schemas/           # Pydantic schemas (master, ground truth, inference, gold standard)
├── scripts/                   # complete workflows, run from the repository root
│   ├── cv/
│   └── nlp/                   # data preparation, parser and baseline evaluation
│       ├── learned_presence/  # classical presence model experiments
│       ├── llm/               # Phase 7 API LLM extraction
│       └── transformers/
│           ├── presence/      # BERT / BioBERT / ClinicalBERT presence
│           └── extraction/    # 6.2 extraction, parser v1.2, Phase 8 freeze, Challenge-B
├── notebooks/                 # exploration and visual inspection
│   ├── cv/
│   └── nlp/
├── tests/                     # pytest suite
├── results/                   # committed metrics, reports and predictions
├── docs/
│   ├── cv/                    # legacy letterbox configuration record
│   ├── media/                 # README animations (SVG)
│   └── nlp/                   # annotation guidelines, phase plans and reports
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

Requires Python 3.12+. Dependencies are declared once, in `pyproject.toml`.
From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"     # add ,llm for the Phase 7 scripts
Copy-Item .env.example .env
```

- The editable install makes `liversira_ai` importable everywhere. Run every
  script from the repository root.
- `.env` sets `PYTHONDONTWRITEBYTECODE=1`; VS Code applies it automatically.
- **GPU:** the default install is CPU-only PyTorch. For CUDA, reinstall the
  same pinned versions from the CUDA 13.2 index:

  ```powershell
  .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
  ```

- **Colab:** notebooks run from `MyDrive/LiverSira AI/Training/` (with
  `data/`, `src/`, `notebooks/`). Re-upload `src/` whenever it changes.
- **LLM keys** are read from the environment variables named in
  `scripts/nlp/llm/config.json` and are never stored.

## Data

Datasets are not in Git. Place the AbdomenAtlas CSV and PDF reports under
`data/raw/nlp/` and the original SMC-LUD dataset under
`data/raw/cv/SMC-LUD/`. [data/README.md](data/README.md) describes the
expected layout and how each generated file is produced.

## NLP workflow

The workflow rests on one idea: **develop on Train and Validation, freeze,
then evaluate once.** No number from a held-out set is ever used to change a
system.

1. **Prepare once.** Clean the reports, audit them against their source
   labels and freeze study-level Train / Validation / Test splits
   (`scripts/nlp/prepare_dataset.py`).
2. **Read only the liver.** Every extraction candidate reads the liver-only
   text, so a lesion in another organ cannot be mistaken for a liver lesion.
3. **Build several candidates, side by side.** A deterministic parser that
   cites its evidence, a classical TF-IDF + LR model, fine-tuned transformers,
   and a hybrid that keeps the parser's answer and lets the model fill only
   what the parser leaves empty. API LLMs are measured as references.
4. **Stress-test with Challenge Sets.** Synthetic items target known hard
   phrasings (ordering, counting, negation, non-lesion sizes). Sets that were
   seen during development are used for diagnosis only.
5. **Freeze, then evaluate once.** Code, weights, configs and the analysis
   plan are hashed before the final run. Each Test evaluation runs once and
   is never re-run, and changes after a result are recorded as dated
   amendments.
6. **Check against humans.** A gold set annotated by two blinded annotators
   and adjudicated by a clinician is the only clinical reference
   (`scripts/nlp/gold_set.py`).

Scripts live in `scripts/nlp/`; those that import each other run as modules
(`python -m scripts.nlp.…`). Each script's docstring explains how to run it,
and [`results/nlp/README.md`](results/nlp/README.md) maps every report to the
script that wrote it. The plans and decisions are in `docs/nlp/`.

## CV workflow

The CV side rests on two ideas: **split by patient, never by image**, and
**keep the image's own shape**.

1. **Patient-level split.** `patient_level_manifest.csv` is the source of
   truth. `scripts/cv/materialize_cv_split.py` copies the images into the
   split and checks every file against its checksum; the originals are never
   moved or modified.
2. **Native-aspect preprocessing** (`liversira_ai.cv.data.preprocessing`):
   the shorter side is resized to 224 and solid black edge strips are
   trimmed, with no padding, crop or stretching.
3. **ResNet18 baseline** (notebook 05) on Train and Validation only. It has
   **no recorded results yet**.

Why the shape matters: the earlier letterbox pipeline (kept in
`letterbox.py` for reference) padded every Hemangioma image with black bars
and no HCC image, because only the HCC images are square. A model could
learn the padding instead of the lesion, so its historical ResNet18 numbers
are not evidence of lesion recognition. No CV model has been evaluated on
Test.

## Tests

```powershell
pytest
```

## Results

`results/` holds the metrics, reports, predictions and manifests of the
experiments above, kept exactly as produced.
[`results/nlp/README.md`](results/nlp/README.md) says which report to read
first. Two things to know:

- Some result files name script paths from before a reorganisation.
- Recorded SHA-256 values were computed on a Windows checkout (CRLF line
  endings), so verify them on Windows or after converting line endings.

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

The animations in `docs/media/` are drawings; they contain no dataset text or
images.
