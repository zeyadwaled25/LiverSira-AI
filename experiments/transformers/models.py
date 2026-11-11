"""Pinned transformer checkpoints for the Phase 6 experiments.

Every experiment loads models through this registry so that all of them use the
same Hugging Face revision and the same tokenizer casing. BioBERT and
ClinicalBERT ship only ``vocab.txt`` (no tokenizer config), so the library
default (lowercase) would be applied silently; ``lowercase`` is set here from
each checkpoint's vocabulary instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    key: str
    repo_id: str
    revision: str
    lowercase: bool
    description: str


CHECKPOINTS: dict[str, Checkpoint] = {
    c.key: c
    for c in (
        Checkpoint(
            "bert",
            "google-bert/bert-base-uncased",
            "86b5e0934494bd15c9632b12f734a8a67f723594",
            lowercase=True,
            description="General-domain BERT base (uncased)",
        ),
        Checkpoint(
            "biobert",
            "dmis-lab/biobert-base-cased-v1.2",
            "67c9c25b46986521ca33df05d8540da1210b3256",
            lowercase=False,
            description="BioBERT v1.2, PubMed (cased)",
        ),
        Checkpoint(
            "clinicalbert",
            "emilyalsentzer/Bio_ClinicalBERT",
            "d5892b39a4adaed74b92212a44081509db72f87b",
            lowercase=False,
            description="Bio_ClinicalBERT, MIMIC-III notes (cased vocab from BioBERT)",
        ),
    )
}

MAX_LENGTH = 512

# Chosen from results/nlp/transformers/token_lengths.md (Train + Validation only):
# head truncation never changes the presence decision; extraction needs the whole report.
TRUNCATION = {
    "presence": {"strategy": "head", "max_length": MAX_LENGTH},
    "extraction": {"strategy": "chunks", "max_length": MAX_LENGTH, "stride": 382},
}


def load_tokenizer(key: str):
    from transformers import AutoTokenizer

    c = CHECKPOINTS[key]
    return AutoTokenizer.from_pretrained(
        c.repo_id, revision=c.revision, do_lower_case=c.lowercase, use_fast=True
    )
