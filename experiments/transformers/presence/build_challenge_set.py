"""Build the presence Challenge Set (v1) from Validation reports only.

The Validation and Test splits are saturated (the TF-IDF model makes 1-2 errors
per ~1,390 studies), so they cannot separate the models. This set changes the
report text in controlled ways where the correct presence answer is known by
construction, and adds natural hard negatives:

* ``negation``           negative report, liver sentences replaced by an explicit
                         negated lesion statement -> False
* ``other_organ_lesion`` unmodified negative report that describes a lesion in
                         another organ -> False
* ``liver_removed``      positive report with every liver sentence removed; the
                         report no longer states anything about the liver -> False
* ``paraphrase``         lesion vocabulary in liver sentences replaced by synonyms
                         -> label unchanged
* ``ocr_noise``          character-level typos / OCR confusions -> label unchanged
* ``liver_last``         liver sentences moved to the end of the report -> label
                         unchanged (reports of at most 300 words)
* ``hedged``             negative report, liver sentences replaced by a hedged
                         statement -> no correct label; only the positive rate is
                         reported (``scored`` is false)

Only Train is used for training, so no model sees these reports during
training; Validation-based early stopping uses the original reports.

Usage (from the repository root):

    .venv\\Scripts\\python.exe experiments/transformers/presence/build_challenge_set.py
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

VERSION = "presence_challenge_v1"
SEED = 42
PER_LABEL = 75  # two-label categories: 75 positive + 75 negative
SINGLE_LABEL = 150  # one-label categories
LIVER_LAST_MAX_WORDS = 300

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
LIVER = re.compile(r"\b(liver|hepatic|hepato\w*)\b", re.IGNORECASE)
LESION = re.compile(
    r"\b(lesions?|mass(es)?|tumou?rs?|nodules?|cysts?|neoplasms?|metasta\w*|"
    r"hypoattenuat\w*|hyperattenuat\w*|hypodens\w*|hyperdens\w*)\b",
    re.IGNORECASE,
)
OTHER_ORGAN = re.compile(
    r"\b(spleen|splenic|pancrea\w*|kidneys?|renal|adrenal\w*|gallbladder|bladder|colon\w*|"
    r"bowel|stomach|gastric|aort\w*|lungs?|pulmonary)\b",
    re.IGNORECASE,
)
NEGATION_CUE = re.compile(r"\b(no|not|without|negative|free of|absence|absent)\b", re.IGNORECASE)

NEGATED = (
    "No focal liver lesion is identified.",
    "There is no evidence of a hepatic mass or lesion.",
    "The liver is unremarkable, without focal lesions.",
    "No liver lesions are seen.",
    "There are no hypoattenuating or hyperattenuating hepatic lesions.",
)
HEDGED = (
    "A subcentimeter hypodensity in the liver is too small to characterize.",
    "A possible small hepatic lesion cannot be excluded; follow-up imaging is suggested.",
    "There is a questionable liver lesion that may represent artifact.",
)
SYNONYMS = {
    "lesions": ("nodules", "masses", "focal lesions"),
    "lesion": ("nodule", "mass", "focal lesion"),
    "masses": ("lesions", "tumors"),
    "mass": ("lesion", "tumor"),
    "hypoattenuating": ("hypodense", "low-attenuation"),
    "hyperattenuating": ("hyperdense", "high-attenuation"),
    "identified": ("seen", "noted", "demonstrated"),
}
SYNONYM_PATTERN = re.compile(r"\b(" + "|".join(sorted(SYNONYMS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
CONFUSABLE = {"l": "1", "o": "0", "e": "c", "i": "l", "m": "rn", "s": "5"}


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_BREAK.split(text.strip()) if s]


def _liver_indices(sentences: list[str]) -> list[int]:
    """Liver sentences plus the lesion sentences that follow them without naming
    another organ ("This lesion is hypoattenuating."), which refer back to the liver."""
    indices: list[int] = []
    for i, s in enumerate(sentences):
        follows = bool(indices) and indices[-1] == i - 1
        if LIVER.search(s) or (follows and LESION.search(s) and not OTHER_ORGAN.search(s)):
            indices.append(i)
    return indices


def _replace_liver(text: str, replacement: str) -> str | None:
    sentences = split_sentences(text)
    liver = _liver_indices(sentences)
    if not liver:
        return None
    kept = [s for i, s in enumerate(sentences) if i not in set(liver)]
    kept.insert(liver[0], replacement)
    return " ".join(kept)


def negation(text: str, rng: random.Random) -> str | None:
    return _replace_liver(text, rng.choice(NEGATED))


def hedged(text: str, rng: random.Random) -> str | None:
    return _replace_liver(text, rng.choice(HEDGED))


def liver_removed(text: str, rng: random.Random) -> str | None:
    sentences = split_sentences(text)
    liver = set(_liver_indices(sentences))
    kept = [s for i, s in enumerate(sentences) if i not in liver]
    if not liver or not kept:
        return None
    # A remaining lesion sentence without another organ may still refer to the liver.
    if any(LESION.search(s) and not OTHER_ORGAN.search(s) for s in kept):
        return None
    return " ".join(kept)


def _match_case(source: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if source[:1].isupper() else replacement


def paraphrase(text: str, rng: random.Random) -> str | None:
    sentences = split_sentences(text)
    changed = False
    for i in _liver_indices(sentences):
        def swap(match: re.Match) -> str:
            return _match_case(match.group(0), rng.choice(SYNONYMS[match.group(0).lower()]))

        new = SYNONYM_PATTERN.sub(swap, sentences[i])
        changed |= new != sentences[i]
        sentences[i] = new
    return " ".join(sentences) if changed else None


def _corrupt(word: str, rng: random.Random) -> str:
    op = rng.choice(("swap", "drop", "confuse"))
    i = rng.randrange(len(word) - 1)
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    candidates = [j for j, ch in enumerate(word) if ch.lower() in CONFUSABLE]
    if not candidates:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    j = rng.choice(candidates)
    return word[:j] + CONFUSABLE[word[j].lower()] + word[j + 1:]


def ocr_noise(text: str, rng: random.Random, rate: float = 0.06) -> str | None:
    parts = re.split(r"(\W+)", text)
    eligible = [i for i, p in enumerate(parts) if p.isalpha() and len(p) >= 4]
    if not eligible:
        return None
    chosen = [i for i in eligible if rng.random() < rate] or [rng.choice(eligible)]
    for i in chosen:
        parts[i] = _corrupt(parts[i], rng)
    return "".join(parts)


def liver_last(text: str, rng: random.Random) -> str | None:
    if len(text.split()) > LIVER_LAST_MAX_WORDS:
        return None
    sentences = split_sentences(text)
    liver = _liver_indices(sentences)
    if not liver:
        return None
    moved = [s for i, s in enumerate(sentences) if i not in set(liver)] + [sentences[i] for i in liver]
    return " ".join(moved) if moved != sentences else None


def is_other_organ_lesion(text: str) -> bool:
    return any(
        LESION.search(s) and OTHER_ORGAN.search(s) and not LIVER.search(s) and not NEGATION_CUE.search(s)
        for s in split_sentences(text)
    )


def _unchanged(text: str, rng: random.Random) -> str | None:
    return text if is_other_organ_lesion(text) else None


# category -> (transform, base labels to draw from, expected label rule, scored)
CATEGORIES: dict[str, tuple[Callable[[str, random.Random], str | None], tuple[bool, ...], str, bool]] = {
    "negation": (negation, (False,), "false", True),
    "other_organ_lesion": (_unchanged, (False,), "false", True),
    "liver_removed": (liver_removed, (True,), "false", True),
    "paraphrase": (paraphrase, (True, False), "unchanged", True),
    "ocr_noise": (ocr_noise, (True, False), "unchanged", True),
    "liver_last": (liver_last, (True, False), "unchanged", True),
    "hedged": (hedged, (False,), "none", False),
}


def build(examples: list[dict[str, Any]], sha: Callable[[str], str]) -> list[dict[str, Any]]:
    records = []
    for category, (transform, base_labels, rule, scored) in CATEGORIES.items():
        quota = SINGLE_LABEL if len(base_labels) == 1 else PER_LABEL
        for base_label in base_labels:
            pool = sorted((e for e in examples if e["label"] == base_label), key=lambda e: e["id"])
            random.Random(f"{SEED}:{category}:{base_label}").shuffle(pool)
            taken = 0
            for example in pool:
                if taken == quota:
                    break
                text = transform(example["text"], random.Random(f"{SEED}:{category}:{example['id']}"))
                if text is None:
                    continue
                label = {"false": False, "unchanged": base_label, "none": None}[rule]
                records.append(
                    {
                        "challenge_id": f"{category}:{example['id']}",
                        "base_id": example["id"],
                        "category": category,
                        "label": label,
                        "scored": scored,
                        "base_label": base_label,
                        "text": text,
                        "base_text_sha256": sha(example["text"]),
                        "text_sha256": sha(text),
                    }
                )
                taken += 1
    return records


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file, sha256_text
    from liversira_ai.nlp import transformer_data as data

    ids = data.load_split_ids(["val"])["val"]
    examples = data.load_examples(ids)
    records = build(examples, sha256_text)

    data.CHALLENGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with data.CHALLENGE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {}
    for r in records:
        key = {True: "positive", False: "negative", None: "unscored"}[r["label"]]
        counts.setdefault(r["category"], {"positive": 0, "negative": 0, "unscored": 0})[key] += 1
    out_dir = Path("results/nlp/transformers/challenge")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "source_split": "val",
        "test_policy": "Built from Validation only; Test reports were not read.",
        "seed": SEED,
        "quotas": {"per_label": PER_LABEL, "single_label": SINGLE_LABEL},
        "categories": {
            name: {"base_labels": list(base), "expected_label": rule, "scored": scored}
            for name, (_, base, rule, scored) in CATEGORIES.items()
        },
        "counts": counts,
        "records": len(records),
        "output": data.CHALLENGE_PATH.as_posix(),
        "output_sha256": sha256_file(data.CHALLENGE_PATH),
        "inputs_sha256": {
            p.as_posix(): sha256_file(p) for p in (data.MASTER_PATH, data.GROUND_TRUTH_PATH, data.SPLITS_PATH)
        },
        "git_commit": git_commit(),
        "items": [
            {k: r[k] for k in ("challenge_id", "base_id", "category", "label", "scored", "text_sha256")}
            for r in records
        ],
    }
    (out_dir / f"{VERSION}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Presence Challenge Set ({VERSION})",
        "",
        f"Built from Validation reports only (seed {SEED}); Test was not read. The text is in "
        f"`{data.CHALLENGE_PATH.as_posix()}` (gitignored); hashes and IDs are in the manifest.",
        "",
        "| Category | Expected label | Scored | Positive | Negative | Unscored |",
        "|---|---|---|---|---|---|",
    ]
    for name, (_, _, rule, scored) in CATEGORIES.items():
        c = counts.get(name, {"positive": 0, "negative": 0, "unscored": 0})
        lines.append(f"| {name} | {rule} | {scored} | {c['positive']} | {c['negative']} | {c['unscored']} |")
    lines += ["", "Category definitions are in the docstring of "
              "`experiments/transformers/presence/build_challenge_set.py`.", ""]
    (out_dir / f"{VERSION}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
