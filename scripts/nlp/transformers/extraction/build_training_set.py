"""Build the Phase 6.2 extraction training data: real Train reports plus augmentation.

Writes, under data/processed/nlp/extraction/ (gitignored):
- extraction_train_v1.jsonl: every Train study (liver-only text, targets from
  ``liversira_ai.nlp.extraction_targets``) plus generated items;
- extraction_val_v1.jsonl: every Validation study, real only, same targets.
Test is not read.

Generated items insert a lesion description into the liver-only text of a
lesion-free Train report (same eligibility and insertion rules as the Challenge
Set), so their targets are known by construction. They cover the phenomena the
Challenge Set tests (count formats, sizes, segment formats, attenuation wording,
largest-lesion order, paraphrase, OCR noise) with a separate lexicon and sentence
templates, so the Challenge Set stays a test of generalization:

- count: "N lesions are present", "N in total", "Lesion 1: ... Lesion 2: ...",
  "One ... Another ... A further ...", "Only one ...", "Innumerable / A few ...";
- size: single mm, "a cm by b cm", "greatest diameter of", "a x b x c cm in size",
  "reaching" / "as large as";
- segments: "liver segment VII", "segment 4A" / "subsegment IVb", "segment 6
  according to Couinaud", "the 4th segment of the liver" / "segment four",
  "extending from segment 5 into segment 6", "the right liver lobe";
- attenuation: "appearing hypodense", "of low density", "darker than the adjacent
  hepatic tissue", "It appears ...";
- order: size-listed lesions, "the larger of the two", "In segment 3 there is ...";
- paraphrase: "focus", "tumor", "space-occupying lesion"; "most prominent",
  "greatest", "principal";
- ocr: the letter-noise model of the Challenge Set applied to these templates and
  to real Train reports.

The build fails if a generated item contains a 4-gram that occurs in the Challenge
Set's inserted text and nowhere in the real Train reports (numbers, segment
numerals, number words and the three attenuation terms masked).
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from liversira_ai.nlp.data.raw_csv import ATTENUATION, COUNT, DIAMETER, ID, LOCATION
from liversira_ai.nlp.extraction_targets import (
    TARGETS_VERSION,
    targets_from_expected,
    targets_from_source,
)
from scripts.nlp.transformers.extraction import build_challenge_set as cs

VERSION = "extraction_train_v1"
SEED = 1337  # not the Challenge Set seed
PER_CATEGORY = 500
REAL_OCR_ITEMS = 500
OCR_RATE = 0.12
NGRAM = 4
OUT_DATA = Path("data/processed/nlp/extraction")
TRAIN_PATH = OUT_DATA / f"{VERSION}.jsonl"
VAL_PATH = OUT_DATA / "extraction_val_v1.jsonl"
OUT_DIR = Path("results/nlp/transformers/extraction")
CHALLENGE_VERSIONS = (cs.VERSION, cs.LIVER_VERSION)

OPEN, CLOSE = "⟦", "⟧"  # mark the largest lesion's size phrase while generating

Lesion, Spec = cs.Lesion, cs.Spec
WORDS, ORDINALS, ROMAN = cs.WORDS, cs.ORDINALS, cs.ROMAN
ORDINAL_SUFFIX = {1: "1st", 2: "2nd", 3: "3rd"}


# ── Phrasing helpers (training lexicon) ────────────────────────────────────
def mark(phrase: str) -> str:
    return f"{OPEN}{phrase}{CLOSE}"


def fmt(value: float) -> str:
    return cs.fmt(value)


def dims(lesion: Lesion) -> str:
    return cs.dims_text(lesion.dims)


def seg(segments: tuple[int, ...]) -> str:
    if len(segments) == 1:
        return f"segment {segments[0]}"
    return f"segments {segments[0]}/{segments[1]}"


def expected(spec: Spec, **overrides: Any) -> dict[str, Any]:
    return cs.expected_fields(spec, overrides)


Generated = tuple[str, dict[str, Any]]


# ── count ───────────────────────────────────────────────────────────────────
def count_present(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=9)
    big = spec.largest
    n = rng.choice((str(len(spec.lesions)), WORDS[len(spec.lesions)]))
    text = (
        f"{n[0].upper() + n[1:]} {big.attenuation} lesions are present in the liver. "
        f"The largest, in {seg(big.segments)}, measures {mark(dims(big))}."
    )
    return text, expected(spec)


def count_in_total(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=9)
    big = spec.largest
    text = (
        f"The liver shows {big.attenuation} lesions, {len(spec.lesions)} in total; the largest lies in "
        f"{seg(big.segments)} with a size of {mark(dims(big))}."
    )
    return text, expected(spec)


def count_listed(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=5, shared_attenuation=False)
    order = list(spec.lesions)
    rng.shuffle(order)
    parts = []
    for i, lesion in enumerate(order, start=1):
        size = mark(dims(lesion)) if lesion is spec.largest else dims(lesion)
        parts.append(f"Lesion {i}: {seg(lesion.segments)}, {size}, {lesion.attenuation}.")
    return " ".join(parts), expected(spec)


def count_another(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    leads = ["One", "Another", "A further", "Yet another"]
    parts = []
    for i, lesion in enumerate(spec.lesions):
        size = mark(dims(lesion)) if i == 0 else dims(lesion)
        noun = f"{lesion.attenuation} lesion" if i == 0 else "lesion"
        parts.append(f"{leads[i]} {noun} is present in {seg(lesion.segments)} ({size}).")
    return " ".join(parts), expected(spec)


def count_only_one(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = rng.choice(
        (
            f"Only one {big.attenuation} lesion is present, in {seg(big.segments)}, with a size of {mark(dims(big))}.",
            f"There is a lone {big.attenuation} lesion in {seg(big.segments)} of {mark(dims(big))}.",
        )
    )
    return text, expected(spec)


def count_vague(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=3, max_count=9)
    big = spec.largest
    word = rng.choice(("Innumerable", "A few", "Scattered", "Various"))
    text = (
        f"{word} {big.attenuation} lesions are present throughout the liver; the largest, in "
        f"{seg(big.segments)}, has a size of {mark(dims(big))}."
    )
    return text, expected(spec, lesion_count=None)


# ── size ────────────────────────────────────────────────────────────────────
def _one(spec: Spec, size: str) -> str:
    big = spec.largest
    return f"A {big.attenuation} lesion is present in {seg(big.segments)}, with a size of {mark(size)}."


def size_single_mm(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = f"A {big.attenuation} lesion of {mark(str(round(big.dims[0] * 10)) + ' mm')} lies in {seg(big.segments)}."
    return text, expected(spec)


def size_cm_by_cm(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    a, b = big.dims
    text = f"A {big.attenuation} lesion of {mark(f'{fmt(a)} cm by {fmt(b)} cm')} is present in {seg(big.segments)}."
    return text, expected(spec)


def size_greatest_diameter(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"A {big.attenuation} lesion located at {seg(big.segments)} has a greatest diameter of "
        f"{mark(fmt(big.dims[0]) + ' cm')}."
    )
    return text, expected(spec)


def size_three_in_size(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    third = max(0.1, round(rng.uniform(0.4, 0.9) * big.dims[1], 1))
    lesion = Lesion(big.segments, (*big.dims, third), big.attenuation)
    spec = Spec((lesion,))
    size = " x ".join(f"{fmt(d)} cm" for d in lesion.dims)  # unit after each dimension
    text = f"There is a {lesion.attenuation} lesion at {seg(lesion.segments)}, {mark(size)} in size."
    return text, expected(spec)


def size_reaching(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    verb = rng.choice(("reaching", "as large as"))
    text = f"A {big.attenuation} lesion at {seg(big.segments)}, {verb} {mark(fmt(big.dims[0]) + ' cm')}, is present."
    return text, expected(spec)


# ── segments ────────────────────────────────────────────────────────────────
def _at(spec: Spec, where: str, **overrides: Any) -> Generated:
    big = spec.largest
    text = f"A {big.attenuation} lesion of {mark(dims(big))} is present {where}."
    return text, expected(spec, **overrides)


def _single_segment(rng) -> Spec:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    return Spec((Lesion(big.segments[:1], big.dims, big.attenuation),))


def segment_roman(rng) -> Generated:
    spec = _single_segment(rng)
    s = spec.largest.segments[0]
    where = rng.choice((f"in liver segment {ROMAN[s]}", f"(segment {ROMAN[s]})"))
    return _at(spec, where)


def segment_subsegment(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    spec = Spec((Lesion((4,), big.dims, big.attenuation),))
    where = rng.choice(("in segment 4A", "in segment 4B", "in subsegment IVa", "in subsegment IVb"))
    return _at(spec, where)


def segment_couinaud(rng) -> Generated:
    spec = _single_segment(rng)
    return _at(spec, f"in segment {spec.largest.segments[0]} according to Couinaud")


def segment_ordinal(rng) -> Generated:
    spec = _single_segment(rng)
    s = spec.largest.segments[0]
    where = rng.choice(
        (
            f"in the {ORDINAL_SUFFIX.get(s, f'{s}th')} segment of the liver",
            f"within segment {WORDS[s]}",
            f"in the {ORDINALS[s]} segment of the liver",
        )
    )
    return _at(spec, where)


def segment_range(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    first = rng.randint(1, 7)
    spec = Spec((Lesion((first, first + 1), big.dims, big.attenuation),))
    where = rng.choice(
        (f"extending from segment {first} into segment {first + 1}", f"spanning segments {first} to {first + 1}")
    )
    return _at(spec, where)


def segment_lobe(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    where = rng.choice(
        ("in the right liver lobe", "in the left liver lobe", "within the right lobe", "within the left lobe")
    )
    return _at(spec, where, location_segments=None)


# ── attenuation ─────────────────────────────────────────────────────────────
DENSE = {"hypoattenuating": "hypodense", "hyperattenuating": "hyperdense", "isoattenuating": "isodense"}
OF_DENSITY = {
    "hypoattenuating": ("of low density", "with low attenuation"),
    "hyperattenuating": ("of high density", "with high attenuation"),
    "isoattenuating": ("of equal density", "with equal attenuation"),
}
RELATIVE = {
    "hypoattenuating": ("darker than the adjacent hepatic tissue", "less dense than background liver"),
    "hyperattenuating": ("brighter than the adjacent hepatic tissue", "denser than background liver"),
    "isoattenuating": ("as dense as the adjacent hepatic tissue", "indistinguishable in density from background liver"),
}
SEPARATE = {
    "hypoattenuating": ("It appears hypoattenuating.", "Its attenuation is low."),
    "hyperattenuating": ("It appears hyperattenuating.", "Its attenuation is high."),
    "isoattenuating": ("It appears isoattenuating.", "Its attenuation matches the liver."),
}


def attenuation_appearing(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = rng.choice(
        (
            f"A lesion appearing {DENSE[big.attenuation]} is present in {seg(big.segments)}, with a size of {mark(dims(big))}.",
            f"{seg(big.segments).capitalize()} contains a {DENSE[big.attenuation]} focus of {mark(dims(big))}.",
        )
    )
    return text, expected(spec)


def attenuation_of_density(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    phrase = rng.choice(OF_DENSITY[big.attenuation])
    text = f"A lesion {phrase} is present in {seg(big.segments)}, with a size of {mark(dims(big))}."
    return text, expected(spec)


def attenuation_relative(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    phrase = rng.choice(RELATIVE[big.attenuation])
    text = f"In {seg(big.segments)} there is a {mark(dims(big))} lesion, {phrase}."
    return text, expected(spec)


def attenuation_separate(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"There is a lesion in {seg(big.segments)} with a size of {mark(dims(big))}. "
        + rng.choice(SEPARATE[big.attenuation])
    )
    return text, expected(spec)


# ── largest-lesion order ────────────────────────────────────────────────────
def order_size_list(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    items = sorted(spec.lesions, key=lambda lesion: lesion.dims[0])  # ascending: the largest comes last
    if rng.random() < 0.5:
        rng.shuffle(items)
    listed = [
        f"{seg(lesion.segments)} ({mark(dims(lesion)) if lesion is spec.largest else dims(lesion)})"
        for lesion in items
    ]
    text = f"{spec.largest.attenuation.capitalize()} lesions are seen in {cs._join(listed)}."
    return text, expected(spec)


def order_larger_of_two(rng) -> Generated:
    spec = cs.sample_spec(rng, count=2, shared_attenuation=False)
    big, small = spec.lesions
    text = (
        f"A {small.attenuation} lesion of {dims(small)} lies in {seg(small.segments)}. "
        f"The larger of the two, {big.attenuation}, lies in {seg(big.segments)} with a size of {mark(dims(big))}."
    )
    return text, expected(spec)


def order_there_is(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4, shared_attenuation=False)
    order = list(spec.lesions)
    rng.shuffle(order)
    parts = [
        f"in {seg(lesion.segments)}, a {mark(dims(lesion)) if lesion is spec.largest else dims(lesion)} "
        f"{lesion.attenuation} lesion"
        for lesion in order
    ]
    return "There are lesions " + "; ".join(parts) + ".", expected(spec)


# ── paraphrase ──────────────────────────────────────────────────────────────
NOUNS = ("focus", "tumor", "space-occupying lesion")
LEADS = ("most prominent", "greatest", "principal")


def paraphrase_single(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    noun = rng.choice(NOUNS)
    text = rng.choice(
        (
            f"{seg(big.segments).capitalize()} is the site of a {mark(dims(big))} {big.attenuation} {noun}.",
            f"A {big.attenuation} {noun} sits in {seg(big.segments)}; size {mark(dims(big))}.",
        )
    )
    return text, expected(spec)


def paraphrase_multiple(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=7)
    big = spec.largest
    noun = rng.choice(("foci", "tumors"))
    lead = rng.choice(LEADS)
    text = (
        f"The liver holds {WORDS[len(spec.lesions)]} {big.attenuation} {noun}. The {lead} of these sits in "
        f"{seg(big.segments)}, sized {mark(dims(big))}; the others are under {fmt(big.dims[0] - cs.MIN_GAP_CM)} cm."
    )
    return text, expected(spec)


CATEGORIES: dict[str, dict[str, Callable[[random.Random], Generated]]] = {
    "count": {
        "present": count_present,
        "in_total": count_in_total,
        "listed": count_listed,
        "another": count_another,
        "only_one": count_only_one,
        "vague": count_vague,
    },
    "size": {
        "single_mm": size_single_mm,
        "cm_by_cm": size_cm_by_cm,
        "greatest_diameter": size_greatest_diameter,
        "three_in_size": size_three_in_size,
        "reaching": size_reaching,
    },
    "segment": {
        "roman": segment_roman,
        "subsegment": segment_subsegment,
        "couinaud": segment_couinaud,
        "ordinal": segment_ordinal,
        "range": segment_range,
        "lobe": segment_lobe,
    },
    "attenuation": {
        "appearing": attenuation_appearing,
        "of_density": attenuation_of_density,
        "relative": attenuation_relative,
        "separate": attenuation_separate,
    },
    "order": {"size_list": order_size_list, "larger_of_two": order_larger_of_two, "there_is": order_there_is},
    "paraphrase": {"single": paraphrase_single, "multiple": paraphrase_multiple},
}
GENERATORS = [(c, s, f) for c, subtypes in CATEGORIES.items() for s, f in subtypes.items()]


# ── Markers and OCR ─────────────────────────────────────────────────────────
def unmark(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove the markers; return the text and the marked spans in it."""
    spans, out, cursor = [], [], 0
    for match in re.finditer(f"{OPEN}(.*?){CLOSE}", text, re.S):
        out.append(text[cursor : match.start()])
        start = sum(len(p) for p in out)
        out.append(match.group(1))
        spans.append((start, start + len(match.group(1))))
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), spans


def mark_spans(text: str, spans: list[tuple[int, int]]) -> str:
    out, cursor = [], 0
    for s, e in sorted(spans):
        out += [text[cursor:s], OPEN, text[s:e], CLOSE]
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


# ── Items ───────────────────────────────────────────────────────────────────
def generated_item(base: dict[str, Any], category: str, subtype: str, fn, rng, ocr: bool) -> dict[str, Any]:
    block, exp = fn(rng)
    block = cs.fix_articles(block)
    marked = cs.insert_after_liver(base["liver_text"], block)
    if ocr:
        marked = cs.ocr(marked, rng, OCR_RATE)
    text, spans = unmark(marked)
    item_id = f"aug:{'ocr:' if ocr else ''}{category}:{subtype}:{base['id']}:{rng.randrange(10**6):06d}"
    return {
        "id": item_id,
        "source": "generated",
        "base_id": base["id"],
        "category": "ocr" if ocr else category,
        "subtype": f"{category}/{subtype}" if ocr else subtype,
        "text": text,
        "inserted_text": unmark(block)[0],
        "targets": targets_from_expected(exp, spans),
    }


def real_item(row: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
    text = row["liver_text"]
    targets = targets_from_source(text, row["count"], row["location"], row["diameter"], row["attenuation"])
    item = {"id": row["id"], "source": "real", "base_id": row["id"], "category": "real", "subtype": "real"}
    if rng is not None:
        marked = cs.ocr(mark_spans(text, targets["diameter_spans"]), rng, OCR_RATE)
        text, spans = unmark(marked)
        targets = {**targets, "diameter_spans": spans}
        item.update({"id": f"ocr:{row['id']}", "category": "ocr", "subtype": "real"})
    return {**item, "text": text, "targets": targets}


def build_generated(bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for category, subtypes in CATEGORIES.items():
        pool = list(bases)
        random.Random(f"{SEED}:{category}").shuffle(pool)
        names = list(subtypes)
        for k in range(PER_CATEGORY):
            subtype = names[k % len(names)]
            base = pool[k % len(pool)]
            rng = random.Random(f"{SEED}:{category}:{subtype}:{base['id']}:{k}")
            items.append(generated_item(base, category, subtype, subtypes[subtype], rng, ocr=False))
    rng = random.Random(f"{SEED}:ocr")
    for k in range(PER_CATEGORY):
        category, subtype, fn = GENERATORS[k % len(GENERATORS)]
        base = bases[rng.randrange(len(bases))]
        items.append(generated_item(base, category, subtype, fn, random.Random(f"{SEED}:ocr:{k}"), ocr=True))
    return items


# ── Template disjointness ───────────────────────────────────────────────────
_MASKS = [
    (re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th|[ab])?\b", re.I), "<n>"),
    (re.compile(r"\b(?:I|II|III|IV|V|VI|VII|VIII)[ab]?\b"), "<n>"),
    (re.compile(rf"\b(?:{'|'.join(WORDS[1:] + ORDINALS[1:])})\b", re.I), "<n>"),
    (re.compile(r"\b(?:hypo|hyper|iso)attenuating\b", re.I), "<att>"),
]


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    tokens = re.findall(r"<\w+>|[a-z]+|[^\sa-z]", text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def challenge_specific(challenge_blocks: list[str], real_texts: list[str]) -> set[tuple[str, ...]]:
    dataset = set().union(*(ngrams(t) for t in real_texts))
    return set().union(*(ngrams(b) for b in challenge_blocks)) - dataset


def overlap(items: list[dict[str, Any]], specific: set[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
    hits: dict[tuple[str, ...], int] = {}
    for item in items:
        if item["source"] != "generated":
            continue
        for gram in ngrams(item["inserted_text"]) & specific:
            hits[gram] = hits.get(gram, 0) + 1
    return hits


# ── Main ────────────────────────────────────────────────────────────────────
def load_rows(split_ids: set[str]) -> list[dict[str, Any]]:
    table = pd.read_csv(cs.LIVER_ONLY_CSV, dtype=str, keep_default_na=False)
    table = table[table[ID].isin(split_ids)]
    return [
        {
            "id": r[ID],
            "full_text": r["narrative_full_clean"],
            "liver_text": r["narrative_clean"],
            "count": int(r[COUNT]),
            "location": r[LOCATION],
            "diameter": r[DIAMETER],
            "attenuation": r[ATTENUATION],
        }
        for _, r in table.iterrows()
    ]


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for item in items:
        key = f"{item['category']}/{item['subtype']}"
        entry = by.setdefault(key, {"n": 0, "stated": {f: 0 for f in ("count", "segments", "attenuation", "diameter_cm")}})
        entry["n"] += 1
        for field in entry["stated"]:
            entry["stated"][field] += item["targets"][field] is not None
    counts: dict[str, int] = {}
    for item in items:
        label = str(item["targets"]["count"])
        counts[label] = counts.get(label, 0) + 1
    return {"items": len(items), "by_subtype": by, "count_classes": dict(sorted(counts.items()))}


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def render(manifest: dict[str, Any]) -> str:
    train, val = manifest["train"], manifest["val"]
    lines = [
        f"# Extraction training data ({VERSION})",
        "",
        "Train: every Train study (liver-only text) plus generated items; Validation: every Validation study, "
        "real only. Test was not read. Targets follow `liversira_ai.nlp.extraction_targets` "
        f"v{TARGETS_VERSION}: a source value is a target only when the text states it, else `null` (unstated). "
        f"Text is in `{TRAIN_PATH.as_posix()}` and `{VAL_PATH.as_posix()}` (gitignored).",
        "",
        "## Template disjointness from the Challenge Set",
        "",
        f"{len(manifest['challenge_specific_ngrams'])} masked {NGRAM}-grams occur in the inserted text of "
        f"{', '.join(f'`{v}`' for v in CHALLENGE_VERSIONS)} and nowhere in the real Train reports. "
        f"Generated training items containing any of them: **{manifest['overlap_items']}**.",
        "",
        "## Train",
        "",
        f"{train['items']} items: {manifest['real_train']} real, {manifest['generated']} generated "
        f"(seed {SEED}, {PER_CATEGORY} per category, OCR on {PER_CATEGORY} generated and {REAL_OCR_ITEMS} real), "
        f"from {manifest['eligible_bases']} eligible lesion-free Train bases.",
        "",
        "| Category / subtype | Items | Count stated | Segments stated | Attenuation stated | Diameter stated |",
        "|---|---|---|---|---|---|",
    ]
    for key, entry in train["by_subtype"].items():
        s = entry["stated"]
        lines.append(
            f"| {key} | {entry['n']} | {s['count']} | {s['segments']} | {s['attenuation']} | {s['diameter_cm']} |"
        )
    lines += [
        "",
        "Count classes (`None` = unstated): "
        + ", ".join(f"{k}: {v}" for k, v in train["count_classes"].items()),
        "",
        "## Validation (real only)",
        "",
        f"{val['items']} studies. Count classes: " + ", ".join(f"{k}: {v}" for k, v in val["count_classes"].items()),
        "",
        "## Notes",
        "",
        "- Real targets: count is unstated when a multi-lesion report gives no number (\"multiple lesions\"); "
        "segments, attenuation and diameter are unstated when the text does not contain the source value.",
        "- Counts of 10 or more share the class `10+`.",
        "- Generated lesions use the Challenge Set's sampling of counts, sizes and segments; only the wording differs.",
        "- Distractors are not generated: other-organ text is removed by the liver-only filter before the model runs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file
    from liversira_ai.nlp import transformer_data as data
    from liversira_ai.nlp.liver_text import LIVER_TEXT_VERSION

    split = data.load_split_ids(["train", "val"])
    train_rows = load_rows(set(split["train"]))
    val_rows = load_rows(set(split["val"]))
    bases = sorted(
        (r for r in train_rows if r["count"] == 0 and cs.is_eligible_base(r["full_text"])), key=lambda r: r["id"]
    )

    real_train = [real_item(r) for r in train_rows]
    positives = [r for r in train_rows if r["count"] > 0]
    ocr_rng = random.Random(f"{SEED}:real-ocr")
    real_ocr = [real_item(r, random.Random(f"{SEED}:real-ocr:{r['id']}")) for r in ocr_rng.sample(positives, REAL_OCR_ITEMS)]
    generated = build_generated(bases)
    train_items = real_train + real_ocr + generated
    val_items = [real_item(r) for r in val_rows]

    blocks = []
    for version in CHALLENGE_VERSIONS:
        with cs.challenge_path(version).open(encoding="utf-8") as handle:
            blocks += [json.loads(line)["inserted_text"] for line in handle if line.strip()]
    specific = challenge_specific(blocks, [r["full_text"] for r in train_rows])
    hits = overlap(generated, specific)
    overlap_items = sum(1 for item in generated if ngrams(item["inserted_text"]) & specific)
    if hits:
        shown = sorted(hits.items(), key=lambda kv: -kv[1])[:20]
        raise SystemExit("Generated items share Challenge-specific n-grams:\n" + "\n".join(f"{' '.join(g)}: {n}" for g, n in shown))

    write_jsonl(TRAIN_PATH, train_items)
    write_jsonl(VAL_PATH, val_items)
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "targets_version": TARGETS_VERSION,
        "liver_text_version": LIVER_TEXT_VERSION,
        "test_policy": "Built from Train and Validation only; Test reports were not read.",
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "eligible_bases": len(bases),
        "real_train": len(real_train),
        "real_ocr": len(real_ocr),
        "generated": len(generated),
        "challenge_specific_ngrams": sorted(" ".join(g) for g in specific),
        "overlap_items": overlap_items,
        "train": summarize(train_items),
        "val": summarize(val_items),
        "outputs_sha256": {p.as_posix(): sha256_file(p) for p in (TRAIN_PATH, VAL_PATH)},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p)
            for p in (cs.LIVER_ONLY_CSV, data.SPLITS_PATH, *(cs.challenge_path(v) for v in CHALLENGE_VERSIONS))
        },
        "git_commit": git_commit(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{VERSION}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = render(manifest)
    (OUT_DIR / f"{VERSION}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
