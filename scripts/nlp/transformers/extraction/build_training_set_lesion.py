"""Build the per-lesion training data (extraction_train_lesion_v1) and Validation labels.

Same real Train reports and the same generated wording as extraction_train_v1,
but every lesion's size phrase is marked (not only the largest), the written
count is marked, and two categories are added:
- repeat: the lesion(s) restated in a closing "IMPRESSION:" line;
- non_lesion: a liver length, a hepatic duct calibre or a lymph node near the
  liver, written in the liver text and not a lesion.

Labels follow ``liversira_ai.nlp.lesion_targets``. Real reports are labelled with
``real_labels``; generated items by construction. Validation is real only. Test
is not read. The build fails if a generated item shares a Challenge-specific
4-gram with extraction_challenge_v1, _v2_liver or _v3.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from liversira_ai.nlp.extraction_targets import targets_from_source
from liversira_ai.nlp.lesion_targets import LESION_TARGETS_VERSION, decode_labels, parse_count, real_labels
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import build_training_set as ts

VERSION = "extraction_train_lesion_v1"
SEED = 1338
PER_CATEGORY = 500
REAL_OCR_ITEMS = 500
OCR_RATE = ts.OCR_RATE
OUT_DATA = ts.OUT_DATA
TRAIN_PATH = OUT_DATA / f"{VERSION}.jsonl"
VAL_PATH = OUT_DATA / "extraction_val_lesion_v1.jsonl"
OUT_DIR = ts.OUT_DIR
CHALLENGE_VERSIONS = (cs.VERSION, cs.LIVER_VERSION, "extraction_challenge_v3")

OPEN, SEP, CLOSE = "⟦", "|", "⟧"
Lesion, Spec = cs.Lesion, cs.Spec
WORDS, ORDINALS, ROMAN = cs.WORDS, cs.ORDINALS, cs.ROMAN
fmt, seg = ts.fmt, ts.seg


def S(i: int, phrase: str) -> str:
    """Mark lesion i's size phrase."""
    return f"{OPEN}S{i}{SEP}{phrase}{CLOSE}"


def N(phrase: str) -> str:
    """Mark the written count."""
    return f"{OPEN}N{SEP}{phrase}{CLOSE}"


def dims(lesion: Lesion) -> str:
    return cs.dims_text(lesion.dims)


Generated = tuple[str, Spec, str, dict[str, Any]]  # marked block, spec, count source, overrides


def _segs(spec: Spec, segments_written: bool = True) -> list[Any]:
    return [list(lesion.segments) if segments_written else None for lesion in spec.lesions]


# ── count ───────────────────────────────────────────────────────────────────
def count_present(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=9)
    big = spec.largest
    n = rng.choice((str(len(spec.lesions)), WORDS[len(spec.lesions)]))
    text = (
        f"{N(n[0].upper() + n[1:])} {big.attenuation} lesions are present in the liver. "
        f"The largest, in {seg(big.segments)}, measures {S(0, dims(big))}."
    )
    return text, spec, "number", {}


def count_in_total(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=9)
    big = spec.largest
    text = (
        f"The liver shows {big.attenuation} lesions, {N(str(len(spec.lesions)))} in total; the largest lies in "
        f"{seg(big.segments)} with a size of {S(0, dims(big))}."
    )
    return text, spec, "number", {}


def count_listed(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=5, shared_attenuation=False)
    order = list(enumerate(spec.lesions))
    rng.shuffle(order)
    parts = [f"Lesion {k}: {seg(lesion.segments)}, {S(i, dims(lesion))}, {lesion.attenuation}."
             for k, (i, lesion) in enumerate(order, start=1)]
    return " ".join(parts), spec, "enumerated", {}


def count_another(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    leads = ["One", "Another", "A further", "Yet another"]
    parts = []
    for i, lesion in enumerate(spec.lesions):
        noun = f"{lesion.attenuation} lesion" if i == 0 else "lesion"
        parts.append(f"{leads[i]} {noun} is present in {seg(lesion.segments)} ({S(i, dims(lesion))}).")
    return " ".join(parts), spec, "enumerated", {}


def count_only_one(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = rng.choice(
        (
            f"Only one {big.attenuation} lesion is present, in {seg(big.segments)}, with a size of {S(0, dims(big))}.",
            f"There is a lone {big.attenuation} lesion in {seg(big.segments)} of {S(0, dims(big))}.",
        )
    )
    return text, spec, "enumerated", {}


def count_vague(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=3, max_count=9)
    big = spec.largest
    word = rng.choice(("Innumerable", "A few", "Scattered", "Various"))
    text = (
        f"{word} {big.attenuation} lesions are present throughout the liver; the largest, in "
        f"{seg(big.segments)}, has a size of {S(0, dims(big))}."
    )
    return text, spec, "unstated", {}


# ── size ────────────────────────────────────────────────────────────────────
def size_single_mm(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = f"There is a {big.attenuation} lesion of {S(0, str(round(big.dims[0] * 10)) + ' mm')} in {seg(big.segments)}."
    return text, spec, "enumerated", {}


def size_cm_by_cm(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    a, b = big.dims
    text = f"There is a {big.attenuation} lesion of {S(0, f'{fmt(a)} cm by {fmt(b)} cm')} at {seg(big.segments)}."
    return text, spec, "enumerated", {}


def size_greatest_diameter(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"There is a {big.attenuation} lesion at {seg(big.segments)} with a greatest diameter of "
        f"{S(0, fmt(big.dims[0]) + ' cm')}."
    )
    return text, spec, "enumerated", {}


def size_three_in_size(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    third = max(0.1, round(rng.uniform(0.4, 0.9) * big.dims[1], 1))
    lesion = Lesion(big.segments, (*big.dims, third), big.attenuation)
    spec = Spec((lesion,))
    size = " x ".join(f"{fmt(d)} cm" for d in lesion.dims)
    text = f"There is a {lesion.attenuation} lesion at {seg(lesion.segments)}, {S(0, size)} in size."
    return text, spec, "enumerated", {}


def size_reaching(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    verb = rng.choice(("reaching", "as large as"))
    text = f"A {big.attenuation} lesion at {seg(big.segments)}, {verb} {S(0, fmt(big.dims[0]) + ' cm')}, is present."
    return text, spec, "enumerated", {}


# ── segments ────────────────────────────────────────────────────────────────
def _at(spec: Spec, where: str, segments_written: bool = True) -> Generated:
    big = spec.largest
    text = f"There is a {big.attenuation} lesion of {S(0, dims(big))} {where}."
    return text, spec, "enumerated", {"segments_written": segments_written}


def _single_segment(rng) -> Spec:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    return Spec((Lesion(big.segments[:1], big.dims, big.attenuation),))


def segment_roman(rng) -> Generated:
    spec = _single_segment(rng)
    s = spec.largest.segments[0]
    return _at(spec, rng.choice((f"in liver segment {ROMAN[s]}", f"(segment {ROMAN[s]})")))


def segment_subsegment(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    spec = Spec((Lesion((4,), big.dims, big.attenuation),))
    return _at(spec, rng.choice(("in segment 4A", "in segment 4B", "in subsegment IVa", "in subsegment IVb")))


def segment_couinaud(rng) -> Generated:
    spec = _single_segment(rng)
    return _at(spec, f"in segment {spec.largest.segments[0]} according to Couinaud")


def segment_ordinal(rng) -> Generated:
    spec = _single_segment(rng)
    s = spec.largest.segments[0]
    where = rng.choice(
        (
            f"in the {ts.ORDINAL_SUFFIX.get(s, f'{s}th')} segment of the liver",
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
    return _at(spec, where, segments_written=False)


# ── attenuation ─────────────────────────────────────────────────────────────
def attenuation_appearing(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    word = ts.DENSE[big.attenuation]
    text = rng.choice(
        (
            f"A lesion appearing {word} is present in {seg(big.segments)}, with a size of {S(0, dims(big))}.",
            f"{seg(big.segments).capitalize()} contains a {word} focus of {S(0, dims(big))}.",
        )
    )
    return text, spec, "enumerated", {}


def attenuation_of_density(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    phrase = rng.choice(ts.OF_DENSITY[big.attenuation])
    text = f"A lesion {phrase} is present in {seg(big.segments)}, with a size of {S(0, dims(big))}."
    return text, spec, "enumerated", {}


def attenuation_relative(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    phrase = rng.choice(ts.RELATIVE[big.attenuation])
    text = f"In {seg(big.segments)} there is a {S(0, dims(big))} lesion, {phrase}."
    return text, spec, "enumerated", {}


def attenuation_separate(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"There is a lesion in {seg(big.segments)} with a size of {S(0, dims(big))}. "
        + rng.choice(ts.SEPARATE[big.attenuation])
    )
    return text, spec, "enumerated", {}


# ── order ───────────────────────────────────────────────────────────────────
def order_size_list(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    items = sorted(enumerate(spec.lesions), key=lambda pair: pair[1].dims[0])
    if rng.random() < 0.5:
        rng.shuffle(items)
    listed = [f"{seg(lesion.segments)} ({S(i, dims(lesion))})" for i, lesion in items]
    text = f"{spec.largest.attenuation.capitalize()} lesions are seen in {cs._join(listed)}."
    return text, spec, "enumerated", {}


def order_larger_of_two(rng) -> Generated:
    spec = cs.sample_spec(rng, count=2, shared_attenuation=False)
    big, small = spec.lesions
    text = (
        f"A {small.attenuation} lesion of {S(1, dims(small))} lies in {seg(small.segments)}. "
        f"The larger of the two, {big.attenuation}, lies in {seg(big.segments)} with a size of {S(0, dims(big))}."
    )
    return text, spec, "enumerated", {}


def order_there_is(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4, shared_attenuation=False)
    order = list(enumerate(spec.lesions))
    rng.shuffle(order)
    parts = [f"in {seg(lesion.segments)}, a {S(i, dims(lesion))} {lesion.attenuation} lesion" for i, lesion in order]
    return "There are lesions " + "; ".join(parts) + ".", spec, "enumerated", {}


# ── paraphrase ──────────────────────────────────────────────────────────────
def paraphrase_single(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    noun = rng.choice(ts.NOUNS)
    text = rng.choice(
        (
            f"{seg(big.segments).capitalize()} is the site of a {S(0, dims(big))} {big.attenuation} {noun}.",
            f"A {big.attenuation} {noun} sits in {seg(big.segments)}; size {S(0, dims(big))}.",
        )
    )
    return text, spec, "enumerated", {}


def paraphrase_multiple(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=7)
    big = spec.largest
    noun = rng.choice(("foci", "tumors"))
    lead = rng.choice(ts.LEADS)
    text = (
        f"The liver holds {N(WORDS[len(spec.lesions)])} {big.attenuation} {noun}. The {lead} of these sits in "
        f"{seg(big.segments)}, sized {S(0, dims(big))}; the others are under {fmt(big.dims[0] - cs.MIN_GAP_CM)} cm."
    )
    return text, spec, "number", {}


# ── repeat (new) ────────────────────────────────────────────────────────────
def repeat_single(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    block = f"There is a {big.attenuation} lesion of {S(0, dims(big))} at {seg(big.segments)}."
    tail = f"IMPRESSION: {big.attenuation} lesion, {seg(big.segments)}, {S(0, dims(big))}."
    return block, spec, "enumerated", {"tail": tail}


def repeat_multiple(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    big = spec.largest
    listed = cs._join([f"{seg(lesion.segments)} ({S(i, dims(lesion))})" for i, lesion in enumerate(spec.lesions)])
    block = f"{big.attenuation.capitalize()} lesions are seen in {listed}."
    tail = f"IMPRESSION: {N(str(len(spec.lesions)))} lesions, of which the largest is {S(0, dims(big))}, {seg(big.segments)}."
    return block, spec, "number", {"tail": tail}


# ── non-lesion measurements (new) ───────────────────────────────────────────
def _with(rng, sentence: Callable[[float], str]) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    other = round(rng.uniform(big.dims[0] + 0.5, big.dims[0] + 6.0), 1)
    lesion = f"{seg(big.segments).capitalize()} contains a {big.attenuation} lesion of {S(0, dims(big))}."
    text = f"{sentence(other)} {lesion}" if rng.random() < 0.5 else f"{lesion} {sentence(other)}"
    return text, spec, "enumerated", {}


def non_lesion_length(rng) -> Generated:
    return _with(rng, lambda v: f"Liver length is {fmt(v + 10)} cm.")


def non_lesion_duct(rng) -> Generated:
    return _with(rng, lambda v: f"The common hepatic duct measures {fmt(min(v, 0.9))} cm.")


def non_lesion_node(rng) -> Generated:
    return _with(rng, lambda v: f"Near the liver, a lymph node of {fmt(v)} cm is noted.")


CATEGORIES: dict[str, dict[str, Callable[[random.Random], Generated]]] = {
    "count": {"present": count_present, "in_total": count_in_total, "listed": count_listed,
              "another": count_another, "only_one": count_only_one, "vague": count_vague},
    "size": {"single_mm": size_single_mm, "cm_by_cm": size_cm_by_cm, "greatest_diameter": size_greatest_diameter,
             "three_in_size": size_three_in_size, "reaching": size_reaching},
    "segment": {"roman": segment_roman, "subsegment": segment_subsegment, "couinaud": segment_couinaud,
                "ordinal": segment_ordinal, "range": segment_range, "lobe": segment_lobe},
    "attenuation": {"appearing": attenuation_appearing, "of_density": attenuation_of_density,
                    "relative": attenuation_relative, "separate": attenuation_separate},
    "order": {"size_list": order_size_list, "larger_of_two": order_larger_of_two, "there_is": order_there_is},
    "paraphrase": {"single": paraphrase_single, "multiple": paraphrase_multiple},
    "repeat": {"single": repeat_single, "multiple": repeat_multiple},
    "non_lesion": {"length": non_lesion_length, "duct": non_lesion_duct, "node": non_lesion_node},
}
GENERATORS = [(c, s, f) for c, subtypes in CATEGORIES.items() for s, f in subtypes.items()]


# ── Markers ─────────────────────────────────────────────────────────────────
_MARK = re.compile(f"{OPEN}([SN]\\d*)\\{SEP}(.*?){CLOSE}", re.S)


def unmark(text: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Remove markers; return the text and (tag, start, end) for each marked phrase."""
    out, tags, cursor, length = [], [], 0, 0
    for match in _MARK.finditer(text):
        piece = text[cursor : match.start()]
        out.append(piece)
        length += len(piece)
        tags.append((match.group(1), length, length + len(match.group(2))))
        out.append(match.group(2))
        length += len(match.group(2))
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), tags


def labels_from_marks(text: str, tags: list[tuple[str, int, int]], spec: Spec, source: str,
                      segments_written: bool) -> dict[str, Any]:
    seen: set[int] = set()
    lesions = []
    for tag, start, end in sorted(tags, key=lambda t: t[1]):
        if not tag.startswith("S"):
            continue
        i = int(tag[1:])
        lesion = spec.lesions[i]
        lesions.append({
            "span": [start, end],
            "repeat": i in seen,
            "largest": i == 0,
            "known": True,
            "segments": list(lesion.segments) if segments_written else None,
            "attenuation": lesion.attenuation,
        })
        seen.add(i)
    count_spans = [[s, e] for tag, s, e in tags if tag == "N"]
    if source == "number" and (not count_spans or parse_count(text[slice(*count_spans[0])]) != len(spec.lesions)):
        source, count_spans = "unstated", []  # e.g. OCR changed the number word
    return {"count_source": source, "count_spans": count_spans, "lesions": lesions}


def generated_item(base: dict[str, Any], category: str, subtype: str, fn, rng, ocr: bool) -> dict[str, Any]:
    block, spec, source, extra = fn(rng)
    block = cs.fix_articles(block)
    marked = cs.insert_after_liver(base["liver_text"], block)
    if extra.get("tail"):
        marked = f"{marked} {cs.fix_articles(extra['tail'])}"
    if ocr:
        marked = cs.ocr(marked, rng, OCR_RATE)
    text, tags = unmark(marked)
    labels = labels_from_marks(text, tags, spec, source, extra.get("segments_written", True))
    inserted = unmark(block)[0] + (f" {unmark(extra['tail'])[0]}" if extra.get("tail") else "")
    return {
        "id": f"aug:{'ocr:' if ocr else ''}{category}:{subtype}:{base['id']}:{rng.randrange(10**6):06d}",
        "source": "generated",
        "base_id": base["id"],
        "category": "ocr" if ocr else category,
        "subtype": f"{category}/{subtype}" if ocr else subtype,
        "text": text,
        "inserted_text": inserted,
        "labels": labels,
        "targets": decode_labels(text, labels),
    }


def real_item(row: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
    text = row["liver_text"]
    labels = real_labels(text, row["count"], row["location"], row["diameter"], row["attenuation"])
    targets = targets_from_source(text, row["count"], row["location"], row["diameter"], row["attenuation"])
    targets.pop("diameter_spans")
    item = {"id": row["id"], "source": "real", "base_id": row["id"], "category": "real", "subtype": "real"}
    if rng is not None:
        marked, k = [], 0
        spans = [("S", *lesion["span"]) for lesion in labels["lesions"]] + [("N", *s) for s in labels["count_spans"]]
        for tag, s, e in sorted(spans, key=lambda t: t[1]):
            marked += [text[k:s], f"{OPEN}{tag}{SEP}{text[s:e]}{CLOSE}"]
            k = e
        marked.append(text[k:])
        noisy, tags = unmark(cs.ocr("".join(marked), rng, OCR_RATE))
        size_spans = [[s, e] for tag, s, e in tags if tag == "S"]
        labels = {**labels, "lesions": [dict(lesion, span=span) for lesion, span in zip(labels["lesions"], size_spans)],
                  "count_spans": [[s, e] for tag, s, e in tags if tag == "N"]}
        if labels["count_source"] == "number" and (
            not labels["count_spans"] or parse_count(noisy[slice(*labels["count_spans"][0])]) is None
        ):
            labels = {**labels, "count_source": "unstated", "count_spans": []}
        text = noisy
        item.update({"id": f"ocr:{row['id']}", "category": "ocr", "subtype": "real"})
    return {**item, "text": text, "labels": labels, "targets": targets}


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


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, int]] = {}
    for item in items:
        key = f"{item['category']}/{item['subtype']}"
        entry = by.setdefault(key, {"n": 0, "lesion_spans": 0, "repeats": 0, **{s: 0 for s in ("none", "number", "enumerated", "unstated")}})
        entry["n"] += 1
        entry[item["labels"]["count_source"]] += 1
        entry["lesion_spans"] += len(item["labels"]["lesions"])
        entry["repeats"] += sum(lesion["repeat"] for lesion in item["labels"]["lesions"])
    return {"items": len(items), "by_subtype": by}


def render(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Per-lesion training data ({VERSION})",
        "",
        "Every Train study (liver-only text) plus generated items, with every lesion size phrase, the written count, "
        "repeats and the largest lesion labelled (`liversira_ai.nlp.lesion_targets` "
        f"v{LESION_TARGETS_VERSION}). Validation: real only. Test was not read. Text is in `{TRAIN_PATH.as_posix()}` "
        f"and `{VAL_PATH.as_posix()}` (gitignored).",
        "",
        "## Checks",
        "",
        f"- Generated items sharing a Challenge-specific 4-gram with {', '.join(f'`{v}`' for v in CHALLENGE_VERSIONS)}: "
        f"**{manifest['overlap_items']}** ({len(manifest['challenge_specific_ngrams'])} such 4-grams).",
        f"- Validation: per-lesion labels decode to targets that differ from the source-derived targets in "
        f"{manifest['val_label_mismatches']} field values; Train: {manifest['train_label_mismatches']} "
        "(the largest lesion's size is not written while another field is).",
        "",
        "## Train",
        "",
        f"{manifest['train']['items']} items: {manifest['real_train']} real, {manifest['real_ocr']} real with OCR noise, "
        f"{manifest['generated']} generated (seed {SEED}).",
        "",
        "| Category / subtype | Items | Lesion spans | Repeats | none | number | enumerated | unstated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, e in manifest["train"]["by_subtype"].items():
        lines.append(f"| {key} | {e['n']} | {e['lesion_spans']} | {e['repeats']} | {e['none']} | {e['number']} "
                     f"| {e['enumerated']} | {e['unstated']} |")
    v = manifest["val"]["by_subtype"]["real/real"]
    lines += [
        "",
        f"## Validation\n\n{v['n']} studies: none {v['none']}, number {v['number']}, enumerated {v['enumerated']}, "
        f"unstated {v['unstated']}; {v['lesion_spans']} lesion spans, {v['repeats']} repeats.",
        "",
        "## Notes",
        "",
        "- Real reports: sizes inside a range (\"ranging from 0.6 to 6.3 cm\") are not lesion sizes; the largest lesion is "
        "the source label's, which is chosen by volume and is not always the longest written dimension.",
        "- Generated wording is that of `extraction_train_v1`, plus the `repeat` and `non_lesion` categories.",
        "",
    ]
    return "\n".join(lines)


def _mismatches(items: list[dict[str, Any]]) -> int:
    n = 0
    for item in items:
        decoded = decode_labels(item["text"], item["labels"])
        for field, target in item["targets"].items():
            got = decoded[field]
            if field == "diameter_cm" and target is not None and got is not None:
                n += abs(got - target) > 0.05
            else:
                n += got != target
    return n


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file
    from liversira_ai.nlp import transformer_data as data
    from liversira_ai.nlp.liver_text import LIVER_TEXT_VERSION

    split = data.load_split_ids(["train", "val"])
    train_rows = ts.load_rows(set(split["train"]))
    val_rows = ts.load_rows(set(split["val"]))
    bases = sorted((r for r in train_rows if r["count"] == 0 and cs.is_eligible_base(r["full_text"])), key=lambda r: r["id"])

    real_train = [real_item(r) for r in train_rows]
    positives = [r for r in train_rows if r["count"] > 0]
    real_ocr = [real_item(r, random.Random(f"{SEED}:real-ocr:{r['id']}"))
                for r in random.Random(f"{SEED}:real-ocr").sample(positives, REAL_OCR_ITEMS)]
    generated = build_generated(bases)
    train_items = real_train + real_ocr + generated
    val_items = [real_item(r) for r in val_rows]

    blocks = []
    for version in CHALLENGE_VERSIONS:
        with cs.challenge_path(version).open(encoding="utf-8") as handle:
            blocks += [json.loads(line)["inserted_text"] for line in handle if line.strip()]
    specific = ts.challenge_specific(blocks, [r["full_text"] for r in train_rows])
    hits = ts.overlap(generated, specific)
    if hits:
        shown = sorted(hits.items(), key=lambda kv: -kv[1])[:20]
        raise SystemExit("Generated items share Challenge-specific n-grams:\n" + "\n".join(f"{' '.join(g)}: {n}" for g, n in shown))

    ts.write_jsonl(TRAIN_PATH, train_items)
    ts.write_jsonl(VAL_PATH, val_items)
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "lesion_targets_version": LESION_TARGETS_VERSION,
        "liver_text_version": LIVER_TEXT_VERSION,
        "test_policy": "Built from Train and Validation only; Test reports were not read.",
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "real_train": len(real_train),
        "real_ocr": len(real_ocr),
        "generated": len(generated),
        "challenge_specific_ngrams": sorted(" ".join(g) for g in specific),
        "overlap_items": 0,
        "val_label_mismatches": _mismatches(val_items),
        "train_label_mismatches": _mismatches(real_train),
        "train": summarize(train_items),
        "val": summarize(val_items),
        "outputs_sha256": {p.as_posix(): sha256_file(p) for p in (TRAIN_PATH, VAL_PATH)},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p)
            for p in (cs.LIVER_ONLY_CSV, data.SPLITS_PATH, *(cs.challenge_path(v) for v in CHALLENGE_VERSIONS))
        },
        "git_commit": git_commit(),
    }
    (OUT_DIR / f"{VERSION}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = render(manifest)
    (OUT_DIR / f"{VERSION}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
