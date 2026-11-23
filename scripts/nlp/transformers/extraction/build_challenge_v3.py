"""Build extraction_challenge_v3: a held-out test for lesion order, counting,
repeated mentions and non-lesion measurements.

Written after the first 6.2 model was scored, to test the failures it showed
(largest lesion not described first, enumerated counts, a lymph-node size kept
by the liver-only filter) in new wording. It is frozen before the per-lesion
model's training data is written; that data must not share any of its
Challenge-specific 4-grams.

Each item is the liver-only text of a lesion-free Validation report with a
generated description inserted after its first liver sentence (and, for
``repeat``, a summary appended at the end). Expected values are set by
construction. Test is not read. Items are scored on the liver-only filter's
output, as the full system would see them.

Categories:
- control: the dataset's own phrasing.
- order: the largest lesion is not first: sizes listed in ascending order,
  smallest described first, largest in the middle with its own attenuation.
- count: a stated total in new words ("Altogether, five lesions"), lesions
  listed one per sentence with no number, one lesion ("an isolated ..."), and
  unquantified ("countless", "a cluster of") where the count is unstated.
- repeat: the same lesion(s) restated in a closing summary; the count must not
  double and the size must not change.
- non_lesion: a liver span or a node next to the liver (larger than the lesion), or
  a hepatic vein or bile duct calibre, written in the liver text.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from scripts.nlp.transformers.extraction import build_challenge_set as cs

VERSION = "extraction_challenge_v3"
SEED = 2026
PER_CATEGORY = {"control": 100, "order": 200, "count": 200, "repeat": 200, "non_lesion": 200}
OUT_DIR = cs.OUT_DIR
CHALLENGE_PATH = cs.challenge_path(VERSION)
TRAIN_V1_PATH = Path("data/processed/nlp/extraction/extraction_train_v1.jsonl")

Lesion, Spec = cs.Lesion, cs.Spec
WORDS = cs.WORDS
ALL = list(cs.FIELDS)
LARGEST = ["location_segments", "derived_max_diameter_cm", "attenuation"]

Generated = tuple[str, str, dict[str, Any], list[str]]  # inserted block, appended tail, expected, targets


def dims(lesion: Lesion) -> str:
    return cs.dims_text(lesion.dims)


def where(segments: tuple[int, ...]) -> str:
    if len(segments) == 1:
        return f"segment {segments[0]}"
    return f"segments {segments[0]} and {segments[1]}"


# ── control ─────────────────────────────────────────────────────────────────
def control(rng) -> Generated:
    spec = cs.sample_spec(rng)
    return cs.canonical(spec, rng), "", cs.expected_fields(spec), []


# ── order ───────────────────────────────────────────────────────────────────
def order_ascending(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=3, max_count=5)
    items = sorted(spec.lesions, key=lambda lesion: lesion.dims[0])
    sizes = [f"{cs.fmt(lesion.dims[0])} cm for {where(lesion.segments)}" for lesion in items]
    listed = "; ".join(sizes[:-1]) + "; and " + sizes[-1]
    text = (
        f"{WORDS[len(items)].capitalize()} {spec.largest.attenuation} lesions are noted, "
        f"measuring {listed}."
    )
    return text, "", cs.expected_fields(spec), LARGEST + ["lesion_count"]


def order_smallest_first(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=3, shared_attenuation=False)
    order = sorted(spec.lesions, key=lambda lesion: lesion.dims[0])
    sentences = [
        f"The smallest one, within {where(order[0].segments)}, is {dims(order[0])} and {order[0].attenuation}."
    ]
    for lesion in order[1:-1]:
        sentences.append(f"Next, a {lesion.attenuation} lesion measuring {dims(lesion)} is housed in {where(lesion.segments)}.")
    big = order[-1]
    sentences.append(
        f"Finally, a {big.attenuation} lesion measuring {dims(big)} is housed in {where(big.segments)}."
    )
    return " ".join(sentences), "", cs.expected_fields(spec), LARGEST


def order_middle(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=3, max_count=4, shared_attenuation=False)
    big = spec.largest
    order = [spec.smaller[-1], big, *spec.smaller[:-1]]
    parts = [f"{where(lesion.segments)} holds a {lesion.attenuation} lesion measuring {dims(lesion)}" for lesion in order]
    text = parts[0][0].upper() + parts[0][1:] + ", while " + ", while ".join(parts[1:]) + "."
    return text, "", cs.expected_fields(spec), LARGEST


# ── count ───────────────────────────────────────────────────────────────────
def count_altogether(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=7)
    big = spec.largest
    n = len(spec.lesions)
    lead = rng.choice((f"Altogether, {WORDS[n]} {big.attenuation} lesions", f"The lesion count is {n}, all {big.attenuation} lesions"))
    text = f"{lead} with the largest of {dims(big)} in {where(big.segments)}."
    return text, "", cs.expected_fields(spec), ["lesion_count"]


def count_listed_sentences(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    leads = ["Separately,", "Also seen is", "In addition,"]
    sentences = [
        f"A {spec.largest.attenuation} lesion measuring {dims(spec.largest)} occupies {where(spec.largest.segments)}."
    ]
    for lead, lesion in zip(leads, spec.smaller):
        sentences.append(f"{lead} a {dims(lesion)} lesion occupies {where(lesion.segments)}.")
    return " ".join(sentences), "", cs.expected_fields(spec), ["lesion_count"]


def count_isolated(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    text = rng.choice(
        (
            f"An isolated {big.attenuation} lesion occupies {where(big.segments)}, {dims(big)} across.",
            f"Just one {big.attenuation} lesion is evident within {where(big.segments)}, {dims(big)} across.",
        )
    )
    return text, "", cs.expected_fields(spec), ["lesion_count"]


def count_unquantified(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=3)
    big = spec.largest
    word = rng.choice(("Countless", "A cluster of", "Myriad"))
    text = f"{word} {big.attenuation} lesions dot the liver, the largest {dims(big)} across in {where(big.segments)}."
    return text, "", cs.expected_fields(spec, {"lesion_count": None}), ["lesion_count"]


# ── repeat ──────────────────────────────────────────────────────────────────
def repeat_single(rng) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    block = f"A {big.attenuation} lesion measuring {dims(big)} occupies {where(big.segments)}."
    tail = f"Summary: {where(big.segments)} {big.attenuation} lesion, {dims(big)}, unchanged in description."
    return block, tail, cs.expected_fields(spec), ["lesion_count", "derived_max_diameter_cm"]


def repeat_multiple(rng) -> Generated:
    spec = cs.sample_spec(rng, min_count=2, max_count=4)
    big = spec.largest
    n = len(spec.lesions)
    others = cs._join([f"{dims(lesion)} in {where(lesion.segments)}" for lesion in spec.smaller])
    block = (
        f"{WORDS[n].capitalize()} {big.attenuation} lesions occupy the liver: {dims(big)} in "
        f"{where(big.segments)}, and {others}."
    )
    tail = f"Summary: {WORDS[n]} {big.attenuation} liver lesions, the largest {dims(big)} in {where(big.segments)}."
    return block, tail, cs.expected_fields(spec), ["lesion_count", "derived_max_diameter_cm"]


# ── non-lesion measurements kept by the liver-only filter ───────────────────
def _with_measure(rng, sentence: Callable[[float], str]) -> Generated:
    spec = cs.sample_spec(rng, count=1)
    big = spec.largest
    other = round(rng.uniform(big.dims[0] + 0.5, big.dims[0] + 6.0), 1)
    lesion = f"A {big.attenuation} lesion measuring {dims(big)} occupies {where(big.segments)}."
    text = f"{sentence(other)} {lesion}" if rng.random() < 0.5 else f"{lesion} {sentence(other)}"
    return text, "", cs.expected_fields(spec), ["lesion_count", "derived_max_diameter_cm"]


def non_lesion_liver_span(rng) -> Generated:
    return _with_measure(rng, lambda v: f"The liver spans {cs.fmt(v + 10)} cm craniocaudally.")


def non_lesion_vessel(rng) -> Generated:
    return _with_measure(
        rng,
        lambda v: rng.choice(
            (
                f"The middle hepatic vein is {cs.fmt(min(v, 1.6))} cm in calibre.",
                f"Intrahepatic biliary ducts are dilated up to {cs.fmt(min(v, 1.2))} cm.",
            )
        ),
    )


def non_lesion_node(rng) -> Generated:
    return _with_measure(rng, lambda v: f"A {cs.fmt(v)} cm node lies adjacent to the liver.")


CATEGORIES: dict[str, dict[str, Callable[[random.Random], Generated]]] = {
    "control": {"canonical": control},
    "order": {"ascending": order_ascending, "smallest_first": order_smallest_first, "middle": order_middle},
    "count": {
        "altogether": count_altogether,
        "listed_sentences": count_listed_sentences,
        "isolated": count_isolated,
        "unquantified": count_unquantified,
    },
    "repeat": {"single": repeat_single, "multiple": repeat_multiple},
    "non_lesion": {"liver_span": non_lesion_liver_span, "vessel": non_lesion_vessel, "node": non_lesion_node},
}


def build(examples: list[dict[str, Any]], liver_texts: dict[str, str], sha: Callable[[str], str]) -> list[dict[str, Any]]:
    bases = sorted((e for e in examples if e["label"] is False and cs.is_eligible_base(e["text"])), key=lambda e: e["id"])
    records = []
    for category, subtypes in CATEGORIES.items():
        pool = list(bases)
        random.Random(f"{SEED}:{category}").shuffle(pool)
        names = list(subtypes)
        for k, base in enumerate(pool[: PER_CATEGORY[category]]):
            subtype = names[k % len(names)]
            rng = random.Random(f"{SEED}:{category}:{subtype}:{base['id']}")
            block, tail, expected, targets = subtypes[subtype](rng)
            block, tail = cs.fix_articles(block), cs.fix_articles(tail)
            text = cs.insert_after_liver(liver_texts[base["id"]], block)
            if tail:
                text = f"{text} {tail}"
            records.append(
                {
                    "challenge_id": f"{category}:{subtype}:{base['id']}",
                    "base_id": base["id"],
                    "category": category,
                    "subtype": subtype,
                    "targets": targets or ALL,
                    "expected": {"has_lesion": True, **expected},
                    "inserted_text": block + (f" {tail}" if tail else ""),
                    "text": text,
                    "text_sha256": sha(text),
                }
            )
    return records


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file, sha256_text
    from liversira_ai.nlp import transformer_data as data
    from liversira_ai.nlp.liver_text import LIVER_TEXT_VERSION, extract_liver_text
    from scripts.nlp.transformers.extraction import build_training_set as ts

    examples = data.load_examples(data.load_split_ids(["val"])["val"])
    liver_texts = cs.load_liver_texts(examples)
    records = build(examples, liver_texts, sha256_text)

    # Held-out check: the filter keeps every inserted block, and the v1 training data
    # (seen by the first model) shares no v3-specific 4-gram.
    dropped = [r["challenge_id"] for r in records
               if not all(s.strip() in extract_liver_text(r["text"]).text for s in re.split(r"(?<=\.)\s+", r["inserted_text"]))]
    train_rows = ts.load_rows(set(data.load_split_ids(["train"])["train"]))
    specific = ts.challenge_specific([r["inserted_text"] for r in records], [r["full_text"] for r in train_rows])
    with TRAIN_V1_PATH.open(encoding="utf-8") as handle:
        train_v1 = [json.loads(line) for line in handle if line.strip()]
    overlap = ts.overlap(train_v1, specific)

    CHALLENGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHALLENGE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {}
    for r in records:
        counts.setdefault(r["category"], {}).setdefault(r["subtype"], 0)
        counts[r["category"]][r["subtype"]] += 1
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "base_text": "liver-only",
        "liver_text_version": LIVER_TEXT_VERSION,
        "scoring_input": "liver-only filter output of each item",
        "source_split": "val",
        "test_policy": "Built from Validation only; Test reports were not read.",
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "counts": counts,
        "records": len(records),
        "items_with_text_dropped_by_filter": dropped,
        "challenge_specific_ngrams": sorted(" ".join(g) for g in specific),
        "train_v1_overlap_ngrams": {" ".join(g): n for g, n in overlap.items()},
        "output": CHALLENGE_PATH.as_posix(),
        "output_sha256": sha256_file(CHALLENGE_PATH),
        "inputs_sha256": {p.as_posix(): sha256_file(p) for p in (cs.LIVER_ONLY_CSV, data.SPLITS_PATH, TRAIN_V1_PATH)},
        "git_commit": git_commit(),
        "items": [
            {k: r[k] for k in ("challenge_id", "base_id", "category", "subtype", "targets", "expected", "text_sha256")}
            for r in records
        ],
    }
    (OUT_DIR / f"{VERSION}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Extraction Challenge Set ({VERSION})",
        "",
        "Held-out test for the per-lesion model: lesion order, counting, repeated mentions and non-lesion "
        f"measurements, in wording not used before. Built from Validation only (seed {SEED}); Test was not read. "
        "Each item is the liver-only text of a lesion-free Validation report with a generated description; "
        f"expected values are set by construction. Text: `{CHALLENGE_PATH.as_posix()}` (gitignored).",
        "",
        f"- Items whose inserted text the liver-only filter drops: {len(dropped)}.",
        f"- Challenge-specific masked 4-grams: {len(specific)}; found in `extraction_train_v1` items: "
        f"{len(overlap)}.",
        "",
        "| Category | Subtype | Items | Target fields |",
        "|---|---|---|---|",
    ]
    for category, subtypes in counts.items():
        for subtype, n in subtypes.items():
            targets = next(r["targets"] for r in records if r["category"] == category and r["subtype"] == subtype)
            shown = "all" if len(targets) == len(ALL) else ", ".join(targets)
            lines.append(f"| {category} | {subtype} | {n} | {shown} |")
    lines += [
        "",
        f"{len(records)} items over {len({r['base_id'] for r in records})} base reports. "
        "Category definitions are in the docstring of `scripts/nlp/transformers/extraction/build_challenge_v3.py`.",
        "",
    ]
    (OUT_DIR / f"{VERSION}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
