"""Build the extraction Challenge Set v1 from Validation reports only.

Each item is a real lesion-free Validation report with a generated liver-lesion
description inserted after its first liver sentence. The lesions are sampled
first and the text is written from them, so the expected count, largest-lesion
segments, maximum diameter and attenuation are known by construction: they do
not come from the source labels or from any parser.

Categories (each varies one aspect; the rest uses the dataset's own phrasing):

- control: the dataset's phrasing. Checks the generator itself.
- count_format: digits, "a total of", enumerated lesions without a total,
  "solitary", and unquantified "multiple" (expected count: unstated).
- size_format: mm, "by", a single dimension, three dimensions, "centimeters",
  "up to".
- largest_order: the largest lesion is not described first, with or without
  the word "largest"; smaller lesions may differ in attenuation.
- distractor: kidney, pancreas or lymph-node measurements and counts next to,
  or in the same sentence as, the liver lesion.
- segment_format: Roman numerals, 4a/4b, "Couinaud", ordinal words, "5-6", and
  lobe only (expected segments: unstated).
- attenuation_format: "-dense", "low/high-attenuation", comparisons with the
  liver parenchyma, and attenuation in a separate sentence.
- ocr_noise: the control description with letter-level OCR errors (numbers kept).
- paraphrase: other sentence structures and lesion nouns.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

VERSION = "extraction_challenge_v1"
SEED = 42
PER_CATEGORY = 200
CHALLENGE_PATH = Path(f"data/processed/nlp/challenge/{VERSION}.jsonl")
OUT_DIR = Path("results/nlp/transformers/challenge")

ATTENUATIONS = ("hypoattenuating", "hyperattenuating", "isoattenuating")
ATTENUATION_WEIGHTS = (0.5, 0.3, 0.2)
COUNT_WEIGHTS = {1: 0.30, 2: 0.20, 3: 0.15, 4: 0.12, 5: 0.10, 6: 0.07, 7: 0.06}
WORDS = "zero one two three four five six seven eight nine".split()
ORDINALS = "zeroth first second third fourth fifth sixth seventh eighth".split()
ROMAN = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII")
MIN_GAP_CM = 0.4  # every smaller lesion is at least this much smaller than the largest

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
LIVER = re.compile(r"\b(liver|hepatic|hepato\w*)\b", re.IGNORECASE)
LESION = re.compile(
    r"\b(lesions?|mass(es)?|tumou?rs?|nodules?|cysts?|neoplasms?|metasta\w*|focal|"
    r"hypoattenuat\w*|hyperattenuat\w*|hypodens\w*|hyperdens\w*|abnormalit\w*)\b",
    re.IGNORECASE,
)
# A lesion-free base must not call the liver normal in appearance: that would
# contradict the inserted lesion. "Normal in size" is kept, as in the dataset.
NORMAL_APPEARANCE = re.compile(
    r"\bunremarkable\b|within normal limits|normal (?:in )?(?:size and )?appearance|"
    r"\bnormal liver\b(?! size| volume)|appears? normal|\bnormal\b[^.]*\bliver\b(?! (?:size|volume))",
    re.IGNORECASE,
)
CONFUSABLE = {"l": "1", "o": "0", "e": "c", "i": "l", "m": "rn", "s": "5"}


@dataclass(frozen=True)
class Lesion:
    segments: tuple[int, ...]
    dims: tuple[float, ...]  # cm, longest first
    attenuation: str


@dataclass(frozen=True)
class Spec:
    lesions: tuple[Lesion, ...]  # largest first

    @property
    def largest(self) -> Lesion:
        return self.lesions[0]

    @property
    def smaller(self) -> tuple[Lesion, ...]:
        return self.lesions[1:]


# ── Sampling ────────────────────────────────────────────────────────────────
def _segments(rng: random.Random, pair_rate: float = 0.3) -> tuple[int, ...]:
    if rng.random() < pair_rate:
        return tuple(sorted(rng.sample(range(1, 9), 2)))
    return (rng.randint(1, 8),)


def _dims(rng: random.Random, low: float, high: float) -> tuple[float, float]:
    first = round(rng.uniform(low, high), 1)
    second = round(rng.uniform(0.5, 0.9) * first, 1)
    return first, max(0.1, min(second, first))


def sample_spec(
    rng: random.Random,
    count: int | None = None,
    min_count: int = 1,
    max_count: int = 7,
    shared_attenuation: bool = True,
) -> Spec:
    if count is None:
        options = [n for n in COUNT_WEIGHTS if min_count <= n <= max_count]
        count = rng.choices(options, weights=[COUNT_WEIGHTS[n] for n in options])[0]
    attenuation = rng.choices(ATTENUATIONS, weights=ATTENUATION_WEIGHTS)[0]
    largest = Lesion(_segments(rng), _dims(rng, 1.5, 9.0), attenuation)
    smaller = []
    for _ in range(count - 1):
        other = attenuation if shared_attenuation else rng.choice(ATTENUATIONS)
        top = max(0.4, largest.dims[0] - MIN_GAP_CM)
        smaller.append(Lesion(_segments(rng, 0.0), _dims(rng, 0.3, top), other))
    smaller.sort(key=lambda lesion: -lesion.dims[0])
    return Spec((largest, *smaller))


def expected_fields(spec: Spec, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = {
        "lesion_count": len(spec.lesions),
        "location_segments": list(spec.largest.segments),
        "derived_max_diameter_cm": spec.largest.dims[0],
        "attenuation": spec.largest.attenuation,
    }
    expected.update(overrides or {})
    return expected


# ── Canonical phrasing (as in the dataset) ─────────────────────────────────
def fmt(value: float) -> str:
    return f"{value:.1f}"


def dims_text(dims: tuple[float, ...], unit: str = "cm") -> str:
    return " x ".join(fmt(d) for d in dims) + f" {unit}"


def seg_text(segments: tuple[int, ...]) -> str:
    if len(segments) == 1:
        return f"hepatic segment {segments[0]}"
    return f"hepatic segments {segments[0]} and {segments[1]}"


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def others_text(smaller: tuple[Lesion, ...]) -> str:
    if not smaller:
        return ""
    if len(smaller) == 1:
        s = smaller[0]
        return f" The other lesion is located in {seg_text(s.segments)}, measuring {dims_text(s.dims)}."
    segments = _join([str(s.segments[0]) for s in smaller])
    sizes = _join([dims_text(s.dims) for s in smaller])
    return f" The other lesions are located in hepatic segments {segments}, measuring {sizes}, respectively."


def canonical(spec: Spec, rng: random.Random) -> str:
    big = spec.largest
    if len(spec.lesions) == 1:
        return (
            f"However, a {big.attenuation} liver lesion is identified in {seg_text(big.segments)}, "
            f"measuring {dims_text(big.dims)}."
        )
    return (
        f"The liver contains {WORDS[len(spec.lesions)]} {big.attenuation} lesions. "
        f"The largest lesion, located in {seg_text(big.segments)}, measures {dims_text(big.dims)}."
        + others_text(spec.smaller)
    )


# ── Variants ────────────────────────────────────────────────────────────────
Variant = Callable[[random.Random], tuple[str, dict[str, Any], list[str]]]


def control(rng: random.Random):
    spec = sample_spec(rng)
    return canonical(spec, rng), expected_fields(spec), []


def _multi_body(spec: Spec, opening: str) -> str:
    big = spec.largest
    return (
        f"{opening} The largest lesion, located in {seg_text(big.segments)}, measures {dims_text(big.dims)}."
        + others_text(spec.smaller)
    )


def count_digits(rng):
    spec = sample_spec(rng, min_count=2)
    opening = f"The liver contains {len(spec.lesions)} {spec.largest.attenuation} lesions."
    return _multi_body(spec, opening), expected_fields(spec), ["lesion_count"]


def count_total_of(rng):
    spec = sample_spec(rng, min_count=2)
    number = rng.choice((WORDS[len(spec.lesions)], str(len(spec.lesions))))
    opening = f"A total of {number} {spec.largest.attenuation} liver lesions are identified."
    return _multi_body(spec, opening), expected_fields(spec), ["lesion_count"]


def count_enumerated(rng):
    spec = sample_spec(rng, min_count=2, max_count=5)
    parts = []
    for i, lesion in enumerate(spec.lesions):
        lead = "A" if i == 0 else f"A {ORDINALS[i + 1]}"
        noun = "liver lesion" if i == 0 else "lesion"
        parts.append(
            f"{lead} {lesion.attenuation} {noun} is seen in {seg_text(lesion.segments)}, "
            f"measuring {dims_text(lesion.dims)}."
        )
    return " ".join(parts), expected_fields(spec), ["lesion_count"]


def count_solitary(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    word = rng.choice(("solitary", "single"))
    text = (
        f"A {word} {big.attenuation} liver lesion is noted in {seg_text(big.segments)}, "
        f"measuring {dims_text(big.dims)}."
    )
    return text, expected_fields(spec), ["lesion_count"]


def count_unquantified(rng):
    spec = sample_spec(rng, min_count=3)
    word = rng.choice(("Multiple", "Several", "Numerous"))
    big = spec.largest
    text = (
        f"{word} {big.attenuation} liver lesions are identified. The largest lesion, located in "
        f"{seg_text(big.segments)}, measures {dims_text(big.dims)}."
    )
    return text, expected_fields(spec, {"lesion_count": None}), ["lesion_count"]


def _single(spec: Spec, size: str) -> str:
    big = spec.largest
    return f"However, a {big.attenuation} liver lesion is identified in {seg_text(big.segments)}, measuring {size}."


def size_mm(rng):
    spec = sample_spec(rng, count=1)
    size = " x ".join(str(round(d * 10)) for d in spec.largest.dims) + " mm"
    return _single(spec, size), expected_fields(spec), ["derived_max_diameter_cm"]


def size_by(rng):
    spec = sample_spec(rng, count=1)
    size = " by ".join(fmt(d) for d in spec.largest.dims) + " cm"
    return _single(spec, size), expected_fields(spec), ["derived_max_diameter_cm"]


def size_single_dimension(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    text = f"However, a {fmt(big.dims[0])} cm {big.attenuation} liver lesion is identified in {seg_text(big.segments)}."
    return text, expected_fields(spec), ["derived_max_diameter_cm"]


def size_three_dimensions(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    third = round(rng.uniform(0.4, 0.9) * big.dims[1], 1)
    lesion = Lesion(big.segments, (*big.dims, max(0.1, third)), big.attenuation)
    spec = Spec((lesion,))
    return _single(spec, dims_text(lesion.dims)), expected_fields(spec), ["derived_max_diameter_cm"]


def size_centimeters(rng):
    spec = sample_spec(rng, count=1)
    size = " x ".join(fmt(d) for d in spec.largest.dims) + " centimeters"
    return _single(spec, size), expected_fields(spec), ["derived_max_diameter_cm"]


def size_up_to(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"However, a {big.attenuation} liver lesion is identified in {seg_text(big.segments)}, "
        f"measuring up to {fmt(big.dims[0])} cm."
    )
    return text, expected_fields(spec), ["derived_max_diameter_cm"]


LARGEST_TARGETS = ["location_segments", "derived_max_diameter_cm", "attenuation"]


def largest_last_keyword(rng):
    spec = sample_spec(rng, min_count=2, max_count=4)
    big = spec.largest
    smaller = spec.smaller
    lead = (
        f"The liver contains {WORDS[len(spec.lesions)]} lesions. "
        + (
            f"A smaller {smaller[0].attenuation} lesion is located in {seg_text(smaller[0].segments)}, "
            f"measuring {dims_text(smaller[0].dims)}."
            if len(smaller) == 1
            else f"Smaller lesions are located in hepatic segments {_join([str(s.segments[0]) for s in smaller])}, "
            f"measuring {_join([dims_text(s.dims) for s in smaller])}, respectively."
        )
    )
    text = (
        f"{lead} The largest lesion is {big.attenuation}, located in {seg_text(big.segments)}, "
        f"and measures {dims_text(big.dims)}."
    )
    return text, expected_fields(spec), LARGEST_TARGETS


def _implicit(spec: Spec, order: list[Lesion]) -> str:
    parts = [f"The liver contains {WORDS[len(spec.lesions)]} lesions."]
    for lesion in order:
        parts.append(
            f"A {lesion.attenuation} lesion in {seg_text(lesion.segments)} measures {dims_text(lesion.dims)}."
        )
    return " ".join(parts)


def largest_last_implicit(rng):
    spec = sample_spec(rng, min_count=2, max_count=4, shared_attenuation=False)
    order = list(reversed(spec.lesions))
    return _implicit(spec, order), expected_fields(spec), LARGEST_TARGETS


def largest_middle_implicit(rng):
    spec = sample_spec(rng, min_count=3, max_count=4, shared_attenuation=False)
    order = [spec.smaller[-1], spec.largest, *spec.smaller[:-1]]
    return _implicit(spec, order), expected_fields(spec), LARGEST_TARGETS


def _bigger_than(rng: random.Random, value: float) -> float:
    return round(rng.uniform(value + 0.5, value + 4.0), 1)


def distractor_same_sentence(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    organ = rng.choice(("right kidney", "left kidney", "pancreatic tail", "pancreatic head"))
    other = _bigger_than(rng, big.dims[0])
    text = (
        f"A {fmt(other)} cm cyst is seen in the {organ}, and a {big.attenuation} liver lesion "
        f"measuring {dims_text(big.dims)} is identified in {seg_text(big.segments)}."
    )
    return text, expected_fields(spec), ["lesion_count", "derived_max_diameter_cm", "location_segments"]


def distractor_preceding_sentence(rng):
    spec = sample_spec(rng)
    other = _bigger_than(rng, spec.largest.dims[0])
    n_other = rng.randint(2, 4)
    organ = rng.choice(("right kidney", "left kidney"))
    text = (
        f"The {organ} contains {WORDS[n_other]} cysts, the largest measuring {fmt(other)} cm. "
        + canonical(spec, rng)
    )
    return text, expected_fields(spec), ["lesion_count", "derived_max_diameter_cm"]


def distractor_lymph_node(rng):
    spec = sample_spec(rng)
    node = _bigger_than(rng, spec.largest.dims[0])
    text = (
        canonical(spec, rng)
        + f" An enlarged lymph node near the liver hilum measures {fmt(node)} cm in short axis."
    )
    return text, expected_fields(spec), ["lesion_count", "derived_max_diameter_cm"]


def _seg_variant(rng, render: Callable[[tuple[int, ...]], str], spec: Spec | None = None, overrides=None):
    spec = spec or sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"However, a {big.attenuation} liver lesion is identified in {render(big.segments)}, "
        f"measuring {dims_text(big.dims)}."
    )
    return text, expected_fields(spec, overrides), ["location_segments"]


def segment_roman(rng):
    def render(segments):
        if len(segments) == 1:
            return f"hepatic segment {ROMAN[segments[0]]}"
        return f"hepatic segments {ROMAN[segments[0]]} and {ROMAN[segments[1]]}"

    return _seg_variant(rng, render)


def segment_subsegment(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    spec = Spec((Lesion((4,), big.dims, big.attenuation),))
    token = rng.choice(("4a", "4b", "IVa", "IVb"))
    return _seg_variant(rng, lambda _: f"hepatic segment {token}", spec)


def segment_couinaud(rng):
    def render(segments):
        if len(segments) == 1:
            return f"Couinaud segment {segments[0]}"
        return f"Couinaud segments {segments[0]} and {segments[1]}"

    return _seg_variant(rng, render)


def segment_ordinal(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    spec = Spec((Lesion(big.segments[:1], big.dims, big.attenuation),))
    return _seg_variant(rng, lambda s: f"the {ORDINALS[s[0]]} hepatic segment", spec)


def segment_hyphen(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    first = rng.randint(1, 7)
    spec = Spec((Lesion((first, first + 1), big.dims, big.attenuation),))
    return _seg_variant(rng, lambda s: f"hepatic segments {s[0]}-{s[1]}", spec)


def segment_lobe_only(rng):
    lobe = rng.choice(("right hepatic lobe", "left hepatic lobe", "right lobe of the liver"))
    return _seg_variant(rng, lambda _: f"the {lobe}", overrides={"location_segments": None})


DENSE = {"hypoattenuating": "hypodense", "hyperattenuating": "hyperdense", "isoattenuating": "isodense"}
LOW_HIGH = {
    "hypoattenuating": ("low-attenuation", "low-density"),
    "hyperattenuating": ("high-attenuation", "high-density"),
    "isoattenuating": ("iso-attenuating", "isodense"),
}
COMPARATIVE = {
    "hypoattenuating": "lower attenuation than the surrounding liver parenchyma",
    "hyperattenuating": "higher attenuation than the surrounding liver parenchyma",
    "isoattenuating": "attenuation similar to the surrounding liver parenchyma",
}


def _att_word(rng, render: Callable[[str], str]):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"However, a {render(big.attenuation)} liver lesion is identified in {seg_text(big.segments)}, "
        f"measuring {dims_text(big.dims)}."
    )
    return text, expected_fields(spec), ["attenuation"]


def attenuation_dense(rng):
    return _att_word(rng, lambda a: DENSE[a])


def attenuation_low_high(rng):
    return _att_word(rng, lambda a: rng.choice(LOW_HIGH[a]))


def attenuation_comparative(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"However, a liver lesion of {COMPARATIVE[big.attenuation]} is identified in "
        f"{seg_text(big.segments)}, measuring {dims_text(big.dims)}."
    )
    return text, expected_fields(spec), ["attenuation"]


def attenuation_separate_sentence(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    text = (
        f"However, a liver lesion is identified in {seg_text(big.segments)}, measuring "
        f"{dims_text(big.dims)}. The lesion is {big.attenuation}."
    )
    return text, expected_fields(spec), ["attenuation"]


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


def ocr(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Letter-level OCR errors on words of four or more letters. A letter may become a
    look-alike digit inside a word ("lesion" -> "1esion"); stated numbers are never changed."""
    parts = re.split(r"(\W+)", text)
    eligible = [i for i, p in enumerate(parts) if p.isalpha() and len(p) >= 4]
    chosen = [i for i in eligible if rng.random() < rate] or [rng.choice(eligible)]
    for i in chosen:
        parts[i] = _corrupt(parts[i], rng)
    return "".join(parts)


def ocr_noise(rng):
    spec = sample_spec(rng)
    return ocr(canonical(spec, rng), rng), expected_fields(spec), []


NOUNS = ("mass", "nodule", "focal lesion")


def paraphrase_single(rng):
    spec = sample_spec(rng, count=1)
    big = spec.largest
    noun = rng.choice(NOUNS)
    seg = seg_text(big.segments)
    templates = (
        f"{seg[0].upper() + seg[1:]} harbors a {dims_text(big.dims)} {big.attenuation} {noun}.",
        f"There is a {big.attenuation} {noun} within {seg}, measuring {dims_text(big.dims)}.",
        f"Noted is a {dims_text(big.dims)} {big.attenuation} {noun} in {seg} of the liver.",
        f"A {big.attenuation} hepatic {noun} measuring {dims_text(big.dims)} occupies {seg}.",
    )
    return rng.choice(templates), expected_fields(spec), []


def paraphrase_multiple(rng):
    spec = sample_spec(rng, min_count=2)
    big = spec.largest
    n = WORDS[len(spec.lesions)]
    lead = rng.choice(("dominant", "biggest", "index"))
    templates = (
        f"{n.capitalize()} {big.attenuation} hepatic masses are seen. The {lead} one lies in "
        f"{seg_text(big.segments)} and measures {dims_text(big.dims)}.",
        f"There are {n} {big.attenuation} nodules in the liver, the {lead} measuring {dims_text(big.dims)} "
        f"in {seg_text(big.segments)}.",
    )
    return rng.choice(templates) + others_text(spec.smaller), expected_fields(spec), []


# category -> subtypes (cycled evenly across the category's items)
CATEGORIES: dict[str, dict[str, Variant]] = {
    "control": {"canonical": control},
    "count_format": {
        "digits": count_digits,
        "total_of": count_total_of,
        "enumerated": count_enumerated,
        "solitary": count_solitary,
        "unquantified": count_unquantified,
    },
    "size_format": {
        "mm": size_mm,
        "by": size_by,
        "single_dimension": size_single_dimension,
        "three_dimensions": size_three_dimensions,
        "centimeters": size_centimeters,
        "up_to": size_up_to,
    },
    "largest_order": {
        "last_keyword": largest_last_keyword,
        "last_implicit": largest_last_implicit,
        "middle_implicit": largest_middle_implicit,
    },
    "distractor": {
        "same_sentence": distractor_same_sentence,
        "preceding_sentence": distractor_preceding_sentence,
        "lymph_node": distractor_lymph_node,
    },
    "segment_format": {
        "roman": segment_roman,
        "subsegment": segment_subsegment,
        "couinaud": segment_couinaud,
        "ordinal": segment_ordinal,
        "hyphen": segment_hyphen,
        "lobe_only": segment_lobe_only,
    },
    "attenuation_format": {
        "dense": attenuation_dense,
        "low_high": attenuation_low_high,
        "comparative": attenuation_comparative,
        "separate_sentence": attenuation_separate_sentence,
    },
    "ocr_noise": {"canonical": ocr_noise},
    "paraphrase": {"single": paraphrase_single, "multiple": paraphrase_multiple},
}
FIELDS = ("lesion_count", "location_segments", "derived_max_diameter_cm", "attenuation")


# ── Base reports and insertion ─────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_BREAK.split(text.strip()) if s]


def is_eligible_base(text: str) -> bool:
    """A lesion-free report whose liver is described only by size, volume or HU,
    and which never negates lesions (that would contradict the inserted lesion)."""
    sentences = split_sentences(text)
    liver = [s for s in sentences if LIVER.search(s)]
    if not liver:
        return False
    if any(LESION.search(s) for s in liver):
        return False
    if any(NORMAL_APPEARANCE.search(s) for s in liver):
        return False
    return not any(re.search(r"\b(no|without|free of|absence)\b", s, re.I) and LESION.search(s) for s in sentences)


_ARTICLE = re.compile(r"\b([Aa]) (?=[aeiou]|8\b|8\.|11\.|18\.)")


def fix_articles(text: str) -> str:
    """'a isoattenuating' -> 'an isoattenuating', 'a 8.2 cm' -> 'an 8.2 cm'."""
    return _ARTICLE.sub(r"\1n ", text)


def insert_after_liver(text: str, block: str) -> str:
    sentences = split_sentences(text)
    first = next(i for i, s in enumerate(sentences) if LIVER.search(s))
    return " ".join(sentences[: first + 1] + [block] + sentences[first + 1:])


def build(examples: list[dict[str, Any]], sha: Callable[[str], str]) -> list[dict[str, Any]]:
    bases = sorted((e for e in examples if e["label"] is False and is_eligible_base(e["text"])), key=lambda e: e["id"])
    records = []
    for category, subtypes in CATEGORIES.items():
        pool = list(bases)
        random.Random(f"{SEED}:{category}").shuffle(pool)
        if len(pool) < PER_CATEGORY:
            raise ValueError(f"only {len(pool)} eligible base reports for {category}")
        names = list(subtypes)
        for k, base in enumerate(pool[:PER_CATEGORY]):
            subtype = names[k % len(names)]
            rng = random.Random(f"{SEED}:{category}:{subtype}:{base['id']}")
            block, expected, targets = subtypes[subtype](rng)
            block = fix_articles(block)
            text = insert_after_liver(base["text"], block)
            records.append(
                {
                    "challenge_id": f"{category}:{subtype}:{base['id']}",
                    "base_id": base["id"],
                    "category": category,
                    "subtype": subtype,
                    "targets": targets or list(FIELDS),
                    "expected": {"has_lesion": True, **expected},
                    "inserted_text": block,
                    "text": text,
                    "base_text_sha256": sha(base["text"]),
                    "text_sha256": sha(text),
                }
            )
    return records


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file, sha256_text
    from liversira_ai.nlp import transformer_data as data

    ids = data.load_split_ids(["val"])["val"]
    examples = data.load_examples(ids)
    eligible = sum(1 for e in examples if e["label"] is False and is_eligible_base(e["text"]))
    records = build(examples, sha256_text)

    CHALLENGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHALLENGE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {}
    for r in records:
        counts.setdefault(r["category"], {}).setdefault(r["subtype"], 0)
        counts[r["category"]][r["subtype"]] += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "source_split": "val",
        "test_policy": "Built from Validation only; Test reports were not read.",
        "label_policy": "Expected values are set by construction from sampled lesions, "
        "not from source labels or a parser.",
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "eligible_base_reports": eligible,
        "unique_base_reports": len({r["base_id"] for r in records}),
        "counts": counts,
        "records": len(records),
        "output": CHALLENGE_PATH.as_posix(),
        "output_sha256": sha256_file(CHALLENGE_PATH),
        "inputs_sha256": {
            p.as_posix(): sha256_file(p) for p in (data.MASTER_PATH, data.GROUND_TRUTH_PATH, data.SPLITS_PATH)
        },
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
        f"Built from Validation reports only (seed {SEED}); Test was not read. Each item is a lesion-free "
        f"Validation report ({eligible} eligible) with a generated liver-lesion description inserted after "
        "its first liver sentence. Expected values are set by construction, not taken from source labels "
        f"or a parser. The text is in `{CHALLENGE_PATH.as_posix()}` (gitignored); hashes, IDs and expected "
        "values are in the manifest.",
        "",
        "| Category | Subtype | Items | Target fields |",
        "|---|---|---|---|",
    ]
    for category, subtypes in counts.items():
        for subtype, n in subtypes.items():
            targets = next(r["targets"] for r in records if r["category"] == category and r["subtype"] == subtype)
            shown = "all" if len(targets) == len(FIELDS) else ", ".join(targets)
            lines.append(f"| {category} | {subtype} | {n} | {shown} |")
    lines += [
        "",
        f"{len(records)} items over {len({r['base_id'] for r in records})} base reports. "
        "Expected `null` means the text does not state the value (count for `unquantified`, "
        "segments for `lobe_only`), so the correct output is to abstain.",
        "",
        "Category definitions are in the docstring of "
        "`scripts/nlp/transformers/extraction/build_challenge_set.py`.",
        "",
    ]
    (OUT_DIR / f"{VERSION}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
