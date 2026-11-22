"""Tests for the extraction Challenge Set generator and its parser scoring."""

import random
import re

import pytest

from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import evaluate_parser as ev

BASE = (
    "The spleen is normal in size, measuring 168.2 cc in volume. "
    "The liver is also normal in size, with a volume of 2860.9 cc and a mean HU value of 89.2 +/- 16.4. "
    "The kidneys are normal in size. IMPRESSION: Normal spleen and kidneys."
)


def all_variants():
    for category, subtypes in cs.CATEGORIES.items():
        for subtype, variant in subtypes.items():
            yield category, subtype, variant


@pytest.mark.parametrize("category,subtype,variant", list(all_variants()))
def test_variants_are_deterministic_and_well_formed(category, subtype, variant):
    for seed in range(25):
        first = variant(random.Random(f"t:{seed}"))
        assert first == variant(random.Random(f"t:{seed}"))
        text, expected, targets = first
        assert set(expected) == set(cs.FIELDS)
        assert set(targets) <= set(cs.FIELDS)
        assert expected["attenuation"] in cs.ATTENUATIONS
        if expected["lesion_count"] is not None:
            assert expected["lesion_count"] >= 1
        if expected["location_segments"] is not None:
            assert expected["location_segments"] == sorted(set(expected["location_segments"]))
            assert all(1 <= s <= 8 for s in expected["location_segments"])


@pytest.mark.parametrize("category,subtype,variant", list(all_variants()))
def test_expected_diameter_is_stated_in_the_text(category, subtype, variant):
    for seed in range(25):
        text, expected, _ = variant(random.Random(f"d:{seed}"))
        diameter = expected["derived_max_diameter_cm"]
        stated = {cs.fmt(diameter), str(round(diameter * 10))}
        numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
        assert stated & numbers, (subtype, diameter, text)


def test_largest_lesion_is_clearly_largest():
    for seed in range(200):
        spec = cs.sample_spec(random.Random(seed))
        for lesion in spec.smaller:
            assert lesion.dims[0] <= spec.largest.dims[0] - cs.MIN_GAP_CM + 1e-9
        for lesion in spec.lesions:
            assert lesion.dims[1] <= lesion.dims[0]


def test_join_and_articles():
    assert cs._join(["5"]) == "5"
    assert cs._join(["5", "8"]) == "5 and 8"
    assert cs._join(["5", "8", "2"]) == "5, 8, and 2"
    assert cs.fix_articles("a isoattenuating lesion and A 8.2 cm cyst") == "an isoattenuating lesion and An 8.2 cm cyst"
    assert cs.fix_articles("a hypoattenuating lesion, a 2.3 cm cyst, a 80 x") == "a hypoattenuating lesion, a 2.3 cm cyst, a 80 x"


def test_ocr_never_changes_numbers():
    # Letters may become look-alike digits ("o" -> "0") inside words; stated numbers stay intact.
    number = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
    text = "The liver contains three lesions measuring 4.8 x 2.9 cm in hepatic segment 7."
    for seed in range(50):
        noisy = cs.ocr(text, random.Random(seed))
        assert number.findall(noisy) == number.findall(text)


def test_base_eligibility():
    assert cs.is_eligible_base(BASE)
    assert not cs.is_eligible_base(BASE.replace("normal in size, with", "unremarkable, with"))
    assert not cs.is_eligible_base(BASE + " No focal lesions are identified in these organs.")
    assert not cs.is_eligible_base(BASE.replace("The liver", "The liver contains a cyst and"))
    assert not cs.is_eligible_base("The spleen is normal. The kidneys are normal.")
    assert not cs.is_eligible_base(BASE + " Normal liver, spleen and kidneys.")


def test_insert_after_first_liver_sentence():
    out = cs.insert_after_liver(BASE, "INSERTED.")
    sentences = cs.split_sentences(out)
    assert sentences[1].startswith("The liver")
    assert sentences[2] == "INSERTED."


def test_build_quota_and_labels(monkeypatch):
    monkeypatch.setattr(cs, "PER_CATEGORY", 12)
    examples = [{"id": f"S{i:03d}", "text": BASE, "label": False} for i in range(20)]
    examples.append({"id": "POS", "text": BASE, "label": True})
    records = cs.build(examples, lambda t: str(len(t)))
    assert len(records) == 12 * len(cs.CATEGORIES)
    assert len({r["challenge_id"] for r in records}) == len(records)
    assert "POS" not in {r["base_id"] for r in records}
    assert all(r["expected"]["has_lesion"] is True for r in records)
    assert all(r["inserted_text"] in r["text"] for r in records)
    counts = [r["subtype"] for r in records if r["category"] == "count_format"]
    assert len(set(counts)) == len(cs.CATEGORIES["count_format"])


def test_build_refuses_small_pool(monkeypatch):
    monkeypatch.setattr(cs, "PER_CATEGORY", 50)
    with pytest.raises(ValueError):
        cs.build([{"id": "A", "text": BASE, "label": False}], str)


def test_outcome_rules():
    assert ev.outcome("lesion_count", None, None) == "correct"
    assert ev.outcome("lesion_count", None, 3) == "wrong"
    assert ev.outcome("lesion_count", 3, None) == "abstained"
    assert ev.outcome("lesion_count", 3, 3) == "correct"
    assert ev.outcome("location_segments", [5, 6], [6, 5]) == "correct"
    assert ev.outcome("location_segments", [5, 6], [5]) == "wrong"
    assert ev.outcome("derived_max_diameter_cm", 4.8, 4.8000001) == "correct"
    assert ev.outcome("derived_max_diameter_cm", 4.8, 4.9) == "wrong"
    assert ev.outcome("attenuation", "hypoattenuating", "hyperattenuating") == "wrong"


def test_gate_flags_only_targeted_failures():
    def row(category, subtype, targets, ok):
        outcomes = {f: "correct" for f in ev.FIELDS}
        if not ok:
            outcomes["attenuation"] = "abstained"
            outcomes["lesion_count"] = "wrong"
        return {"challenge_id": f"{category}:{subtype}:{id(outcomes)}", "category": category, "subtype": subtype,
                "targets": targets, "presence": True, "outcomes": outcomes}

    rows = [row("control", "canonical", list(ev.FIELDS), True) for _ in range(10)]
    rows += [row("attenuation_format", "x", ["attenuation"], False) for _ in range(10)]
    gate = ev.summarize(rows)["gate"]
    assert gate["attenuation"]["needs_model"] is True
    assert gate["attenuation"]["failing_subtypes"] == ["attenuation_format/x"]
    # lesion_count failed too, but no subtype targets it: not flagged.
    assert gate["lesion_count"]["needs_model"] is False
