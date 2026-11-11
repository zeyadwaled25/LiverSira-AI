"""Golden tests for parser v1.1.

Each case in ``fixtures/parser_v1_1_golden.json`` is a synthetic report written
for one behaviour (not copied from Validation or Test reports). A case fixes the
expected value of every field it lists and, where given, the exact evidence
texts for that field. Changing parser behaviour means changing this file on
purpose, in review.
"""

import json
from pathlib import Path

import pytest

from liversira_ai.nlp.parser import PARSER_VERSION, parse_liver_report

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "parser_v1_1_golden.json").read_text(encoding="utf-8")
)
FIELD_TARGETS = {
    "has_lesion": "liver_lesions.has_lesion",
    "lesion_count": "liver_lesions.lesion_count",
    "location_segments": "liver_lesions.largest_lesion.location_segments",
    "reported_dimensions_cm": "liver_lesions.largest_lesion.reported_dimensions_cm",
    "attenuation": "liver_lesions.largest_lesion.attenuation",
}


def test_golden_cases_target_current_parser_version():
    assert PARSER_VERSION == "1.1.0"
    assert len({case["id"] for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_golden_values(case):
    parsed = parse_liver_report(case["text"], include_broad_attenuation_fallback=True)
    for field, expected in case["expected"].items():
        actual = getattr(parsed, field)
        assert actual == expected, f"{case['id']}: {field} = {actual!r}, expected {expected!r}"
    if parsed.reported_dimensions_cm:
        assert parsed.derived_max_diameter_cm == max(parsed.reported_dimensions_cm)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_golden_evidence_provenance(case):
    text = case["text"]
    parsed = parse_liver_report(text, include_broad_attenuation_fallback=True)

    # Every span is verbatim text at its offsets.
    for item in parsed.evidence:
        assert text[item.start:item.end] == item.text

    by_target: dict[str, list[str]] = {}
    for item in parsed.evidence:
        by_target.setdefault(item.target_field, []).append(item.text)

    # A value-bearing field has evidence exactly when it has a value.
    values = {
        "lesion_count": parsed.lesion_count if parsed.presence is True else None,
        "location_segments": parsed.location_segments,
        "reported_dimensions_cm": parsed.reported_dimensions_cm,
        "attenuation": parsed.attenuation,
    }
    for field, value in values.items():
        assert bool(by_target.get(FIELD_TARGETS[field])) == (value is not None), (
            f"{case['id']}: evidence for {field} does not match its value {value!r}"
        )
    # Diameter evidence is the same span as the dimensions it was derived from.
    assert by_target.get("liver_lesions.largest_lesion.derived_max_diameter_cm") == by_target.get(
        FIELD_TARGETS["reported_dimensions_cm"]
    )

    for field, expected in case.get("evidence", {}).items():
        assert by_target.get(FIELD_TARGETS[field], []) == expected, f"{case['id']}: {field} evidence"
