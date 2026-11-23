"""Parser v1.2: every rule has a Train-aug item that needs it, and v1.1 stays the default."""

import json
import re
from pathlib import Path

import pytest

from liversira_ai.nlp import baseline, parser_v1_1, parser_v1_2
from liversira_ai.nlp.parser_v1_2_rules import RULES, ablate
from scripts.nlp.transformers.extraction.compare import parser_targets
from scripts.nlp.transformers.extraction.train import field_correct

DATA = Path("data/processed/nlp/extraction")


def test_v1_1_is_still_the_default():
    assert parser_v1_1.PARSER_VERSION == "1.1.0" and parser_v1_2.PARSER_VERSION == "1.2.0"
    text = "The liver holds two hyperattenuating foci."
    assert baseline.extract_liver_lesion_features(text)["has_lesion"] is None  # v1.1: "foci" is not a lesion noun
    assert baseline.extract_liver_lesion_features_v1_2(text)["lesion_count"] == 2


def test_rule_tags_match_the_registry():
    source = Path(parser_v1_2.__file__).read_text(encoding="utf-8")
    tagged = set(re.findall(r"\bR(\d+)\b", source)) - {"2"}  # R2 is only mentioned as dropped
    assert {f"R{n}" for n in tagged} == {r.rule_id for r in RULES}


def _items(name: str) -> dict:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built")
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, handle)}


@pytest.mark.parametrize("rule", RULES, ids=[r.rule_id for r in RULES])
def test_each_rule_is_needed_by_its_train_aug_item(rule):
    item = _items(rule.evidence_set)[rule.evidence_id]
    assert item["category"] not in ("real", "ocr") and f"{item['category']}/{item['subtype']}" == rule.family
    target = item["targets"][rule.field]
    full = parser_targets(baseline.extract_liver_lesion_features_v1_2(item["text"]))[rule.field]
    with ablate(rule.rule_id):
        off = parser_targets(baseline.extract_liver_lesion_features_v1_2(item["text"]))[rule.field]
    assert field_correct(rule.field, full, target)
    assert not field_correct(rule.field, off, target)


def test_ablation_restores_the_module():
    before = parser_v1_2._CONTINUATION_RE
    with ablate("R1"):
        assert parser_v1_2._CONTINUATION_RE is not before
    assert parser_v1_2._CONTINUATION_RE is before
