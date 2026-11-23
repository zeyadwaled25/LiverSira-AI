"""The rules parser v1.2 adds to v1.1, each with the Train-aug item that justifies it.

A rule is justified only by a 6.2 Train-aug item (extraction_train_v1 or
extraction_train_lesion_v1) on which parser v1.2 is right and v1.2 with that one
rule switched off is wrong. ``ablate(rule_id)`` switches a rule off; the evidence
ids in ``RULES`` are checked by ``tests/test_parser_v1_2.py`` and written by
``scripts/nlp/transformers/extraction/parser_v1_2_report.py``. No rule may be
added without such an item. Challenge Sets, Validation and Test are not used.

R2 (more words for the largest lesion) was written and dropped: no Train-aug
item needed it. The ids are not renumbered.
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from liversira_ai.nlp import parser_v1_1 as v11
from liversira_ai.nlp import parser_v1_2 as v12

_NEVER = re.compile(r"(?!x)x")
_ALWAYS = re.compile(r"")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    family: str  # Train-aug family "category/subtype" the evidence comes from
    field: str  # extraction target the evidence item is scored on
    evidence_set: str
    evidence_id: str


RULES: tuple[Rule, ...] = (
    Rule("R1", "An elliptical clause that continues a lesion statement is part of it "
         "(\"The largest, in segment 6, measures ...\", \"; size 8.2 x 6.8 cm\").",
         "count/present", "segments", "extraction_train_v1", "aug:count:present:BDMAP_00005631:059723"),
    Rule("R3", "\"N in total\" states the count.",
         "count/in_total", "count", "extraction_train_v1", "aug:count:in_total:BDMAP_00001937:826725"),
    Rule("R4", "A labelled list (\"Lesion 1: ... Lesion 2: ...\") is counted by its highest label.",
         "count/listed", "count", "extraction_train_v1", "aug:count:listed:BDMAP_00002550:372864"),
    Rule("R5", "A number followed by \".<digit>\" after a segment list is a size, not a segment.",
         "count/listed", "segments", "extraction_train_v1", "aug:count:listed:BDMAP_00002550:372864"),
    Rule("R6", "\"Another lesion\" and \"A further lesion\" each add one lesion.",
         "count/another", "count", "extraction_train_v1", "aug:count:another:BDMAP_00005815:239600"),
    Rule("R7", "Subsegments (\"4A\", \"IVa\") map to their segment.",
         "segment/subsegment", "segments", "extraction_train_v1", "aug:segment:subsegment:BDMAP_00003845:231586"),
    Rule("R8", "Ordinal and word segment names (\"the second segment\", \"segment eight\").",
         "segment/ordinal", "segments", "extraction_train_v1", "aug:segment:ordinal:BDMAP_00003333:473223"),
    Rule("R9", "\"segments 2 to 3\" is a range of segments.",
         "segment/range", "segments", "extraction_train_v1", "aug:segment:range:BDMAP_00003237:113871"),
    Rule("R10", "\"of low/high/equal density\" and \"with low/high/equal attenuation\".",
         "attenuation/of_density", "attenuation", "extraction_train_v1",
         "aug:attenuation:of_density:BDMAP_00004586:460124"),
    Rule("R11", "Comparisons with the liver: \"darker/less dense than\", \"brighter/denser than\", "
         "\"as dense as\", \"indistinguishable in density\".",
         "attenuation/relative", "attenuation", "extraction_train_v1",
         "aug:attenuation:relative:BDMAP_00000881:711770"),
    Rule("R12", "A sentence about the previous lesion's attenuation (\"It appears hyperattenuating.\", "
         "\"Its attenuation is low.\").",
         "attenuation/separate", "attenuation", "extraction_train_v1",
         "aug:attenuation:separate:BDMAP_00001252:554569"),
    Rule("R13", "\"focus\" and \"foci\" are lesion nouns.",
         "attenuation/appearing", "count", "extraction_train_v1",
         "aug:attenuation:appearing:BDMAP_00008962:339767"),
    Rule("R14", "A bare plural with no unquantified word (\"lesions are seen in segment 1 (1.3 x 1.1 cm) and ...\") "
         "is counted from the sizes it lists.",
         "order/size_list", "count", "extraction_train_v1", "aug:order:size_list:BDMAP_00005472:950414"),
    Rule("R15", "\"the larger of the two\" states two lesions.",
         "order/larger_of_two", "count", "extraction_train_v1", "aug:order:larger_of_two:BDMAP_00007427:855788"),
)


def _segment_re(token: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?i:(?:hepatic\s+|couinaud\s+|liver\s+)?segments?)\s+({token}"
        rf"(?:\s*(?:,|/|&|-|(?i:and)|(?i:,\s*and))\s*{token})*)"
    )


def _ablation(rule_id: str) -> dict[str, Any]:
    """Module attributes of parser_v1_2 that switch one rule off."""
    if rule_id == "R1":
        return {"_CONTINUATION_RE": _NEVER}
    if rule_id == "R3":
        return {"_TOTAL_RE": v11._TOTAL_RE}
    if rule_id == "R4":
        return {"_LABELLED_RE": _NEVER}
    if rule_id == "R5":
        return {"_SEGMENT_RE": _segment_re(r"(?:[1-8][abAB]?|(?:VIII|VII|VI|IV|V|III|II|I)[ab]?)(?![A-Za-z0-9])")}
    if rule_id == "R6":
        return {"_SINGULAR_RE": re.compile(v12._SINGULAR_RE.pattern.replace("|another)", ")"), re.I),
                "_ADDITIONAL_WORD_RE": v11._ADDITIONAL_WORD_RE}
    if rule_id == "R7":
        return {"_SEGMENT_RE": _segment_re(r"(?:[1-8][ab]?|VIII|VII|VI|IV|V|III|II|I)(?![A-Za-z0-9]|\.\d)")}
    if rule_id == "R8":
        return {"_ORDINAL_SEGMENT_RE": _NEVER}
    if rule_id == "R9":
        return {"_SEGMENT_SPAN_RE": _NEVER}
    if rule_id in ("R10", "R11"):
        keep = v12._ATTENUATION_WORDS[3:] if rule_id == "R10" else v12._ATTENUATION_WORDS[:3]
        return {"_ATTENUATION_WORDS": keep}
    if rule_id == "R12":
        return {"_ATTENUATION_SENTENCE_RE": _NEVER}
    if rule_id == "R13":
        return {"_LESION_RE": v11._LESION_RE, "_PLURAL_LESION_RE": v11._PLURAL_LESION_RE,
                "_PLAIN_COUNT_RE": v11._PLAIN_COUNT_RE,
                "_SINGULAR_RE": re.compile(v12._SINGULAR_RE.pattern.replace("|focus)", ")"), re.I)}
    if rule_id == "R14":
        return {"_UNQUANTIFIED_RE": _ALWAYS}
    if rule_id == "R15":
        return {"_OF_THE_TWO_RE": _NEVER}
    raise KeyError(rule_id)


@contextmanager
def ablate(rule_id: str) -> Iterator[None]:
    """Run parser v1.2 with one rule switched off (not thread-safe; for checks only)."""
    patch = _ablation(rule_id)
    saved = {name: getattr(v12, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(v12, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(v12, name, value)
