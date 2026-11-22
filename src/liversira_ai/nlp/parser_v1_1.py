"""Rule-based parsing of liver findings in radiology reports (v1.1).

Changes from v1.0 (frozen in ``parser_v1_0``):

* **Organ scope.** Every clause is assigned the organ it is about. A clause
  that names no organ ("This lesion is hyperattenuating") inherits the organ of
  the previous clause, so kidney and pancreas lesions no longer leak into liver
  fields, and "common hepatic artery" is not a liver mention.
* **Negation.** Explicit negations ("no evidence of tumor in the spleen, liver
  ..."), and normal-appearance statements ("unremarkable", "within normal limits
  in terms of size and appearance") are negative. Size-only statements ("the
  liver is normal in size") are not: they do not rule out a lesion.
* **Uncertainty scope.** Hedging only affects the clause it appears in.
* **Largest lesion.** Each liver lesion mention carries its own size, segments
  and attenuation. The largest lesion is the one the report calls largest, or
  else the one with the largest stated dimension. "measures", "centimeters",
  "mm", sizes before the noun, ranges and "respectively" lists are supported.
* **Count.** Stated totals, "N additional lesions", bullet lists and
  "respectively" lists are counted. "Multiple lesions" without a number is left
  unquantified instead of being set to 1.
* **Evidence provenance.** Every evidence span is the exact text that produced
  the value: the deciding clause for presence, the count phrase, and the size,
  segment and attenuation phrases of the selected largest lesion.
"""

from dataclasses import dataclass, field
import re

PARSER_VERSION = "1.1.0"

NUMBER_WORDS: dict[str, int] = {
    word: index
    for index, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
NUMBER_WORDS.update({"a": 1, "an": 1, "single": 1})
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}

# ── Shared vocabulary ────────────────────────────────────────────────────────
_LESION = r"(?:lesions?|mass(?:es)?(?!\s+effect)|nodules?|tumou?rs?|metastas[ie]s|neoplasms?|cysts?)"
_LESION_RE = re.compile(rf"\b{_LESION}\b", re.I)
_PLURAL_LESION_RE = re.compile(
    r"\b(?:lesions|masses|nodules|tumou?rs|metastases)\b|\b(?:multiple|several|numerous|many)\b",
    re.I,
)
_ORGANS = {
    "liver": re.compile(r"\bliver\b|\bhepatic\b(?!\s+(?:arter|vein|duct|flexure))", re.I),
    "pancreas": re.compile(r"\bpancrea\w*", re.I),
    "kidney": re.compile(r"\bkidneys?\b|\brenal\b", re.I),
    "spleen": re.compile(r"\bspleen\b|\bsplenic\b(?!\s+(?:arter|vein))", re.I),
}
# Comparisons name a reference organ, not the subject ("relative to the liver").
_COMPARISON_RE = re.compile(
    r"\b(?:relative|compared|in comparison)\s+(?:to|with)\s+(?:the\s+)?(?:surrounding\s+|adjacent\s+|normal\s+)?"
    r"(?:liver|hepatic|renal|kidney|pancrea\w*|splenic|spleen)(?:\s+parenchyma)?",
    re.I,
)
_LIVER_ATTACHED_RE = re.compile(
    rf"\b(?:liver|hepatic)\s+(?:[a-z\-]+\s+){{0,2}}?{_LESION}\b"
    rf"|\b{_LESION}\s+(?:[\w\-,()]+\s+){{0,5}}?(?:in|of|within|involving)\s+(?:the\s+)?(?:liver|hepatic)\b"
    r"|\bhepatic\s+segments?\s+[1-8IV]",
    re.I,
)
_NEGATION_RE = re.compile(
    rf"\b(?:no|without|negative\s+for|free\s+of|absence\s+of|none\s+of|zero|total\s+of\s+0"
    rf"|(?:does|do|did)\s+not\s+(?:\w+\s+)?(?:demonstrate|show|reveal|contain|have|appear)"
    rf"|not\s+(?:seen|identified|present|detected|visualized|noted))\b[^;]*?\b{_LESION}\b",
    re.I,
)
_PARTIAL_NEGATION_RE = re.compile(r"\bno\s+(?:other|additional|new|further)\b", re.I)
_UNCERTAINTY_RE = re.compile(
    r"cannot\s+be\s+(?:excluded|ruled\s+out)|cannot\s+(?:exclude|rule\s+out)|can't\s+be\s+excluded"
    r"|\bpossibl[ey]\b|\bprobabl[ey]\b|\bquestionable\b|\bindeterminate\b|\bequivocal\b"
    r"|\bsuspicious\s+for\b|\bsuspected\b|\bsuggestive\s+of\b|\blikely\b"
    r"|\b(?:may|might|could)\s+represent\b",
    re.I,
)
_NORMAL_RE = re.compile(
    r"\bunremarkable\b(?!\s+in\s+(?:terms\s+of\s+)?(?:size|volume)\b(?!\s+and\s+appearance))"
    r"|\bwithin\s+normal\s+limits\b(?!\s+in\s+(?:terms\s+of\s+)?(?:size|volume)\b(?!\s+and\s+(?:appearance|enhancement|attenuation)))"
    r"|\bnormal\s+(?:in\s+)?(?:size\s+and\s+)?appearance\b"
    r"|\bnormal\s+liver\s+size\s+and\s+appearance\b"
    r"|\bnormal\s+liver\b(?!\s+(?:size|volume))"
    r"|\bappears?\s+normal\b(?!\s+in\s+size)",
    re.I,
)
_ANAPHOR_RE = re.compile(
    rf"^(?:this|that|the|these|those|each|all|both)\s+(?:\w+\s+)?{_LESION}\b|^(?:it|they)\b",
    re.I,
)
_LARGEST_RE = re.compile(r"\b(?:largest|larger|dominant|biggest|index)\b", re.I)
_UNIT = r"(?:cm|centimet(?:er|re)s?|mm|millimet(?:er|re)s?)\b"
_NUMBER = r"\d+(?:\.\d+)?"
_DIMENSION_RE = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*(?:x|×|by)\s*({_NUMBER})(?:\s*(?:x|×|by)\s*({_NUMBER}))?\s*({_UNIT})"
    rf"|(?<![\w.])({_NUMBER})\s*({_UNIT})",
    re.I,
)
_RANGE_RE = re.compile(r"\branging\b|\brange\b|\bfrom\b", re.I)
_SEGMENT_TOKEN = r"(?:[1-8][ab]?|VIII|VII|VI|IV|V|III|II|I)(?![A-Za-z0-9])"
_SEGMENT_RE = re.compile(
    rf"(?i:(?:hepatic\s+|couinaud\s+|liver\s+)?segments?)\s+({_SEGMENT_TOKEN}"
    rf"(?:\s*(?:,|/|&|-|(?i:and)|(?i:,\s*and))\s*{_SEGMENT_TOKEN})*)"
)
_SEGMENT_GROUP_SPLIT_RE = re.compile(r"\s*(?:,\s*(?:and\s+)?|\s+and\s+|&)\s*", re.I)
_ATTENUATION_RE = re.compile(
    r"\b(hypo|iso|hyper)-?(?:attenuating|attenuation|dense|density)\b", re.I
)
_NUM = (
    r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)
_TOTAL_RE = re.compile(
    rf"\b(?:total\s+of|totall?ing)\s+({_NUM})\b(?!\s*(?:cc|cm|mm|ml))", re.I
)
_ADDITIONAL_RE = re.compile(
    rf"(?<![\d.])\b({_NUM})\s+(?:additional|other|more|further|smaller|remaining)\s+"
    rf"(?:[a-z\-]+\s+){{0,2}}?(?:liver\s+|hepatic\s+)?(?:lesions?|masses|nodules|tumou?rs)\b"
    rf"|\b(?:remaining|other)\s+({_NUM})\s+(?:[a-z\-]+\s+){{0,2}}?(?:liver\s+|hepatic\s+)?"
    rf"(?:lesions|masses|nodules|tumou?rs)\b",
    re.I,
)
_ADDITIONAL_WORD_RE = re.compile(
    r"\b(?:additional(?:ly)?|another|second|smaller|remaining|other)\b", re.I
)
_IMPRESSION_RE = re.compile(r"\bIMPRESSION\b")
_PLAIN_COUNT_RE = re.compile(
    rf"(?<![\d.])\b({_NUM})\s+"
    rf"(?:(?!additional\b|other\b|more\b|further\b|smaller\b|new\b|x\b|cm\b|mm\b|cc\b)[a-z\-]+\s+){{0,3}}?"
    rf"(?:liver\s+|hepatic\s+)?(?:lesions|masses|nodules|tumou?rs|metastases)\b",
    re.I,
)
_SINGULAR_RE = re.compile(
    rf"\b(?:a|an|one|single)\s+(?:[\w.\-]+\s+){{0,8}}?(?:liver\s+|hepatic\s+)?(?:lesion|mass|nodule|tumou?r)\b(?!s)",
    re.I,
)
_SENTENCE_END_RE = re.compile(r"\.(?:\s+(?=[A-Z])|\s*$)")
_CLAUSE_BOUNDARY_RE = re.compile(
    r";\s+|,\s+(?=(?:but|however|while|whereas)\b)|\s+(?=(?:but|whereas)\s)|\s+\*\s+|:\s*\*\s+"
    r"|\s+(?=\d{1,2}\.\s+[A-Z])"
)


# ── Data structures ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ParsedEvidence:
    """A verbatim text span supporting one parsed target."""

    target_field: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedLiverReport:
    """Text-derived liver values. Field names are shared with v1.0."""

    liver_sentences: list[str]
    liver_text: str
    has_mention: bool
    presence: bool | None
    presence_reason: str | None
    lesion_count: int | None
    location_segments: list[int] | None
    reported_dimensions_cm: list[float] | None
    derived_max_diameter_cm: float | None
    attenuation: str | None
    evidence: list[ParsedEvidence]
    count_reason: str | None = None
    parser_version: str = PARSER_VERSION


@dataclass
class _Clause:
    text: str
    start: int
    end: int
    organs: frozenset[str]
    subject: str | None
    bullet: bool = False


@dataclass
class _Span:
    value: object
    start: int
    end: int
    # Start of the full phrase ("hepatic segment 6") when ``start`` points at a part.
    phrase_start: int | None = None


@dataclass
class _Mention:
    clause: _Clause
    dimensions: _Span | None = None  # value: list[float] in cm
    segments: _Span | None = None  # value: list[int]
    attenuation: _Span | None = None  # value: str
    largest: bool = False
    group_level: bool = False


@dataclass
class _Decision:
    presence: bool | None
    reason: str | None
    evidence: list[_Clause] = field(default_factory=list)


# ── Segmentation ────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """Split text into sentences (without the final period), preserving decimals."""
    return [sentence.rstrip(".") for sentence, _, _ in _sentence_spans(text or "")]


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        _append_span(text, start, match.start() + 1, spans)  # keep the period
        start = match.end()
    _append_span(text, start, len(text), spans)
    return spans


def _append_span(text: str, start: int, end: int, spans: list[tuple[str, int, int]]) -> None:
    chunk = text[start:end]
    stripped = chunk.strip()
    if stripped:
        left = start + len(chunk) - len(chunk.lstrip())
        spans.append((stripped, left, left + len(stripped)))


def get_liver_sentences(text: str) -> list[str]:
    """Return sentences that mention the liver."""
    return [s for s in split_sentences(text) if _ORGANS["liver"].search(_without_comparisons(s))]


def _without_comparisons(text: str) -> str:
    return _COMPARISON_RE.sub(lambda m: " " * len(m.group(0)), text)


def _organs(text: str) -> frozenset[str]:
    cleaned = _without_comparisons(text)
    return frozenset(name for name, pattern in _ORGANS.items() if pattern.search(cleaned))


def _clauses(text: str) -> list[_Clause]:
    clauses: list[_Clause] = []
    subject: str | None = None
    for _, sentence_start, sentence_end in _sentence_spans(text):
        cursor = sentence_start
        pieces: list[tuple[int, int, bool]] = []
        sentence = text[sentence_start:sentence_end]
        bullet_next = False
        for match in _CLAUSE_BOUNDARY_RE.finditer(sentence):
            pieces.append((cursor, sentence_start + match.start(), bullet_next))
            cursor = sentence_start + match.end()
            bullet_next = "*" in match.group(0)
        pieces.append((cursor, sentence_end, bullet_next))
        for start, end, bullet in pieces:
            chunk = text[start:end]
            stripped = chunk.strip()
            if not stripped:
                continue
            left = start + len(chunk) - len(chunk.lstrip())
            organs = _organs(stripped)
            if len(organs) == 1:
                subject = next(iter(organs))
            elif len(organs) > 1:
                subject = "multiple"
            clauses.append(
                _Clause(stripped, left, left + len(stripped), organs, subject, bullet)
            )
    return clauses


# ── Presence ────────────────────────────────────────────────────────────────
def _in_liver_scope(clause: _Clause) -> bool:
    """Is a lesion in this clause about the liver?"""
    if clause.subject == "liver":
        return True
    if clause.subject == "multiple" and "liver" in clause.organs:
        return bool(_LIVER_ATTACHED_RE.search(clause.text))
    return False


def _negates_liver(clause: _Clause) -> bool:
    if _PARTIAL_NEGATION_RE.search(clause.text):
        return False
    liver_in_scope = clause.subject == "liver" or "liver" in clause.organs
    if not liver_in_scope:
        return False
    if _NEGATION_RE.search(clause.text):
        return True
    if not _NORMAL_RE.search(clause.text):
        return False
    if clause.subject == "liver" or re.search(r"\bnormal\s+liver\b", clause.text, re.I):
        return True
    # Organ lists ("The spleen, liver and kidneys are unremarkable") need a plural verb.
    return bool(re.search(r"\b(?:are|were|appear)\b", clause.text, re.I))


def _classify(clauses: list[_Clause]) -> tuple[list[_Clause], list[_Clause], list[_Clause]]:
    affirmative, uncertain, negative = [], [], []
    for clause in clauses:
        has_lesion = bool(_LESION_RE.search(clause.text))
        if _negates_liver(clause):
            negative.append(clause)
            continue
        if not has_lesion or not _in_liver_scope(clause):
            continue
        if _NEGATION_RE.search(clause.text) or _PARTIAL_NEGATION_RE.search(clause.text):
            continue
        if _UNCERTAINTY_RE.search(clause.text):
            uncertain.append(clause)
        else:
            affirmative.append(clause)
    return affirmative, uncertain, negative


def _decide(
    affirmative: list[_Clause], uncertain: list[_Clause], negative: list[_Clause]
) -> _Decision:
    if affirmative and negative:
        return _Decision(None, "conflicting_terms", sorted(negative + affirmative, key=lambda c: c.start))
    if affirmative:
        return _Decision(True, None, affirmative)
    if uncertain:
        reason = "conflicting_terms" if negative else "ambiguous"
        return _Decision(None, reason, sorted(uncertain + negative, key=lambda c: c.start))
    if negative:
        return _Decision(False, None, negative)
    return _Decision(None, "not_mentioned", [])


# ── Lesion mentions ─────────────────────────────────────────────────────────
def _to_cm(value: str, unit: str) -> float:
    number = float(value)
    return round(number / 10.0, 4) if unit.lower().startswith("m") else number


def _dimensions(clause: _Clause) -> list[_Span]:
    spans = []
    for match in _DIMENSION_RE.finditer(clause.text):
        if match.group(4):
            values = [match.group(1), match.group(2), match.group(3)]
            unit = match.group(4)
        else:
            values, unit = [match.group(5)], match.group(6)
        dims = [_to_cm(value, unit) for value in values if value]
        spans.append(_Span(dims, clause.start + match.start(), clause.start + match.end()))
    return spans


def _segment_number(token: str) -> int | None:
    token = token.strip()
    if token.upper() in ROMAN and token.isupper():
        return ROMAN[token.upper()]
    digits = re.match(r"[1-8]", token)
    return int(digits.group(0)) if digits else None


def _segment_groups(clause: _Clause) -> list[_Span]:
    """Segment groups in order: "segments 5/6, 2 and 1" -> [5,6], [2], [1]."""
    groups: list[_Span] = []
    for match in _SEGMENT_RE.finditer(clause.text):
        phrase_start = clause.start + match.start()
        offset = clause.start + match.start(1)
        listing = match.group(1)
        parts = [p for p in _SEGMENT_GROUP_SPLIT_RE.split(listing) if p]
        cursor = 0
        for part in parts:
            numbers = sorted(
                {n for n in (_segment_number(t) for t in re.split(r"[/\-]", part)) if n}
            )
            local = listing.find(part, cursor)
            cursor = local + len(part)
            if not numbers:
                continue
            if len(parts) == 1:
                # A single group: the evidence is the whole phrase.
                groups.append(_Span(numbers, phrase_start, clause.start + match.end()))
            else:
                groups.append(
                    _Span(numbers, offset + local, offset + local + len(part), phrase_start)
                )
    return groups


def _merged_segments(groups: list[_Span]) -> _Span | None:
    if not groups:
        return None
    values = sorted({n for group in groups for n in group.value})
    start = min(g.phrase_start if g.phrase_start is not None else g.start for g in groups)
    return _Span(values, start, groups[-1].end)


def _attenuation(clause: _Clause) -> _Span | None:
    matches = list(_ATTENUATION_RE.finditer(clause.text))
    if not matches or len({m.group(1).lower() for m in matches}) > 1:
        # Mixed descriptors in one clause ("some hypo-, some hyperattenuating")
        # cannot be assigned to a single lesion.
        return None
    match = matches[0]
    return _Span(
        f"{match.group(1).lower()}attenuating",
        clause.start + match.start(),
        clause.start + match.end(),
    )


def _mentions(affirmative: list[_Clause], all_clauses: list[_Clause]) -> list[_Mention]:
    mentions: list[_Mention] = []
    affirmative_ids = {id(c) for c in affirmative}
    previous: list[_Mention] = []
    for clause in all_clauses:
        if id(clause) not in affirmative_ids:
            # An anaphoric clause in another organ's context ends the chain.
            if clause.organs:
                previous = []
            continue
        dims = _dimensions(clause)
        groups = _segment_groups(clause)
        attenuation = _attenuation(clause)
        largest = bool(_LARGEST_RE.search(clause.text))
        anaphoric = not clause.organs and bool(_ANAPHOR_RE.search(clause.text))

        if anaphoric and previous and not dims:
            # "This lesion is hyperattenuating" describes the previous mention.
            for mention in previous:
                if attenuation and mention.attenuation is None:
                    mention.attenuation = attenuation
                if groups and mention.segments is None:
                    mention.segments = _merged_segments(groups)
            continue

        respectively = "respectively" in clause.text.lower()
        ranged = len(dims) == 2 and bool(_RANGE_RE.search(clause.text))
        group_level = bool(_PLURAL_LESION_RE.search(clause.text)) and (not dims or ranged)
        current: list[_Mention] = []
        if dims:
            for index, span in enumerate(dims):
                if respectively and len(groups) == len(dims):
                    segments = groups[index]
                elif len(groups) > 1 and len(dims) > 1 and not ranged:
                    # "one in segment 7, measuring A, and another in segment 5,
                    # measuring B": pair each size with the nearest segment before it.
                    before = [g for g in groups if g.start < span.start]
                    segments = before[-1] if before else groups[0]
                else:
                    segments = _merged_segments(groups)
                current.append(
                    _Mention(clause, span, segments, attenuation, largest, group_level)
                )
        else:
            current.append(
                _Mention(clause, None, _merged_segments(groups), attenuation, largest, group_level)
            )
        mentions.extend(current)
        previous = current
    return mentions


def _select_largest(mentions: list[_Mention]) -> _Mention | None:
    sized = [m for m in mentions if m.dimensions is not None]
    flagged = [m for m in sized if m.largest and not m.group_level]
    if flagged:
        return flagged[0]
    flagged_unsized = [m for m in mentions if m.largest and m.dimensions is None]
    if flagged_unsized and not sized:
        return flagged_unsized[0]
    if sized:
        return max(sized, key=lambda m: (max(m.dimensions.value), -m.dimensions.start))
    return mentions[0] if mentions else None


# ── Count ───────────────────────────────────────────────────────────────────
def _number(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def _count(
    affirmative: list[_Clause], text: str
) -> tuple[int | None, list[_Span], str | None]:
    """Stated lesion count with its evidence spans, in priority order.

    Returns ``(count, spans, reason)``; ``count`` is ``None`` when not stated.

    1. An explicit total ("a total of 25 lesions", "totaling 16").
    2. A plain count ("three liver lesions") combined with "N smaller /
       additional / remaining" lesions: a smaller additional number is read as a
       subset of the plain count, a larger or equal one as extra lesions
       ("two large masses ... 18 smaller lesions" = 20).
    3. A bullet list of lesions.
    4. "N additional lesions" alone: N plus the lesion already described.
    5. A "respectively" list of sizes, plus one if it lists additional lesions.
    6. Singular mentions ("a liver mass", then "additionally, a smaller lesion"),
       counted in the findings before the impression.
    "Multiple lesions" with no number is reported as unquantified.
    """

    def first(pattern: re.Pattern[str]) -> tuple[int, _Span] | None:
        for clause in affirmative:
            match = pattern.search(clause.text)
            if match:
                token = next(group for group in match.groups() if group)
                span = _Span(_number(token), clause.start + match.start(), clause.start + match.end())
                return span.value, span
        return None

    total = first(_TOTAL_RE)
    if total is not None:
        return total[0], [total[1]], None

    plain, additional = first(_PLAIN_COUNT_RE), first(_ADDITIONAL_RE)
    if plain is not None:
        if additional is not None and additional[0] >= plain[0]:
            return plain[0] + additional[0], [plain[1], additional[1]], None
        return plain[0], [plain[1]], None

    bullets = [c for c in affirmative if c.bullet]
    if len(bullets) >= 2:
        return len(bullets), [_Span(None, c.start, c.end) for c in bullets], None

    if additional is not None:
        return additional[0] + 1, [additional[1]], None

    for clause in affirmative:
        if "respectively" in clause.text.lower():
            dims = _dimensions(clause)
            if len(dims) >= 2:
                extra = 1 if _ADDITIONAL_WORD_RE.search(clause.text) else 0
                return len(dims) + extra, [_Span(None, dims[0].start, dims[-1].end)], None

    if any(_PLURAL_LESION_RE.search(c.text) for c in affirmative):
        return None, [], "unquantified"

    # Count singular mentions in the findings; the impression restates them.
    # A report that is only an impression is counted from the impression.
    impression = _IMPRESSION_RE.search(text)
    cutoff = impression.start() if impression else len(text)
    findings = [c for c in affirmative if c.start < cutoff] or affirmative
    singular = [
        _Span(None, clause.start + match.start(), clause.start + match.end())
        for clause in findings
        for match in [_SINGULAR_RE.search(clause.text)]
        if match
    ]
    if singular:
        # Later singular mentions only add a lesion when they say so
        # ("additionally, a smaller lesion"); otherwise they restate it.
        extra = [s for s in singular[1:] if _ADDITIONAL_WORD_RE.search(_clause_text_at(affirmative, s))]
        return 1 + len(extra), [singular[0], *extra], None
    return None, [], "not_mentioned"


def _clause_text_at(clauses: list[_Clause], span: _Span) -> str:
    return next(c.text for c in clauses if c.start <= span.start < c.end)


# ── Public API ──────────────────────────────────────────────────────────────
def _evidence(target: str, span: _Span | _Clause, text: str) -> ParsedEvidence:
    return ParsedEvidence(target, text[span.start:span.end], span.start, span.end)


def parse_liver_report(
    text: str | None,
    *,
    include_broad_attenuation_fallback: bool = False,
) -> ParsedLiverReport:
    """Parse liver findings with organ-scoped, provenance-tracked rules.

    ``include_broad_attenuation_fallback`` allows the largest lesion to take an
    attenuation stated for a group of liver lesions ("multiple hypoattenuating
    liver lesions") when no lesion-specific attenuation is stated.
    """
    normalized = text or ""
    clauses = _clauses(normalized)
    affirmative, uncertain, negative = _classify(clauses)
    decision = _decide(affirmative, uncertain, negative)
    liver_sentences = get_liver_sentences(normalized)

    evidence = [_evidence("liver_lesions.has_lesion", c, normalized) for c in decision.evidence]
    count = segments = dimensions = attenuation = None
    count_reason = None
    if decision.presence is True:
        mentions = _mentions(affirmative, clauses)
        largest = _select_largest(mentions)
        # Count from liver-only clauses first: a multi-organ impression ("a liver
        # mass ... two masses in the right kidney") would otherwise count kidney lesions.
        liver_only = [c for c in affirmative if c.subject == "liver"]
        count, count_spans, count_reason = _count(liver_only or affirmative, normalized)
        for span in count_spans:
            evidence.append(_evidence("liver_lesions.lesion_count", span, normalized))
        if largest is not None:
            # Fallbacks may only borrow from statements that can describe the
            # largest lesion, never from "smaller / additional / other" lesions.
            shared = [
                m for m in mentions
                if m is largest or not _ADDITIONAL_WORD_RE.search(m.clause.text)
            ]
            segment_span = largest.segments
            if segment_span is None:
                distinct = {tuple(m.segments.value) for m in shared if m.segments}
                if len(distinct) == 1:
                    segment_span = next(m.segments for m in shared if m.segments)
            if segment_span is not None:
                segments = list(segment_span.value)
                evidence.append(
                    _evidence("liver_lesions.largest_lesion.location_segments", segment_span, normalized)
                )
            if largest.dimensions is not None:
                dimensions = list(largest.dimensions.value)
                for target in ("reported_dimensions_cm", "derived_max_diameter_cm"):
                    evidence.append(
                        _evidence(f"liver_lesions.largest_lesion.{target}", largest.dimensions, normalized)
                    )
            attenuation_span = largest.attenuation
            if attenuation_span is None and include_broad_attenuation_fallback:
                group = [m.attenuation for m in shared if m.group_level and m.attenuation]
                values = {m.attenuation.value for m in shared if m.attenuation}
                if group:
                    attenuation_span = group[0]
                elif len(values) == 1:
                    attenuation_span = next(m.attenuation for m in shared if m.attenuation)
            if attenuation_span is not None:
                attenuation = str(attenuation_span.value)
                evidence.append(
                    _evidence("liver_lesions.largest_lesion.attenuation", attenuation_span, normalized)
                )
    elif decision.presence is False:
        count = 0

    return ParsedLiverReport(
        liver_sentences=liver_sentences,
        liver_text=" ".join(liver_sentences),
        has_mention=decision.presence is True,
        presence=decision.presence,
        presence_reason=decision.reason,
        lesion_count=count,
        location_segments=segments,
        reported_dimensions_cm=dimensions,
        derived_max_diameter_cm=max(dimensions) if dimensions else None,
        attenuation=attenuation,
        evidence=evidence,
        count_reason=count_reason,
    )
