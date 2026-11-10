"""Frozen rule-based parser v1.0.0 (kept unchanged for reproducibility).

Used by the target-consistency audit, whose frozen outputs (audit CSV, split,
gold-set strata) were produced with this version. New work uses ``parser``.
"""

from dataclasses import dataclass
import re


WORD_TO_NUM: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

@dataclass(frozen=True)
class ParsedEvidence:
    """A verbatim text span supporting one parsed target."""

    target_field: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedLiverReport:
    """Common text-derived values used by the baseline and consistency audit."""

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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving decimal numbers."""
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"\.(?:\s+(?=[A-Z])|\s*$)", text)
        if sentence.strip()
    ]


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    if not text:
        return []
    spans = []
    start = 0
    for match in re.finditer(r"\.(?:\s+(?=[A-Z])|\s*$)", text):
        end = match.start() + 1
        sentence = text[start:end].strip()
        if sentence:
            left = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            spans.append((sentence, left, left + len(sentence)))
        start = match.end()
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            left = start + (len(text[start:]) - len(text[start:].lstrip()))
            spans.append((sentence, left, left + len(sentence)))
    return spans


def get_liver_sentences(text: str) -> list[str]:
    """Return sentences relevant to liver findings or the impression."""
    return [
        sentence
        for sentence in split_sentences(text)
        if any(term in sentence.lower() for term in ("liver", "hepatic", "impression"))
    ]


def _evidence(
    target_field: str, text: str, match: re.Match[str]
) -> ParsedEvidence:
    start, end = match.span()
    return ParsedEvidence(target_field, text[start:end], start, end)


def _unique_evidence(items: list[ParsedEvidence]) -> list[ParsedEvidence]:
    seen: set[tuple[str, int, int]] = set()
    unique = []
    for item in items:
        key = (item.target_field, item.start, item.end)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _has_lesion_mention(sentences: list[str]) -> bool:
    for sentence in sentences:
        mentions_lesion = re.search(
            r"(?:liver|hepatic)[^.]*?(?:lesion|mass|nodule|tumor)",
            sentence,
            re.I,
        ) or re.search(
            r"(?:lesion|mass|tumor)[^.]*?(?:in|of) (?:the )?(?:liver|hepatic segment)",
            sentence,
            re.I,
        )
        if mentions_lesion and not re.search(
            r"\bno\b|\bwithout\b|\bnegative\b|\bfree of\b", sentence, re.I
        ):
            return True
    return False


def _presence_from_sentences(sentences: list[str]) -> tuple[bool | None, str | None]:
    """Classify lesion presence without treating an absent mention as negative."""
    text = " ".join(sentences)
    if not text:
        return None, "not_mentioned"

    has_lesion_language = bool(
        re.search(
            r"(?:liver|hepatic)[^.]*?(?:lesion|mass|nodule|tumor)"
            r"|(?:lesion|mass|tumor)[^.]*?(?:in|of) (?:the )?"
            r"(?:liver|hepatic segment)",
            text,
            re.I,
        )
    )
    if not has_lesion_language:
        return None, "not_mentioned"

    uncertain = bool(
        re.search(
            r"(?:cannot be excluded|cannot exclude|cannot rule out|"
            r"possible|possibly|questionable|indeterminate|suspicious for|"
            r"suspected|suggestive of)",
            text,
            re.I,
        )
    )
    negative = bool(
        re.search(
            r"\b(?:no|without|negative for|free of)\b"
            r"[^.]{0,80}\b(?:liver|hepatic)?\s*"
            r"(?:lesion|lesions|mass|masses|nodule|nodules|tumor|tumors)\b",
            text,
            re.I,
        )
        or re.search(
            r"\b(?:no|without)\s+(?:focal\s+)?"
            r"(?:liver|hepatic)\s+(?:lesion|lesions|mass|masses)\b",
            text,
            re.I,
        )
    )
    affirmative = _has_lesion_mention(sentences)

    if uncertain or (negative and affirmative):
        return None, "conflicting_terms" if negative and affirmative else "ambiguous"
    if negative:
        return False, None
    if affirmative:
        return True, None
    return None, "not_mentioned"


def _extract_lesion_count(text: str) -> int | None:
    match = re.search(r"total of (\d+)(?: liver)? lesions", text, re.I)
    if match:
        return int(match.group(1))

    match = re.search(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten) additional liver lesions",
        text,
        re.I,
    )
    if match:
        value = match.group(1).lower()
        return WORD_TO_NUM.get(value, int(value) if value.isdigit() else 0) + 1

    match = re.search(
        r"\b(a|an|one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:single\s+)?(?:small\s+|large\s+)?(?:\w+attenuating\s+)?"
        r"liver (?:lesions|masses|lesion|mass)",
        text,
        re.I,
    )
    if match:
        value = match.group(1).lower()
        return WORD_TO_NUM.get(value, int(value) if value.isdigit() else None)

    match = re.search(
        r"\b(a|an|one|single)\s+(?:small\s+|large\s+)?"
        r"(?:\w+attenuating\s+)?liver (?:lesion|mass)",
        text,
        re.I,
    )
    if match:
        return 1

    lowered = text.lower()
    for phrase, count in (
        ("contains two lesions", 2),
        ("contains three lesions", 3),
        ("contains four lesions", 4),
    ):
        if phrase in lowered:
            return count
    return None


def _extract_location_segments(text: str) -> list[int] | None:
    match = re.search(
        r"hepatic segment[s]?\s*([1-8\s,and/-]+?)"
        r"(?:\s+measuring|\s+with|\s+and\s+has|\.|\(|\n|,?\s*measur)",
        text,
        re.I,
    )
    if not match:
        return None
    return sorted({int(value) for value in re.findall(r"[1-8]", match.group(1))})


def _extract_dimensions(text: str) -> tuple[list[float] | None, float | None]:
    patterns = (
        r"largest (?:liver )?(?:lesion|mass|one)[^.]*?measuring\s*"
        r"(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm",
        r"(?:liver|hepatic)\s+(?:lesion|mass)[^.]*?measuring\s*"
        r"(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm",
        r"measuring\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm"
        r"[^.]*?hepatic segment",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            break
    if not match:
        return None, None
    first = float(match.group(1))
    second = float(match.group(2)) if match.group(2) else None
    dimensions = [first, second] if second is not None else [first]
    return dimensions, max(dimensions)


def _extract_attenuation(
    text: str,
    liver_text: str,
    *,
    include_broad_fallback: bool,
) -> str | None:
    patterns = (
        r"(hypoattenuating|hyperattenuating|isoattenuating)\s+"
        r"(?:relative to the liver\s+)?liver (?:lesion|mass)",
        r"largest (?:liver )?(?:lesion|mass|one)[^.]*?"
        r"(hypoattenuating|hyperattenuating|isoattenuating)",
        r"(?:liver|hepatic)\s+(?:lesion|mass)[^.]*?"
        r"(hypoattenuating|hyperattenuating|isoattenuating)",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            break
    if not match and include_broad_fallback:
        match = re.search(
            r"(?:lesions|masses)[^.]*?"
            r"(hypoattenuating|hyperattenuating|isoattenuating)",
            text,
            re.I,
        )
    if not match and include_broad_fallback:
        match = re.search(
            r"(hypoattenuating|hyperattenuating|isoattenuating)",
            liver_text,
            re.I,
        )
    return match.group(1).lower() if match else None


def parse_liver_report(
    text: str | None,
    *,
    include_broad_attenuation_fallback: bool = False,
) -> ParsedLiverReport:
    """Parse shared liver findings without applying task-specific label logic."""
    normalized_text = text or ""
    liver_sentences = get_liver_sentences(normalized_text)
    liver_text = " ".join(liver_sentences)
    presence, presence_reason = _presence_from_sentences(liver_sentences)
    dimensions, max_diameter = _extract_dimensions(normalized_text)
    evidence: list[ParsedEvidence] = []
    sentence_spans = _sentence_spans(normalized_text)

    presence_spans = [
        (sentence, start, end)
        for sentence, start, end in sentence_spans
        if any(term in sentence.lower() for term in ("liver", "hepatic", "impression"))
        and re.search(
            r"(?:lesion|mass|nodule|tumor)", sentence, re.I
        )
    ]
    for sentence, start, end in presence_spans:
        evidence.append(
            ParsedEvidence(
                "liver_lesions.has_lesion",
                sentence,
                start,
                end,
            )
        )

    count_patterns = (
        r"total of \d+(?: liver)? lesions",
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r" additional liver lesions",
        r"\b(?:a|an|one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        r"\s+(?:single\s+)?(?:small\s+|large\s+)?(?:\w+attenuating\s+)?"
        r"liver (?:lesions|masses|lesion|mass)",
        r"\b(?:a|an|one|single)\s+(?:small\s+|large\s+)?"
        r"(?:\w+attenuating\s+)?liver (?:lesion|mass)",
        r"contains (?:two|three|four) lesions",
    )
    for pattern in count_patterns:
        for match in re.finditer(pattern, normalized_text, re.I):
            evidence.append(
                _evidence("liver_lesions.lesion_count", normalized_text, match)
            )

    location_pattern = (
        r"hepatic segment[s]?\s*[1-8\s,and/-]+?"
        r"(?=\s+measuring|\s+with|\s+and\s+has|\.|\(|\n|,?\s*measur)"
    )
    for match in re.finditer(location_pattern, normalized_text, re.I):
        evidence.append(
            _evidence("liver_lesions.largest_lesion.location_segments", normalized_text, match)
        )

    dimension_patterns = (
        r"largest (?:liver )?(?:lesion|mass|one)[^.]*?measuring\s*"
        r"\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?cm",
        r"(?:liver|hepatic)\s+(?:lesion|mass)[^.]*?measuring\s*"
        r"\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?cm",
        r"measuring\s*\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?cm",
    )
    for pattern in dimension_patterns:
        for match in re.finditer(pattern, normalized_text, re.I):
            evidence.append(
                _evidence(
                    "liver_lesions.largest_lesion.reported_dimensions_cm",
                    normalized_text,
                    match,
                )
            )
            evidence.append(
                _evidence(
                    "liver_lesions.largest_lesion.derived_max_diameter_cm",
                    normalized_text,
                    match,
                )
            )

    attenuation_pattern = (
        r"(?:hypoattenuating|hyperattenuating|isoattenuating)"
        r"(?:\s+relative to the liver)?"
        r"(?:\s+(?:liver|hepatic)\s+(?:lesion|mass))?"
    )
    for match in re.finditer(attenuation_pattern, normalized_text, re.I):
        sentence = normalized_text[
            normalized_text.rfind(".", 0, match.start()) + 1 :
            normalized_text.find(".", match.end())
            if normalized_text.find(".", match.end()) != -1
            else len(normalized_text)
        ]
        if any(term in sentence.lower() for term in ("liver", "hepatic", "lesion", "mass")):
            evidence.append(
                _evidence(
                    "liver_lesions.largest_lesion.attenuation",
                    normalized_text,
                    match,
                )
            )

    return ParsedLiverReport(
        liver_sentences=liver_sentences,
        liver_text=liver_text,
        has_mention=presence is True,
        presence=presence,
        presence_reason=presence_reason,
        lesion_count=_extract_lesion_count(normalized_text),
        location_segments=_extract_location_segments(normalized_text),
        reported_dimensions_cm=dimensions,
        derived_max_diameter_cm=max_diameter,
        attenuation=_extract_attenuation(
            normalized_text,
            liver_text,
            include_broad_fallback=include_broad_attenuation_fallback,
        ),
        evidence=_unique_evidence(evidence),
    )
