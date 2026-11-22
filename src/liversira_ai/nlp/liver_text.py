"""Keep only the liver-related text of a report, with offsets into the source.

The report is cut into sections (``IMPRESSION:``, ``FINDINGS:``...), units
(sentences, bullets, numbered items) and clauses. A clause is kept when it is
about the liver:

- it names the liver (``liver``, ``hepatic``...) and no other organ;
- it names no organ and follows a liver clause ("This lesion is hypoattenuating");
- it names the liver among other organs and cannot be split further
  ("The spleen, liver, pancreas, and kidneys are normal").

A unit that names the liver and another organ is split at commas and
semicolons when the next piece starts with its own organ ("..., and the largest
kidney mass measures 1.3 cm"). Reference organs in comparisons ("relative to
the kidney") do not count. Kept text is copied verbatim, joined by spaces, and a
clause without final punctuation gets a closing period; ``spans`` gives its
offsets in the input, so evidence found in the liver text can be mapped back.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# 1.0.1: a kept clause without final punctuation ends with "." (was: only between clauses).
LIVER_TEXT_VERSION = "1.0.1"

_LIVER = re.compile(
    r"\bliver\b|\b(?:intra|extra)?hepat(?:ic|o\w*)\b(?!\s+(?:arter|flexure))|\bcouinaud\b",
    re.I,
)
_OTHER_ORGANS = re.compile(
    r"\b(?:"
    r"spleen|splen(?:ic|omegaly)|pancrea\w*|kidneys?|renal|adrenals?|"
    r"colon\w*|rect(?:um|al)|sigmoid|cec(?:um|al)|bowel|intestin\w*|duoden\w*|jejun\w*|ile(?:um|al)|"
    r"stomach|gastric|esophag\w*|gallbladder|bladder|uter(?:us|ine)|ovar(?:y|ies|ian)|prostat\w*|"
    r"lungs?|pulmonary|aort\w*|vena\s+cava|portal\s+vein|mesenteric|celiac|"
    r"(?:common\s+)?hepatic\s+arter\w*|SMA|SMV|IVC|CHA|lymph|nodes?|bones?|osseous|vertebra\w*|spine"
    r")\b",
    re.I,
)
_ORGAN = re.compile(rf"(?:{_LIVER.pattern})|(?:{_OTHER_ORGANS.pattern})", re.I)
# "hypoattenuating relative to the kidney" names a reference, not the subject.
_COMPARISON = re.compile(
    r"\b(?:relative|compared|in\s+comparison)\s+(?:to|with)\s+|\bthan\s+",
    re.I,
)
_REFERENCE_TAIL = re.compile(
    r"(?:the\s+)?(?:surrounding\s+|adjacent\s+|normal\s+|background\s+)?(?:\w+\s+)?"
    rf"(?:{_ORGAN.pattern})(?:\s+parenchyma)?",
    re.I,
)

# Section headers: one or more upper-case words ending in a colon.
_HEADER = re.compile(r"\b[A-Z][A-Z/&]+(?:\s+[A-Z][A-Z/&]+)*:")
# Unit boundaries: sentence ends, bullets (" - Liver"), numbered items (" 2. A ...").
_UNIT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=\(?[A-Z])"  # sentence end
    r"|\s+(?=-\s+[A-Z])"  # bullet
    r"|(?<=[.!?:])\s+(?=\d{1,2}\.\s+[A-Z])"  # numbered item
)
_MARKER = re.compile(r"^(?:-\s+|\d{1,2}\.\s+)")
_BARE_MARKER = re.compile(r"^(?:-|\d{1,2}\.)$")
_PIECE_BOUNDARY = re.compile(r"\s*[,;]\s+")
_LEADING_LINK = re.compile(r"^(?:and|or|but|while|whereas|with|as\s+well\s+as)\s+", re.I)
# A bare list item ("pancreas", "or kidneys"): no predicate of its own.
_BARE_ITEM = re.compile(rf"^(?:(?:and|or)\s+)?(?:the\s+)?(?:{_ORGAN.pattern})\.?$", re.I)
_SENTENCE_END = re.compile(r"[.!?:]$")
_AND = re.compile(r"\s+and\s+", re.I)


@dataclass(frozen=True)
class LiverText:
    text: str
    spans: list[tuple[int, int]]


@dataclass
class _Clause:
    start: int
    end: int
    organs: frozenset[str]  # subset of {"liver", "other"}


def _strip_comparisons(text: str) -> str:
    out, cursor = [], 0
    for match in _COMPARISON.finditer(text):
        if match.start() < cursor:
            continue
        tail = _REFERENCE_TAIL.match(text, match.end())
        if tail is None:
            continue
        out.append(text[cursor : match.start()])
        out.append(" " * (tail.end() - match.start()))
        cursor = tail.end()
    out.append(text[cursor:])
    return "".join(out)


def organs_in(text: str) -> frozenset[str]:
    """Organs a piece of text is about: ``liver`` and/or ``other``."""
    cleaned = _strip_comparisons(text)
    found = set()
    if _LIVER.search(cleaned):
        found.add("liver")
    if _OTHER_ORGANS.search(cleaned):
        found.add("other")
    return frozenset(found)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " ,;":
        start += 1
    while end > start and text[end - 1] in " ,;":
        end -= 1
    return start, end


def _sections(text: str) -> list[tuple[tuple[int, int] | None, int, int]]:
    """(header span or None, body start, body end) for each section."""
    sections, header, cursor = [], None, 0
    for match in _HEADER.finditer(text):
        if len(re.sub(r"[^A-Z]", "", match.group(0))) < 5:
            continue  # "HU:", "SMA:" are not headers
        sections.append((header, cursor, match.start()))
        header, cursor = (match.start(), match.end()), match.end()
    sections.append((header, cursor, len(text)))
    return sections


def _units(text: str, start: int, end: int) -> list[tuple[int, int]]:
    units, cursor = [], start
    body = text[start:end]
    for match in _UNIT_BOUNDARY.finditer(body):
        units.append(_trim(text, cursor, start + match.start()))
        cursor = start + match.end()
    units.append(_trim(text, cursor, end))
    merged: list[tuple[int, int]] = []
    pending: int | None = None  # start of a bare marker ("2.") waiting for its item
    for s, e in units:
        if e <= s:
            continue
        if _BARE_MARKER.match(text[s:e]):
            pending = s if pending is None else pending
            continue
        merged.append((pending if pending is not None else s, e))
        pending = None
    return merged


def _starts_with_organ(piece: str) -> bool:
    words = _LEADING_LINK.sub("", piece).split()
    return bool(_ORGAN.search(" ".join(words[:5])))


def _clauses(text: str, start: int, end: int) -> list[_Clause]:
    """Split a unit into clauses; only units naming the liver and another organ are split."""
    unit = text[start:end]
    organs = organs_in(unit)
    if organs != {"liver", "other"}:
        return [_Clause(start, end, organs)]

    pieces, cursor = [], 0
    for match in _PIECE_BOUNDARY.finditer(unit):
        pieces.append((cursor, match.start()))
        cursor = match.end()
    pieces.append((cursor, len(unit)))
    if any(_BARE_ITEM.match(unit[s:e].strip()) for s, e in pieces):
        return [_Clause(start, end, organs)]  # an organ list shares one predicate

    groups: list[list[int]] = []
    for s, e in pieces:
        if not groups or _starts_with_organ(unit[s:e]):
            groups.append([s, e])
        else:
            groups[-1][1] = e
    clauses = []
    for s, e in groups:
        link = _LEADING_LINK.match(unit[s:e])
        if link and groups[0][0] != s:
            s += link.end()
        for cs, ce in _split_and(text, start + s, start + e):
            clauses.append(_Clause(cs, ce, organs_in(text[cs:ce])))
    return clauses


def _split_and(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split "an enlarged spleen and multiple liver lesions" where each side names one organ."""
    for match in _AND.finditer(text, start, end):
        left, right = (start, match.start()), (match.end(), end)
        left_organs, right_organs = organs_in(text[slice(*left)]), organs_in(text[slice(*right)])
        if (
            len(left_organs) == len(right_organs) == 1
            and left_organs != right_organs
            and not _BARE_ITEM.match(text[slice(*left)].strip())
            and not _BARE_ITEM.match(text[slice(*right)].strip())
        ):
            return [left, *_split_and(text, *right)]
    return [(start, end)]


def extract_liver_text(text: str | None) -> LiverText:
    """Return the liver-related text of ``text`` and its spans in ``text``."""
    if not text:
        return LiverText("", [])

    kept: list[tuple[int, int]] = []
    for header, body_start, body_end in _sections(text):
        topic: frozenset[str] = frozenset()  # organs of the last clause that named one
        section_kept: list[tuple[int, int]] = []
        for unit_start, unit_end in _units(text, body_start, body_end):
            marker = _MARKER.match(text, unit_start)
            content_start = marker.end() if marker else unit_start
            for index, clause in enumerate(_clauses(text, content_start, unit_end)):
                if clause.organs:
                    topic = clause.organs
                if "liver" not in topic:
                    continue
                start = unit_start if index == 0 else clause.start
                section_kept.append(_trim(text, start, clause.end))
        if section_kept and header is not None:
            kept.append(header)
        kept.extend(section_kept)

    # A clause cut from a longer sentence gets a closing period, so the next
    # sentence (or text inserted after it) does not run on.
    pieces = [text[s:e] if _SENTENCE_END.search(text[s:e]) else text[s:e] + "." for s, e in kept]
    return LiverText(" ".join(pieces), kept)
