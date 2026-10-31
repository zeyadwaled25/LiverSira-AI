"""Rule-Based NLP Baseline Extraction Pipeline.

Extracts structured liver findings from raw/cleaned radiology narrative reports:
1. Liver Lesion Presence (has_lesion: bool)
2. Number of Liver Lesions (lesion_count: int)
3. Largest Lesion Location (location_segments: list[int] in Couinaud 1-8)
4. Largest Lesion Dimensions/Diameter (reported_dimensions_cm and derived_max_diameter_cm)
5. Largest Lesion Attenuation (hypo/iso/hyperattenuating)

Outputs structured predictions following schemas.inference_schema.InferenceRecord
into evaluation/baseline_inferences.jsonl and calculates performance metrics.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from schemas.inference_schema import (
    InferenceRecord,
    InputReference,
    LargestLesionInference,
    LiverLesionsInference,
    StudyMetadata,
)

MASTER_RECORDS_PATH = Path("data/processed/master_records.jsonl")
SPLITS_JSON_PATH = Path("data/processed/splits.json")
OUTPUT_INFERENCE_PATH = Path("evaluation/baseline_inferences.jsonl")

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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting decimal numbers."""
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"\.(?:\s+(?=[A-Z])|\s*$)", text)
        if s.strip()
    ]


def extract_liver_lesion_features(text: str | None) -> dict[str, Any]:
    """Rule-based extractor for the 4 core liver lesion targets."""
    if not text:
        return {
            "has_lesion": False,
            "lesion_count": 0,
            "location_segments": None,
            "reported_dimensions_cm": None,
            "derived_max_diameter_cm": None,
            "attenuation": None,
        }

    sents = split_sentences(text)
    liver_sents = [
        s
        for s in sents
        if any(k in s.lower() for k in ["liver", "hepatic", "impression"])
    ]
    liver_text = " ".join(liver_sents)

    # 1. Affirmative liver lesion detection
    has_mention = False
    for s in liver_sents:
        if re.search(
            r"(?:liver|hepatic)[^.]*?(?:lesion|mass|nodule|tumor)", s, re.I
        ) or re.search(
            r"(?:lesion|mass|tumor)[^.]*?(?:in|of) (?:the )?(?:liver|hepatic segment)",
            s,
            re.I,
        ):
            if not re.search(r"\bno\b|\bwithout\b|\bnegative\b|\bfree of\b", s, re.I):
                has_mention = True
                break

    if not has_mention:
        return {
            "has_lesion": False,
            "lesion_count": 0,
            "location_segments": None,
            "reported_dimensions_cm": None,
            "derived_max_diameter_cm": None,
            "attenuation": None,
        }

    # 2. Lesion count extraction
    found_cnt: int | None = None
    m_tot = re.search(r"total of (\d+)(?: liver)? lesions", text, re.I)
    if m_tot:
        found_cnt = int(m_tot.group(1))
    else:
        m_add = re.search(
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten) additional liver lesions",
            text,
            re.I,
        )
        if m_add:
            w = m_add.group(1).lower()
            add = WORD_TO_NUM.get(w, int(w) if w.isdigit() else 0)
            found_cnt = add + 1
        else:
            m_cnt = re.search(
                r"\b(a|an|one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:single\s+)?(?:small\s+|large\s+)?(?:\w+attenuating\s+)?liver (?:lesions|masses|lesion|mass)",
                text,
                re.I,
            )
            if m_cnt:
                w = m_cnt.group(1).lower()
                found_cnt = WORD_TO_NUM.get(w, int(w) if w.isdigit() else None)
            else:
                m_single = re.search(
                    r"\b(a|an|one|single)\s+(?:small\s+|large\s+)?(?:\w+attenuating\s+)?liver (?:lesion|mass)",
                    text,
                    re.I,
                )
                if m_single:
                    found_cnt = 1
                elif "contains two lesions" in text.lower():
                    found_cnt = 2
                elif "contains three lesions" in text.lower():
                    found_cnt = 3
                elif "contains four lesions" in text.lower():
                    found_cnt = 4

    # Default to 1 if lesion confirmed affirmative but unquantified
    if found_cnt is None:
        found_cnt = 1

    # 3. Location segments (Couinaud 1-8)
    m_seg = re.search(
        r"hepatic segment[s]?\s*([1-8\s,and/-]+?)(?:\s+measuring|\s+with|\s+and\s+has|\.|\(|\n|,?\s*measur)",
        text,
        re.I,
    )
    segs = (
        sorted(list(set([int(x) for x in re.findall(r"[1-8]", m_seg.group(1))])))
        if m_seg
        else None
    )

    # 4. Dimensions and Diameter
    m_diam = re.search(
        r"largest (?:liver )?(?:lesion|mass|one)[^.]*?measuring\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm",
        text,
        re.I,
    )
    if not m_diam:
        m_diam = re.search(
            r"(?:liver|hepatic)\s+(?:lesion|mass)[^.]*?measuring\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm",
            text,
            re.I,
        )
    if not m_diam:
        m_diam = re.search(
            r"measuring\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*cm[^.]*?hepatic segment",
            text,
            re.I,
        )

    dimensions: list[float] | None = None
    max_d: float | None = None
    if m_diam:
        d1 = float(m_diam.group(1))
        d2 = float(m_diam.group(2)) if m_diam.group(2) else None
        dimensions = [d1, d2] if d2 is not None else [d1]
        max_d = max(dimensions)

    # 5. Attenuation
    m_att = re.search(
        r"(hypoattenuating|hyperattenuating|isoattenuating)\s+(?:relative to the liver\s+)?liver (?:lesion|mass)",
        text,
        re.I,
    )
    if not m_att:
        m_att = re.search(
            r"largest (?:liver )?(?:lesion|mass|one)[^.]*?(hypoattenuating|hyperattenuating|isoattenuating)",
            text,
            re.I,
        )
    if not m_att:
        m_att = re.search(
            r"(?:liver|hepatic)\s+(?:lesion|mass)[^.]*?(hypoattenuating|hyperattenuating|isoattenuating)",
            text,
            re.I,
        )
    if not m_att:
        m_att = re.search(
            r"(?:lesions|masses)[^.]*?(hypoattenuating|hyperattenuating|isoattenuating)",
            text,
            re.I,
        )
    if not m_att:
        m_att = re.search(
            r"(hypoattenuating|hyperattenuating|isoattenuating)",
            liver_text,
            re.I,
        )

    att = m_att.group(1).lower() if m_att else None

    return {
        "has_lesion": True,
        "lesion_count": found_cnt,
        "location_segments": segs,
        "reported_dimensions_cm": dimensions,
        "derived_max_diameter_cm": max_d,
        "attenuation": att,
    }


def run_baseline_inference(
    master_path: Path,
    output_path: Path,
    target_split_ids: set[str] | None = None,
) -> list[InferenceRecord]:
    """Run baseline on master records and write structured InferenceRecord JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferences: list[InferenceRecord] = []

    print(f"Reading master records: {master_path}")
    with open(master_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for idx, line in enumerate(f_in, start=1):
            master_rec = json.loads(line)
            rec_id = master_rec["id"]

            if target_split_ids is not None and rec_id not in target_split_ids:
                continue

            text = master_rec.get("reports", {}).get("narrative_clean")
            features = extract_liver_lesion_features(text)

            if features["has_lesion"]:
                largest_obj = LargestLesionInference(
                    location_segments=features["location_segments"],
                    reported_dimensions_cm=features["reported_dimensions_cm"],
                    derived_max_diameter_cm=features["derived_max_diameter_cm"],
                    attenuation=features["attenuation"],
                )
            else:
                largest_obj = None

            inference_rec = InferenceRecord(
                id=rec_id,
                input_reference=InputReference(
                    field="narrative_clean", text_version="1.0"
                ),
                study=StudyMetadata(modality="CT"),
                liver_lesions=LiverLesionsInference(
                    has_lesion=features["has_lesion"],
                    lesion_count=features["lesion_count"],
                    largest_lesion=largest_obj,
                ),
                evidence_spans=[],
                missing_or_unclear=[],
                confidence=None,
            )

            f_out.write(inference_rec.model_dump_json() + "\n")
            inferences.append(inference_rec)

            if len(inferences) % 2000 == 0:
                print(f"Inferred {len(inferences):,} records...")

    print(f"[DONE] Saved {len(inferences):,} baseline inference records to: {output_path}")
    return inferences


def main() -> None:
    """CLI execution."""
    run_baseline_inference(MASTER_RECORDS_PATH, OUTPUT_INFERENCE_PATH)


if __name__ == "__main__":
    main()
