"""Rule-Based NLP Baseline Extraction Pipeline.

Extracts structured liver findings from raw/cleaned radiology narrative reports:
1. Liver Lesion Presence (has_lesion: bool)
2. Number of Liver Lesions (lesion_count: int)
3. Largest Lesion Location (location_segments: list[int] in Couinaud 1-8)
4. Largest Lesion Dimensions/Diameter (reported_dimensions_cm and derived_max_diameter_cm)
5. Largest Lesion Attenuation (hypo/iso/hyperattenuating)

Outputs structured predictions following liversira_ai.nlp.schemas.InferenceRecord.
"""

from pathlib import Path
import json
import hashlib
from typing import Any

from liversira_ai.nlp.schemas.inference_schema import (
    InferenceRecord,
    InputReference,
    EvidenceSpan,
    LargestLesionInference,
    LiverLesionsInference,
    MissingOrUnclearField,
    StudyMetadata,
)
from liversira_ai.nlp.parser import parse_liver_report

MASTER_RECORDS_PATH = Path("data/processed/nlp/master_records.jsonl")
OUTPUT_INFERENCE_PATH = Path("data/processed/nlp/baseline_inferences.jsonl")
OUTPUT_MANIFEST_PATH = Path("data/processed/nlp/baseline_manifest.json")
BASELINE_VERSION = "rule-based-v1.0.0"

def extract_liver_lesion_features(text: str | None) -> dict[str, Any]:
    """Extract liver targets while preserving unknown presence as ``None``."""
    parsed = parse_liver_report(text, include_broad_attenuation_fallback=True)
    if parsed.presence is False:
        return {
            "has_lesion": False,
            "lesion_count": 0,
            "location_segments": None,
            "reported_dimensions_cm": None,
            "derived_max_diameter_cm": None,
            "attenuation": None,
            "missing_or_unclear": [],
            "evidence": [item.__dict__ for item in parsed.evidence],
        }

    if parsed.presence is None:
        return {
            "has_lesion": None,
            "lesion_count": None,
            "location_segments": None,
            "reported_dimensions_cm": None,
            "derived_max_diameter_cm": None,
            "attenuation": None,
            "missing_or_unclear": [
                {
                    "field": "liver_lesions.has_lesion",
                    "reason": parsed.presence_reason or "ambiguous",
                }
            ],
            "evidence": [item.__dict__ for item in parsed.evidence],
        }

    return {
        "has_lesion": True,
        "lesion_count": (
            parsed.lesion_count if parsed.lesion_count is not None else 1
        ),
        "location_segments": parsed.location_segments,
        "reported_dimensions_cm": parsed.reported_dimensions_cm,
        "derived_max_diameter_cm": parsed.derived_max_diameter_cm,
        "attenuation": parsed.attenuation,
        "missing_or_unclear": [],
        "evidence": [item.__dict__ for item in parsed.evidence],
    }


def run_baseline_inference(
    master_path: Path,
    output_path: Path,
    target_split_ids: set[str] | None = None,
    manifest_path: Path | None = None,
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

            if features["has_lesion"] is True:
                largest_obj = LargestLesionInference(
                    location_segments=features["location_segments"],
                    reported_dimensions_cm=features["reported_dimensions_cm"],
                    derived_max_diameter_cm=features["derived_max_diameter_cm"],
                    attenuation=features["attenuation"],
                )
            else:
                largest_obj = None

            missing_or_unclear = [
                MissingOrUnclearField(**item)
                for item in features["missing_or_unclear"]
            ]
            evidence_spans = [
                EvidenceSpan(**item) for item in features["evidence"]
            ]

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
                evidence_spans=evidence_spans,
                missing_or_unclear=missing_or_unclear,
                confidence=None,
            )

            f_out.write(inference_rec.model_dump_json() + "\n")
            inferences.append(inference_rec)

            if len(inferences) % 2000 == 0:
                print(f"Inferred {len(inferences):,} records...")

    print(f"[DONE] Saved {len(inferences):,} baseline inference records to: {output_path}")
    if manifest_path is None:
        manifest_path = (
            OUTPUT_MANIFEST_PATH
            if output_path == OUTPUT_INFERENCE_PATH
            else output_path.with_name(f"{output_path.stem}_manifest.json")
        )
    manifest = {
        "baseline_version": BASELINE_VERSION,
        "input_field": "reports.narrative_clean",
        "parser": "liversira_ai.nlp.parser.parse_liver_report",
        "parser_options": {"include_broad_attenuation_fallback": True},
        "input_path": str(master_path),
        "output_path": str(output_path),
        "record_count": len(inferences),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Baseline manifest: {manifest_path}")
    return inferences


def main() -> None:
    """CLI execution."""
    run_baseline_inference(MASTER_RECORDS_PATH, OUTPUT_INFERENCE_PATH)


if __name__ == "__main__":
    main()
