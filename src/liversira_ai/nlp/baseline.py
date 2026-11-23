"""Rule-based extraction: presence, count, and the largest lesion's segments, size and attenuation."""

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
from liversira_ai.nlp.parser_v1_1 import PARSER_VERSION, parse_liver_report
from liversira_ai.nlp import parser_v1_2

MASTER_RECORDS_PATH = Path("data/processed/nlp/master_records.jsonl")
OUTPUT_INFERENCE_PATH = Path("data/processed/nlp/baseline_inferences.jsonl")
OUTPUT_MANIFEST_PATH = Path("data/processed/nlp/baseline_manifest.json")
# v1.0.0 outputs (results/nlp/predictions/baseline_inferences_updated.jsonl) were
# produced with the frozen liversira_ai.nlp.parser_v1_0.
BASELINE_VERSION = "rule-based-v1.1.0"

def extract_liver_lesion_features(text: str | None, parse=parse_liver_report) -> dict[str, Any]:
    """Extract liver targets while preserving unknown presence as ``None``.

    ``parse`` defaults to the frozen parser v1.1; see ``extract_liver_lesion_features_v1_2``."""
    parsed = parse(text, include_broad_attenuation_fallback=True)
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

    # v1.1 never imputes: a field the report does not state stays None and is
    # listed with its reason (v1.0 set an unstated count to 1).
    missing = []
    if parsed.lesion_count is None:
        missing.append(
            {"field": "liver_lesions.lesion_count", "reason": parsed.count_reason or "not_mentioned"}
        )
    for field, value in (
        ("location_segments", parsed.location_segments),
        ("reported_dimensions_cm", parsed.reported_dimensions_cm),
        ("attenuation", parsed.attenuation),
    ):
        if value is None:
            missing.append(
                {"field": f"liver_lesions.largest_lesion.{field}", "reason": "not_mentioned"}
            )
    return {
        "has_lesion": True,
        "lesion_count": parsed.lesion_count,
        "location_segments": parsed.location_segments,
        "reported_dimensions_cm": parsed.reported_dimensions_cm,
        "derived_max_diameter_cm": parsed.derived_max_diameter_cm,
        "attenuation": parsed.attenuation,
        "missing_or_unclear": missing,
        "evidence": [item.__dict__ for item in parsed.evidence],
    }


def extract_liver_lesion_features_v1_2(text: str | None) -> dict[str, Any]:
    """Parser v1.2 (rules tuned on 6.2 Train-aug only) in the same output form."""
    return extract_liver_lesion_features(text, parse=parser_v1_2.parse_liver_report)


def run_baseline_inference(
    master_path: Path,
    output_path: Path,
    target_split_ids: set[str] | None = None,
    manifest_path: Path | None = None,
) -> list[InferenceRecord]:
    """Run baseline on master records and write structured InferenceRecord JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferences: list[InferenceRecord] = []

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
        "parser": "liversira_ai.nlp.parser_v1_1.parse_liver_report",
        "parser_version": PARSER_VERSION,
        "split_filter": None if target_split_ids is None else f"{len(target_split_ids)} selected IDs",
        "parser_options": {"include_broad_attenuation_fallback": True},
        "input_path": master_path.as_posix(),
        "input_sha256": hashlib.sha256(master_path.read_bytes()).hexdigest(),
        "output_path": output_path.as_posix(),
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
