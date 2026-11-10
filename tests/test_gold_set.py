"""End-to-end tests for the two-annotator gold-set workflow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from liversira_ai.nlp.data.adjudication import adjudicate_cohort
from liversira_ai.nlp.data.gold_set import (
    ANNOTATION_COLUMNS,
    compare_annotations,
    create_sample,
    finalize,
)
from liversira_ai.nlp.schemas.ground_truth_schema import (
    GroundTruthLabels,
    GroundTruthRecord,
    LargestLesionGT,
)

POSITIVE_IDS = ["P1", "P2", "P3"]
NEGATIVE_IDS = ["N1", "N2", "N3"]


def _setup(tmp_path: Path) -> dict[str, Path]:
    master, gt, audit = [], [], []
    for record_id in POSITIVE_IDS + NEGATIVE_IDS:
        positive = record_id.startswith("P")
        master.append(
            {
                "id": record_id,
                "reports": {
                    "narrative_original": f"Report {record_id}: hypoattenuating liver lesion in hepatic segment 5."
                    if positive
                    else f"Report {record_id}: no liver lesion.",
                    "narrative_clean": "unused",
                },
            }
        )
        labels = GroundTruthLabels(
            has_lesion=positive,
            lesion_count=1 if positive else 0,
            largest_lesion=LargestLesionGT(
                location_segments=[5], source_diameter_cm=2.0, attenuation="hypoattenuating"
            )
            if positive
            else None,
        )
        gt.append(GroundTruthRecord(id=record_id, ground_truth=labels).model_dump_json())
        audit.append(
            {
                "id": record_id,
                "has_lesion_gt": positive,
                "has_lesion_status": "MATCH",
                "count_gt": 1 if positive else 0,
                "overall_status": "MATCH",
                "discrepancy_details": "",
            }
        )
    paths = {
        "master": tmp_path / "master.jsonl",
        "gt": tmp_path / "gt.jsonl",
        "audit": tmp_path / "audit.csv",
        "splits": tmp_path / "splits.json",
        "work": tmp_path / "work",
        "manifest": tmp_path / "results" / "manifest.json",
    }
    paths["master"].write_text("\n".join(json.dumps(r) for r in master) + "\n", encoding="utf-8")
    paths["gt"].write_text("\n".join(gt) + "\n", encoding="utf-8")
    pd.DataFrame(audit).to_csv(paths["audit"], index=False)
    paths["splits"].write_text(
        json.dumps({"train": [], "val": ["P1", "P2", "N1", "N2"], "test": ["P3", "N3"]}),
        encoding="utf-8",
    )
    return paths


def _sample(paths: dict[str, Path]) -> pd.DataFrame:
    return create_sample(
        master_path=paths["master"],
        audit_path=paths["audit"],
        splits_path=paths["splits"],
        output_dir=paths["work"],
        manifest_path=paths["manifest"],
        allocation={stratum: None for stratum in ("label_conflict", "positive_multiple", "positive_single", "negative_not_stated", "negative_stated")},
    )


def _fill(path: Path, reviewer: str, overrides: dict[str, dict[str, str]] | None = None) -> None:
    sheet = pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    for index, record_id in enumerate(sheet["BDMAP ID"]):
        positive = record_id.startswith("P")
        values = {
            "review_presence": "YES" if positive else "NO",
            "reviewed_count": "1" if positive else "0",
            "reviewed_location": "5" if positive else "NOT_APPLICABLE",
            "reviewed_diameter_cm": "UNDOCUMENTED" if positive else "NOT_APPLICABLE",
            "reviewed_attenuation": "hypoattenuating" if positive else "NOT_APPLICABLE",
            "evidence_quote": "liver lesion" if positive else "no liver lesion",
            "review_notes": "ok",
            "reviewer_id": reviewer,
            "reviewed_at": "2026-11-10T10:00:00+00:00",
        }
        values.update((overrides or {}).get(record_id, {}))
        for column, value in values.items():
            sheet.loc[index, column] = value
    sheet.to_csv(path, index=False, encoding="utf-8-sig")


def test_sample_sheets_are_blinded_identical_and_protected(tmp_path: Path):
    paths = _setup(tmp_path)
    key = _sample(paths)

    sheet_a = (paths["work"] / "annotation_sheet_A.csv").read_text(encoding="utf-8-sig")
    sheet_b = (paths["work"] / "annotation_sheet_B.csv").read_text(encoding="utf-8-sig")
    assert sheet_a == sheet_b
    header = sheet_a.splitlines()[0].split(",")
    assert header == ["BDMAP ID", "report_text", *ANNOTATION_COLUMNS]
    assert set(key["BDMAP ID"]) == set(POSITIVE_IDS + NEGATIVE_IDS)
    assert set(key["inclusion_probability"]) == {1.0}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["total_sampled"] == 6 and "P1" not in json.dumps(manifest)

    with pytest.raises(FileExistsError):
        _sample(paths)


def test_full_workflow_produces_gold_standard(tmp_path: Path):
    paths = _setup(tmp_path)
    _sample(paths)
    work = paths["work"]
    _fill(work / "annotation_sheet_A.csv", "ann-a")
    _fill(
        work / "annotation_sheet_B.csv",
        "ann-b",
        {
            "P2": {"reviewed_count": "2"},  # disagreement -> clinician
            "N2": {"review_presence": "UNCLEAR"},  # unclear -> clinician
        },
    )

    agreement = compare_annotations(
        work / "annotation_sheet_A.csv",
        work / "annotation_sheet_B.csv",
        work / "adjudication.csv",
        tmp_path / "results" / "agreement.json",
    )
    assert agreement["studies"] == 6
    assert agreement["needs_clinician"] == 2
    assert agreement["presence"]["unclear_by_either"] == 1
    # P2 differs and N2 is UNCLEAR for annotator B, so 4 of 6 counts agree.
    assert agreement["lesion_count"]["exact_agreement"] == pytest.approx(4 / 6)

    sheet = pd.read_csv(work / "adjudication.csv", dtype=object, keep_default_na=False, encoding="utf-8-sig")
    pending = sheet.set_index("BDMAP ID")["needs_clinician"]
    assert pending["P2"] == "YES" and pending["N2"] == "YES" and pending["P1"] == "NO"
    # The clinician resolves P2 (count 2) and leaves N2 for later.
    row = sheet.index[sheet["BDMAP ID"] == "P2"][0]
    sheet.loc[row, ["review_presence", "reviewed_count", "reviewed_location", "reviewed_diameter_cm",
                    "reviewed_attenuation", "review_notes", "reviewer_id", "reviewed_at"]] = [
        "YES", "2", "5", "UNDOCUMENTED", "hypoattenuating", "Report lists two lesions.",
        "clinician-1", "2026-11-11T09:00:00+00:00",
    ]
    sheet.to_csv(work / "adjudication.csv", index=False, encoding="utf-8-sig")

    finalize(work / "adjudication.csv", paths["gt"], work / "review.csv")
    gold_path = tmp_path / "gold.jsonl"
    assert adjudicate_cohort(work / "review.csv", paths["gt"], gold_path) == 2  # N1, N3

    gold = {r["id"]: r for r in map(json.loads, gold_path.read_text(encoding="utf-8").splitlines())}
    assert set(gold) == {"P1", "P2", "P3", "N1", "N3"}
    assert gold["P1"]["provenance"]["method"] == "consensus"
    assert gold["P1"]["provenance"]["annotators"] == ["ann-a", "ann-b"]
    assert gold["P1"]["gold"]["largest_lesion"]["diameter_cm"] is None
    assert gold["P2"]["provenance"]["method"] == "clinician_adjudication"
    assert gold["P2"]["provenance"]["adjudicator"] == "clinician-1"
    assert gold["P2"]["gold"]["lesion_count"] == 2
    assert gold["P2"]["provenance"]["source_label_agreement"]["lesion_count"] is False

    statuses = {
        r["id"]: r["validation"]["status"]
        for r in map(json.loads, paths["gt"].read_text(encoding="utf-8").splitlines())
    }
    assert statuses["N1"] == "verified"
    assert statuses["P1"] == "partially_verified"  # diameter undocumented in the report
    assert statuses["N2"] == "needs_adjudication"  # clinician still pending


def test_same_reviewer_on_both_sheets_is_rejected(tmp_path: Path):
    paths = _setup(tmp_path)
    _sample(paths)
    work = paths["work"]
    _fill(work / "annotation_sheet_A.csv", "same")
    _fill(work / "annotation_sheet_B.csv", "same")
    with pytest.raises(ValueError, match="same reviewer"):
        compare_annotations(
            work / "annotation_sheet_A.csv",
            work / "annotation_sheet_B.csv",
            work / "adjudication.csv",
            tmp_path / "agreement.json",
        )
