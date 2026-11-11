"""Two-annotator human gold set for the liver lesion extraction task.

Workflow (each step refuses to overwrite files that may hold human work):

1. ``create_sample``: stratified, seeded sample of Validation and Test studies.
   Writes two identical *blinded* annotation sheets (report text only, no
   source labels or strata) and a separate key with strata and inclusion
   probabilities for weighted estimates.
2. Two annotators fill their sheets independently (see
   ``docs/nlp/gold_set_annotation_guidelines.md``).
3. ``compare_annotations``: inter-annotator agreement (Cohen's kappa, exact
   agreement) and an adjudication sheet. Rows where both annotators agree on
   every field are pre-filled as consensus; the rest are left for a clinician.
4. ``finalize``: converts the completed adjudication sheet into the review CSV
   consumed by ``adjudication.adjudicate_cohort``, which writes the gold
   standard and updates source-label validation status.

Test-split gold labels are for final reporting only; parser and model
development must use Validation gold labels.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from liversira_ai.common.provenance import sha256_file, sha256_text
from liversira_ai.nlp.data.adjudication import (
    NOT_APPLICABLE,
    UNDOCUMENTED,
    ReviewedValues,
    _parse_reviewed_at,
    parse_review_values,
)

ID = "BDMAP ID"
TEXT = "report_text"
ANNOTATION_COLUMNS = (
    "review_presence",
    "reviewed_count",
    "reviewed_location",
    "reviewed_diameter_cm",
    "reviewed_attenuation",
    "evidence_quote",
    "review_notes",
    "reviewer_id",
    "reviewed_at",
)
VALUE_COLUMNS = ANNOTATION_COLUMNS[:5]
UNCLEAR = "UNCLEAR"

STRATA = (
    "label_conflict",
    "positive_multiple",
    "positive_single",
    "negative_not_stated",
    "negative_stated",
)
# Per split. ``None`` takes the whole stratum.
DEFAULT_ALLOCATION: dict[str, int | None] = {
    "label_conflict": None,
    "positive_multiple": 20,
    "positive_single": 20,
    "negative_not_stated": 25,
    "negative_stated": 15,
}
DIAMETER_AGREEMENT_CM = 0.1


def _refuse_overwrite(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite files that may contain human annotations: "
            + ", ".join(existing)
        )


def assign_stratum(row: pd.Series) -> str:
    """Map one audit row to its sampling stratum."""
    if row["overall_status"] == "MISMATCH":
        return "label_conflict"
    if bool(row["has_lesion_gt"]):
        return "positive_multiple" if int(row["count_gt"]) >= 2 else "positive_single"
    if row["has_lesion_status"] == "MATCH":
        return "negative_stated"
    return "negative_not_stated"


def create_sample(
    *,
    master_path: Path,
    audit_path: Path,
    splits_path: Path,
    output_dir: Path,
    manifest_path: Path,
    seed: int = 42,
    allocation: dict[str, int | None] | None = None,
) -> pd.DataFrame:
    """Draw the stratified sample and write blinded sheets, key and manifest."""
    allocation = allocation or DEFAULT_ALLOCATION
    sheet_a = output_dir / "annotation_sheet_A.csv"
    sheet_b = output_dir / "annotation_sheet_B.csv"
    key_path = output_dir / "gold_set_key.csv"
    _refuse_overwrite(sheet_a, sheet_b, key_path)

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    audit = audit[audit["discrepancy_details"] != "Missing narrative report"]
    texts: dict[str, str] = {}
    with master_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reports = record.get("reports", {})
            text = reports.get("narrative_original") or reports.get("narrative_clean")
            if text and str(text).strip():
                texts[record["id"]] = str(text)

    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("val", "test"):
        split_audit = audit[audit["id"].isin(set(splits[split]))].copy()
        split_audit = split_audit[split_audit["id"].isin(texts)]
        split_audit["stratum"] = split_audit.apply(assign_stratum, axis=1)
        counts[split] = {}
        for stratum in STRATA:
            pool = sorted(split_audit.loc[split_audit["stratum"] == stratum, "id"])
            wanted = allocation.get(stratum)
            size = len(pool) if wanted is None else min(wanted, len(pool))
            chosen = sorted(rng.choice(pool, size=size, replace=False)) if size else []
            counts[split][stratum] = {"population": len(pool), "sampled": len(chosen)}
            for record_id in chosen:
                selected.append(
                    {
                        ID: record_id,
                        "split": split,
                        "stratum": stratum,
                        "stratum_population": len(pool),
                        "inclusion_probability": len(chosen) / len(pool),
                    }
                )

    key = pd.DataFrame(selected).sort_values(ID).reset_index(drop=True)
    order = rng.permutation(len(key))
    blinded = pd.DataFrame(
        {ID: key[ID].to_numpy()[order], TEXT: [texts[i] for i in key[ID].to_numpy()[order]]}
    )
    for column in ANNOTATION_COLUMNS:
        blinded[column] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    blinded.to_csv(sheet_a, index=False, encoding="utf-8-sig")
    blinded.to_csv(sheet_b, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "allocation_per_split": allocation,
        "strata_definitions": {
            "label_conflict": "audit overall_status MISMATCH (either label)",
            "positive_multiple": "source positive, count >= 2, no conflict",
            "positive_single": "source positive, count == 1, no conflict",
            "negative_not_stated": "source negative, report never states presence",
            "negative_stated": "source negative, report states no liver lesion",
        },
        "counts": counts,
        "total_sampled": len(key),
        "sample_ids_sha256": sha256_text("\n".join(key[ID])),
        "inputs_sha256": {
            path.as_posix(): sha256_file(path) for path in (master_path, audit_path, splits_path)
        },
        "outputs": {
            "annotation_sheets": [sheet_a.as_posix(), sheet_b.as_posix()],
            "key": key_path.as_posix(),
        },
        "blinding": "Sheets contain BDMAP ID and report text only; strata and source labels are in the key.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return key


def _read_sheet(path: Path) -> pd.DataFrame:
    sheet = pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    missing = [column for column in (ID, *ANNOTATION_COLUMNS) if column not in sheet.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if sheet[ID].duplicated().any():
        raise ValueError(f"{path} contains duplicate BDMAP IDs")
    return sheet.set_index(ID)


def _parse_annotation(row: pd.Series, label: str, record_id: str) -> ReviewedValues | None:
    """Parse one annotator row; ``None`` means presence UNCLEAR."""
    if not str(row["reviewer_id"]).strip():
        raise ValueError(f"Annotator {label} has no reviewer_id for {record_id}")
    _parse_reviewed_at(row["reviewed_at"])
    if str(row["review_presence"]).strip() == UNCLEAR:
        return None
    try:
        return parse_review_values(*(row[column] for column in VALUE_COLUMNS))
    except ValueError as exc:
        raise ValueError(f"Annotator {label}, {record_id}: {exc}") from exc


def _count_category(values: ReviewedValues | None) -> str:
    if values is None:
        return UNCLEAR
    if values.lesion_count is None:
        return UNDOCUMENTED
    return str(min(values.lesion_count, 6)) + ("+" if values.lesion_count >= 6 else "")


def _kappa(first: list[str], second: list[str]) -> float | None:
    if not first or len(set(first) | set(second)) < 2:
        return None
    return float(cohen_kappa_score(first, second))


def _agreement(first: list[Any], second: list[Any]) -> dict[str, Any]:
    matches = [a == b for a, b in zip(first, second)]
    return {
        "studies": len(matches),
        "exact_agreement": float(np.mean(matches)) if matches else None,
    }


def _same_diameter(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= DIAMETER_AGREEMENT_CM


def _value_text(values: ReviewedValues, field: str) -> str:
    if not values.has_lesion:
        return "0" if field == "count" else NOT_APPLICABLE
    value = {
        "count": values.lesion_count,
        "location": ",".join(map(str, values.location_segments or [])) or None,
        "diameter": values.diameter_cm,
        "attenuation": values.attenuation,
    }[field]
    return UNDOCUMENTED if value is None else str(value)


def compare_annotations(
    sheet_a: Path,
    sheet_b: Path,
    adjudication_path: Path,
    agreement_path: Path,
) -> dict[str, Any]:
    """Compute inter-annotator agreement and write the adjudication sheet."""
    _refuse_overwrite(adjudication_path)
    a, b = _read_sheet(sheet_a), _read_sheet(sheet_b)
    if set(a.index) != set(b.index):
        raise ValueError("Annotation sheets contain different study IDs")
    ids = sorted(a.index)

    parsed_a = {i: _parse_annotation(a.loc[i], "A", i) for i in ids}
    parsed_b = {i: _parse_annotation(b.loc[i], "B", i) for i in ids}
    for i in ids:
        if str(a.loc[i, "reviewer_id"]).strip() == str(b.loc[i, "reviewer_id"]).strip():
            raise ValueError(f"{i}: both sheets were filled by the same reviewer")

    def presence(values: ReviewedValues | None) -> str:
        return UNCLEAR if values is None else ("YES" if values.has_lesion else "NO")

    presence_a = [presence(parsed_a[i]) for i in ids]
    presence_b = [presence(parsed_b[i]) for i in ids]
    both_positive = [
        i
        for i in ids
        if parsed_a[i] is not None
        and parsed_b[i] is not None
        and parsed_a[i].has_lesion
        and parsed_b[i].has_lesion
    ]

    def field(i_list: list[str], getter) -> tuple[list[Any], list[Any]]:
        return [getter(parsed_a[i]) for i in i_list], [getter(parsed_b[i]) for i in i_list]

    count_a, count_b = field(ids, _count_category)
    location_a, location_b = field(both_positive, lambda v: tuple(v.location_segments or ()))
    diameter_pairs = field(both_positive, lambda v: v.diameter_cm)
    attenuation_a, attenuation_b = field(both_positive, lambda v: v.attenuation or UNDOCUMENTED)

    agreement = {
        "studies": len(ids),
        "presence": {
            **_agreement(presence_a, presence_b),
            "cohen_kappa": _kappa(presence_a, presence_b),
            "unclear_by_either": sum(UNCLEAR in pair for pair in zip(presence_a, presence_b)),
        },
        "lesion_count": {**_agreement(count_a, count_b), "cohen_kappa": _kappa(count_a, count_b)},
        "location_on_both_positive": _agreement(location_a, location_b),
        "diameter_on_both_positive": {
            "studies": len(both_positive),
            "agreement_within_cm": DIAMETER_AGREEMENT_CM,
            "agreement": (
                float(np.mean([_same_diameter(x, y) for x, y in zip(*diameter_pairs)]))
                if both_positive
                else None
            ),
        },
        "attenuation_on_both_positive": {
            **_agreement(attenuation_a, attenuation_b),
            "cohen_kappa": _kappa(attenuation_a, attenuation_b),
        },
        "annotators": sorted(
            {str(a.loc[i, "reviewer_id"]).strip() for i in ids}
            | {str(b.loc[i, "reviewer_id"]).strip() for i in ids}
        ),
    }

    rows = []
    for i in ids:
        va, vb = parsed_a[i], parsed_b[i]
        full_agreement = (
            va is not None
            and vb is not None
            and va.has_lesion == vb.has_lesion
            and va.lesion_count == vb.lesion_count
            and va.location_segments == vb.location_segments
            and _same_diameter(va.diameter_cm, vb.diameter_cm)
            and va.attenuation == vb.attenuation
        )
        row: dict[str, Any] = {ID: i, TEXT: a.loc[i, TEXT] if TEXT in a.columns else ""}
        for label, sheet in (("A", a), ("B", b)):
            for column in (*VALUE_COLUMNS, "evidence_quote", "review_notes", "reviewer_id"):
                row[f"{label}_{column}"] = sheet.loc[i, column]
        row["needs_clinician"] = "NO" if full_agreement else "YES"
        reviewers = f"{a.loc[i, 'reviewer_id'].strip()};{b.loc[i, 'reviewer_id'].strip()}"
        row["annotator_ids"] = reviewers
        if full_agreement:
            row.update(
                {
                    "review_method": "consensus",
                    "review_presence": "YES" if va.has_lesion else "NO",
                    "reviewed_count": _value_text(va, "count"),
                    "reviewed_location": _value_text(va, "location"),
                    "reviewed_diameter_cm": _value_text(va, "diameter"),
                    "reviewed_attenuation": _value_text(va, "attenuation"),
                    "evidence_quote": a.loc[i, "evidence_quote"] or b.loc[i, "evidence_quote"],
                    "review_notes": "Two independent annotations agree on every field.",
                    "reviewer_id": f"consensus:{reviewers}",
                    "reviewed_at": max(
                        _parse_reviewed_at(a.loc[i, "reviewed_at"]),
                        _parse_reviewed_at(b.loc[i, "reviewed_at"]),
                    ).isoformat(),
                }
            )
        else:
            row.update(
                {
                    "review_method": "clinician_adjudication",
                    **{column: "" for column in (*VALUE_COLUMNS, "evidence_quote", "review_notes", "reviewer_id", "reviewed_at")},
                }
            )
        rows.append(row)

    agreement["needs_clinician"] = sum(row["needs_clinician"] == "YES" for row in rows)
    adjudication_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(adjudication_path, index=False, encoding="utf-8-sig")
    agreement_path.parent.mkdir(parents=True, exist_ok=True)
    agreement_path.write_text(json.dumps(agreement, indent=2) + "\n", encoding="utf-8")
    return agreement


def _review_group(ground_truth: dict[str, Any]) -> str:
    labels = ground_truth["ground_truth"]
    if not labels["has_lesion"]:
        return "NO_LESION"
    largest = labels.get("largest_lesion") or {}
    return "LESION_WITH_LOCATION" if largest.get("location_segments") else "LESION_MISSING_LOCATION"


def finalize(adjudication_path: Path, gt_path: Path, review_csv_path: Path) -> pd.DataFrame:
    """Convert the completed adjudication sheet into the adjudication review CSV.

    Rows a clinician has not completed keep blank provenance, so
    ``adjudicate_cohort`` marks them ``needs_adjudication`` instead of promoting them.
    """
    _refuse_overwrite(review_csv_path)
    sheet = pd.read_csv(adjudication_path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    with gt_path.open(encoding="utf-8") as handle:
        ground_truth = {record["id"]: record for record in map(json.loads, handle)}
    unknown = sorted(set(sheet[ID]) - set(ground_truth))
    if unknown:
        raise ValueError(f"Adjudication sheet has IDs absent from Ground Truth: {unknown[:5]}")

    review = pd.DataFrame(
        {
            ID: sheet[ID],
            "review_group": [_review_group(ground_truth[i]) for i in sheet[ID]],
            **{column: sheet[column] for column in VALUE_COLUMNS},
            # The comparison with the source label is computed per field by
            # adjudicate_cohort; this column only records that it was not judged here.
            "targets_match_report": "UNCERTAIN",
            "review_notes": sheet["review_notes"],
            "reviewer_id": sheet["reviewer_id"],
            "reviewed_at": sheet["reviewed_at"],
            "review_method": sheet["review_method"],
            "annotator_ids": sheet["annotator_ids"],
            "evidence_quote": sheet["evidence_quote"],
        }
    )
    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_csv_path, index=False, encoding="utf-8-sig")
    return review
