"""Automated Target Consistency Audit Pipeline.

Compares ground truth labels against cleaned narrative reports across all 9,262 records.
Categorizes field-level and study-level agreement into:
- MATCH: Clear text concordance with the source label.
- MISMATCH: Apparent contradiction or conflicting measurement.
- UNCERTAIN: Wording is unquantified, multi-focal, or ambiguous.

Produces:
- data/processed/nlp/target_consistency_audit.csv
- Structured statistical audit report for clinical review.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd
from liversira_ai.nlp.parser import parse_liver_report, split_sentences

MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
GT_PATH = Path("data/processed/nlp/ground_truth.jsonl")
OUTPUT_CSV = Path("data/processed/nlp/target_consistency_audit.csv")

def audit_single_record(master: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """Perform rule-based consistency check for a single record."""
    rec_id: str = master["id"]
    text: str | None = master.get("reports", {}).get("narrative_clean")
    gt_lbl: dict[str, Any] = gt["ground_truth"]

    if not text:
        return {
            "id": rec_id,
            "has_lesion_gt": gt_lbl["has_lesion"],
            "has_lesion_status": "UNCERTAIN",
            "count_gt": gt_lbl["lesion_count"],
            "count_text": None,
            "count_status": "UNCERTAIN",
            "location_gt": None,
            "location_text": None,
            "location_status": "UNCERTAIN",
            "diameter_gt": None,
            "diameter_text": None,
            "diameter_status": "UNCERTAIN",
            "attenuation_gt": None,
            "attenuation_text": None,
            "attenuation_status": "UNCERTAIN",
            "overall_status": "UNCERTAIN",
            "discrepancy_details": "Missing narrative report",
        }

    has_l: bool = gt_lbl["has_lesion"]
    l_cnt: int = gt_lbl["lesion_count"]
    largest: dict[str, Any] | None = gt_lbl.get("largest_lesion")

    parsed = parse_liver_report(text)
    liver_text = parsed.liver_text

    # 1. has_lesion check
    has_mention = parsed.presence

    if not has_l:
        # Expected negative
        if has_mention is False:
            hl_status = "MATCH"
            cnt_status = "MATCH"
            loc_status = "MATCH"
            diam_status = "MATCH"
            att_status = "MATCH"
            disc = ""
        elif has_mention is None:
            hl_status = "UNCERTAIN"
            cnt_status = "UNCERTAIN"
            loc_status = "UNCERTAIN"
            diam_status = "UNCERTAIN"
            att_status = "UNCERTAIN"
            disc = "No definitive lesion presence conclusion found in text"
        else:
            hl_status = "MISMATCH"
            cnt_status = "MISMATCH"
            loc_status = "MATCH"
            diam_status = "MATCH"
            att_status = "MATCH"
            disc = "GT has 0 lesions, but affirmative liver lesion found in text"

        overall = "MATCH" if hl_status == "MATCH" else "MISMATCH"
        return {
            "id": rec_id,
            "has_lesion_gt": False,
            "has_lesion_status": hl_status,
            "count_gt": 0,
            "count_text": 0 if has_mention is False else None,
            "count_status": cnt_status,
            "location_gt": None,
            "location_text": None,
            "location_status": loc_status,
            "diameter_gt": None,
            "diameter_text": None,
            "diameter_status": diam_status,
            "attenuation_gt": None,
            "attenuation_text": None,
            "attenuation_status": att_status,
            "overall_status": overall,
            "discrepancy_details": disc,
        }

    # has_lesion == True
    discrepancies: list[str] = []
    if has_mention is True:
        hl_status = "MATCH"
    elif has_mention is False:
        hl_status = "MISMATCH"
    else:
        hl_status = "UNCERTAIN"
    if hl_status == "MISMATCH":
        discrepancies.append(
            "Lesions in GT but no affirmative liver lesion found in text"
        )
    elif hl_status == "UNCERTAIN":
        discrepancies.append(
            "Lesion presence is ambiguous or not definitively stated in text"
        )

    # 2. Count check
    found_cnt = parsed.lesion_count

    if found_cnt is not None:
        if found_cnt == l_cnt:
            cnt_status = "MATCH"
        else:
            cnt_status = "MISMATCH"
            discrepancies.append(f"Count mismatch: GT={l_cnt} vs text={found_cnt}")
    else:
        cnt_status = "UNCERTAIN"
        discrepancies.append(f"Count unquantified in text (GT={l_cnt})")

    # 3. Location segments check
    gt_segs = (
        set(largest["location_segments"])
        if (largest and largest.get("location_segments"))
        else None
    )
    gt_loc_str = ", ".join(str(s) for s in sorted(gt_segs)) if gt_segs else None

    text_segs = set(parsed.location_segments) if parsed.location_segments else None
    text_loc_str = (
        ", ".join(str(s) for s in sorted(text_segs)) if text_segs else None
    )

    if gt_segs is None:
        if text_segs is None:
            loc_status = "MATCH"
        else:
            loc_status = "MISMATCH"
            discrepancies.append(
                f"Location missing in GT but text mentions segment(s) [{text_loc_str}]"
            )
    else:
        if text_segs is not None:
            if text_segs == gt_segs:
                loc_status = "MATCH"
            else:
                loc_status = "MISMATCH"
                discrepancies.append(
                    f"Location mismatch: GT=[{gt_loc_str}] vs text=[{text_loc_str}]"
                )
        else:
            loc_status = "UNCERTAIN"
            discrepancies.append(
                f"Location in GT [{gt_loc_str}] not parsed deterministically from text"
            )

    # 4. Diameter check
    gt_diam = largest.get("source_diameter_cm") if largest else None
    text_diam = parsed.derived_max_diameter_cm
    if text_diam is not None and gt_diam is not None:
        if abs(text_diam - gt_diam) <= 0.05:
            diam_status = "MATCH"
        else:
            diam_status = "MISMATCH"
            discrepancies.append(
                f"Diameter mismatch: GT={gt_diam} cm vs text={text_diam} cm"
            )
    else:
        diam_status = "UNCERTAIN"
        discrepancies.append(
            f"Diameter in GT ({gt_diam} cm) not extracted deterministically"
        )

    # 5. Attenuation check
    gt_att = largest.get("attenuation") if largest else None
    gt_att_str = str(gt_att).lower() if gt_att else None

    text_att = parsed.attenuation
    att_in_liver = [
        t
        for t in ["hypoattenuating", "hyperattenuating", "isoattenuating"]
        if t in liver_text.lower()
    ]

    if text_att and gt_att_str:
        if text_att == gt_att_str:
            if len(set(att_in_liver)) > 1:
                att_status = "UNCERTAIN"
                discrepancies.append(
                    f"Attenuation matches ({text_att}) but multiple descriptors in liver context: {att_in_liver}"
                )
            else:
                att_status = "MATCH"
        else:
            att_status = "MISMATCH"
            discrepancies.append(f"Attenuation mismatch: GT={gt_att_str} vs text={text_att}")
    else:
        att_status = "UNCERTAIN"
        discrepancies.append(
            f"Attenuation not extracted deterministically (GT={gt_att_str})"
        )

    # Study-level status
    statuses = [hl_status, cnt_status, loc_status, diam_status, att_status]
    if "MISMATCH" in statuses:
        overall = "MISMATCH"
    elif "UNCERTAIN" in statuses:
        overall = "UNCERTAIN"
    else:
        overall = "MATCH"

    return {
        "id": rec_id,
        "has_lesion_gt": True,
        "has_lesion_status": hl_status,
        "count_gt": l_cnt,
        "count_text": found_cnt,
        "count_status": cnt_status,
        "location_gt": gt_loc_str,
        "location_text": text_loc_str,
        "location_status": loc_status,
        "diameter_gt": gt_diam,
        "diameter_text": text_diam,
        "diameter_status": diam_status,
        "attenuation_gt": gt_att_str,
        "attenuation_text": text_att,
        "attenuation_status": att_status,
        "overall_status": overall,
        "discrepancy_details": "; ".join(discrepancies),
    }


def run_consistency_audit(
    master_path: Path, gt_path: Path, output_csv: Path
) -> pd.DataFrame:
    """Execute consistency audit across all paired records and output report."""
    print("=" * 65)
    print("STARTING AUTOMATED TARGET CONSISTENCY AUDIT")
    print("=" * 65)
    print(f"Reading Master Records: {master_path}")
    print(f"Reading Ground Truth:   {gt_path}")

    records: list[dict[str, Any]] = []
    with open(master_path, encoding="utf-8") as f_m, open(
        gt_path, encoding="utf-8"
    ) as f_gt:
        for idx, (line_m, line_gt) in enumerate(zip(f_m, f_gt), start=1):
            master_rec = json.loads(line_m)
            gt_rec = json.loads(line_gt)
            records.append(audit_single_record(master_rec, gt_rec))

            if idx % 2000 == 0:
                print(f"Audited {idx:,} / 9,262 records...")

    df_audit = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Calculate statistics
    total = len(df_audit)
    c_overall = Counter(df_audit["overall_status"])
    c_hl = Counter(df_audit["has_lesion_status"])
    c_cnt = Counter(df_audit["count_status"])
    c_loc = Counter(df_audit["location_status"])
    c_diam = Counter(df_audit["diameter_status"])
    c_att = Counter(df_audit["attenuation_status"])

    print("\n" + "=" * 65)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Studies Audited: {total:,}\n")

    print(
        f"{'Target Field':<22} | {'MATCH (%)':<15} | {'MISMATCH (%)':<15} | {'UNCERTAIN (%)':<15}"
    )
    print("-" * 65)

    fields_counters = [
        ("has_lesion", c_hl),
        ("lesion_count", c_cnt),
        ("location_segments", c_loc),
        ("source_diameter_cm", c_diam),
        ("attenuation", c_att),
    ]

    for name, c in fields_counters:
        m = c.get("MATCH", 0)
        mm = c.get("MISMATCH", 0)
        u = c.get("UNCERTAIN", 0)
        print(
            f"{name:<22} | {m:>5} ({m/total*100:>5.1f}%) | {mm:>5} ({mm/total*100:>5.1f}%) | {u:>5} ({u/total*100:>5.1f}%)"
        )

    print("-" * 65)
    om = c_overall.get("MATCH", 0)
    omm = c_overall.get("MISMATCH", 0)
    ou = c_overall.get("UNCERTAIN", 0)
    print(
        f"{'OVERALL STUDY STATUS':<22} | {om:>5} ({om/total*100:>5.1f}%) | {omm:>5} ({omm/total*100:>5.1f}%) | {ou:>5} ({ou/total*100:>5.1f}%)"
    )
    print("=" * 65)

    print(f"\n[DONE] Detailed audit log saved to: {output_csv}")
    return df_audit


def main() -> None:
    """CLI Entry Point."""
    run_consistency_audit(MASTER_PATH, GT_PATH, OUTPUT_CSV)


if __name__ == "__main__":
    main()
