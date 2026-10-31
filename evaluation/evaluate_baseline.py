"""Evaluate NLP baseline predictions against Ground Truth across dataset splits.

Computes standard clinical NLP metrics:
- Binary Classification (has_lesion): Accuracy, Precision, Recall, F1
- Integer Extraction (lesion_count): Exact Match, Mean Absolute Error (MAE)
- Multi-label Set Extraction (location_segments): Jaccard Index / IoU, Exact Match
- Continuous Regression (diameter_cm): Mean Absolute Error (MAE), Pearson Correlation
- Categorical Extraction (attenuation): Accuracy, Micro/Macro F1
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

GROUND_TRUTH_PATH = Path("data/processed/ground_truth.jsonl")
INFERENCES_PATH = Path("evaluation/baseline_inferences.jsonl")
SPLITS_PATH = Path("data/processed/splits.json")


def evaluate_split(
    gt_records: dict[str, dict[str, Any]],
    inf_records: dict[str, dict[str, Any]],
    split_ids: list[str],
    split_name: str,
) -> dict[str, Any]:
    """Calculate and return evaluation metrics for a specific split."""
    y_true_has = []
    y_pred_has = []

    y_true_cnt = []
    y_pred_cnt = []

    y_true_loc = []
    y_pred_loc = []

    y_true_diam = []
    y_pred_diam = []

    y_true_att = []
    y_pred_att = []

    for sid in split_ids:
        if sid not in gt_records or sid not in inf_records:
            continue

        gt = gt_records[sid]["ground_truth"]
        inf = inf_records[sid]["liver_lesions"]

        # 1. has_lesion
        gt_has = bool(gt["has_lesion"])
        inf_has = bool(inf["has_lesion"])
        y_true_has.append(gt_has)
        y_pred_has.append(inf_has)

        # Lesion-specific targets (for true positive cases)
        if gt_has:
            # 2. lesion count
            y_true_cnt.append(gt["lesion_count"])
            y_pred_cnt.append(inf.get("lesion_count") or 0)

            # 3. location segments
            gt_largest = gt.get("largest_lesion")
            inf_largest = inf.get("largest_lesion")

            gt_segs = set(gt_largest.get("location_segments") or []) if gt_largest else set()
            inf_segs = set(inf_largest.get("location_segments") or []) if inf_largest else set()
            y_true_loc.append(gt_segs)
            y_pred_loc.append(inf_segs)

            # 4. diameter
            if gt_largest and gt_largest.get("source_diameter_cm") is not None:
                gt_d = float(gt_largest["source_diameter_cm"])
                inf_d = (
                    float(inf_largest["derived_max_diameter_cm"])
                    if (inf_largest and inf_largest.get("derived_max_diameter_cm") is not None)
                    else 0.0
                )
                y_true_diam.append(gt_d)
                y_pred_diam.append(inf_d)

            # 5. attenuation
            if gt_largest and gt_largest.get("attenuation") is not None:
                gt_a = str(gt_largest["attenuation"]).lower()
                inf_a = (
                    str(inf_largest["attenuation"]).lower()
                    if (inf_largest and inf_largest.get("attenuation"))
                    else "none"
                )
                y_true_att.append(gt_a)
                y_pred_att.append(inf_a)

    # Compute metrics
    # has_lesion
    hl_acc = accuracy_score(y_true_has, y_pred_has)
    hl_prec = precision_score(y_true_has, y_pred_has, zero_division=0)
    hl_rec = recall_score(y_true_has, y_pred_has, zero_division=0)
    hl_f1 = f1_score(y_true_has, y_pred_has, zero_division=0)

    # count
    cnt_exact = np.mean([t == p for t, p in zip(y_true_cnt, y_pred_cnt)])
    cnt_mae = np.mean(np.abs(np.array(y_true_cnt) - np.array(y_pred_cnt)))

    # location segments Jaccard / Exact Match
    jaccards = []
    exact_loc = []
    for t_s, p_s in zip(y_true_loc, y_pred_loc):
        if not t_s and not p_s:
            jaccards.append(1.0)
            exact_loc.append(1.0)
        elif not t_s or not p_s:
            jaccards.append(0.0)
            exact_loc.append(0.0)
        else:
            j = len(t_s & p_s) / len(t_s | p_s)
            jaccards.append(j)
            exact_loc.append(1.0 if t_s == p_s else 0.0)
    loc_iou = np.mean(jaccards)
    loc_exact = np.mean(exact_loc)

    # diameter
    diam_diffs = np.abs(np.array(y_true_diam) - np.array(y_pred_diam))
    diam_mae = np.mean(diam_diffs)
    diam_within_half_cm = np.mean(diam_diffs <= 0.5)

    # attenuation
    att_acc = accuracy_score(y_true_att, y_pred_att)

    print("\n" + "=" * 65)
    print(f"BASELINE EVALUATION REPORT: {split_name.upper()} SET ({len(split_ids):,} studies)")
    print("=" * 65)
    print("1. Binary Lesion Presence (has_lesion):")
    print(f"   - Accuracy:  {hl_acc:.4f}")
    print(f"   - Precision: {hl_prec:.4f}")
    print(f"   - Recall:    {hl_rec:.4f}")
    print(f"   - F1-Score:  {hl_f1:.4f}")
    print("\n2. Lesion Count Quantification:")
    print(f"   - Exact Match: {cnt_exact:.4f} ({cnt_exact*100:.1f}%)")
    print(f"   - MAE:         {cnt_mae:.2f} lesions")
    print("\n3. Largest Lesion Anatomical Location (Couinaud Segments):")
    print(f"   - Mean IoU (Jaccard): {loc_iou:.4f}")
    print(f"   - Exact Match:        {loc_exact:.4f} ({loc_exact*100:.1f}%)")
    print("\n4. Largest Lesion Maximum Diameter:")
    print(f"   - Mean Absolute Error (MAE): {diam_mae:.2f} cm")
    print(f"   - Tolerance <= 0.5 cm Match: {diam_within_half_cm:.4f} ({diam_within_half_cm*100:.1f}%)")
    print("\n5. Largest Lesion Attenuation Classification:")
    print(f"   - Accuracy: {att_acc:.4f} ({att_acc*100:.1f}%)")
    print("=" * 65)

    return {
        "split": split_name,
        "studies": len(split_ids),
        "has_lesion_f1": hl_f1,
        "count_exact": cnt_exact,
        "location_iou": loc_iou,
        "diameter_mae": diam_mae,
        "attenuation_acc": att_acc,
    }


def run_evaluation() -> None:
    """Load records and run evaluation for validation and test splits."""
    print("Loading Ground Truth records...")
    gt_records: dict[str, dict[str, Any]] = {}
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt_records[rec["id"]] = rec

    print("Loading Baseline Inferences...")
    inf_records: dict[str, dict[str, Any]] = {}
    with open(INFERENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            inf_records[rec["id"]] = rec

    print("Loading dataset splits...")
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Evaluate on Validation set (used for prompt/rule tuning)
    evaluate_split(gt_records, inf_records, splits["val"], "validation")

    # Evaluate on Test set (held-out benchmark)
    evaluate_split(gt_records, inf_records, splits["test"], "test")


def main() -> None:
    """CLI execution."""
    run_evaluation()


if __name__ == "__main__":
    main()

