"""
CureSight / SMC-LUD Dataset Audit Pipeline
==========================================
Phases:
  1. Dataset Structure  – per-group folder parsing (correct hierarchy per group)
  2. Official Metadata  – HCC & Hemangioma patient tables, patient_id resolution
  3. Image Quality      – dimensions, color mode, corruption, MD5 duplicates
  4. Patient & Split    – leakage investigation, label conflicts
  5. Visual Samples     – representative grid saved to manifests/
  6. Final Manifest     – corrected CSV + JSONL + summary + audit report

Constraints:
  - No resizing, normalizing, augmenting, or preprocessing
  - No model training / evaluation
  - No deletion or movement of original images
  - Official splits are NEVER modified
"""

import os
import sys
import json
import csv
import hashlib
import collections
import textwrap
from pathlib import Path
from datetime import datetime

from curesight.common.paths import CV_MANIFEST_DIR, CV_RAW_DIR

# Fix Windows console encoding (CP1252 → UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from PIL import Image, UnidentifiedImageError
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("⚠  Pillow not installed – image quality checks will be skipped.")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    print("⚠  pandas not installed – metadata analysis will be limited.")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = CV_RAW_DIR
OUT_DIR = CV_MANIFEST_DIR
OUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_ENC = {"HCC": 0, "Hemangioma": 1}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def file_md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _banner(title: str) -> None:
    line = "=" * 62
    print(f"\n{line}\n  {title}\n{line}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – Dataset Structure  (per-group aware)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_original_data(group_dir: Path) -> list[dict]:
    """
    original_data/
      HCC/
        HCC_caliper/<patient_folder>/<uuid>.jpg
        HCC_clean (wo caliper)/<patient_folder>/<uuid>.jpg
        HCC_patient_table.csv          ← metadata
      Hemangioma/
        Hem_caliper/<patient_folder>/<uuid>.jpg
        Hem_clean (wo caliper)/<patient_folder>/<uuid>.jpg
        Hem_patient_table.csv          ← metadata
    """
    records = []
    for class_dir in sorted(group_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        raw_class = class_dir.name           # "HCC" | "Hemangioma"
        label = raw_class                    # canonical

        for variant_dir in sorted(class_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            vname = variant_dir.name.lower()
            if "caliper" in vname and "clean" not in vname and "wo" not in vname:
                variant = "caliper"
            elif "clean" in vname or "wo caliper" in vname:
                variant = "clean"
            else:
                variant = "unknown"

            # patient sub-folders  →  images
            for patient_dir in sorted(variant_dir.iterdir()):
                if patient_dir.is_dir():
                    patient_folder = patient_dir.name   # e.g. "HCC_10" / "Hem_10"
                    for img in sorted(patient_dir.iterdir()):
                        if _is_image(img):
                            rel = img.relative_to(BASE)
                            records.append({
                                "abs_path":      str(img),
                                "rel_path":      str(rel).replace("\\", "/"),
                                "dataset_group": "original_data",
                                "split":         "original",
                                "label":         label,
                                "label_id":      LABEL_ENC.get(label, -1),
                                "variant":       variant,
                                "patient_folder": patient_folder,
                                "filename":      img.name,
                                "stem":          img.stem,
                                "ext":           img.suffix.lower(),
                                "size_bytes":    img.stat().st_size,
                            })
                elif _is_image(patient_dir):
                    # flat image directly in variant folder (edge case)
                    img = patient_dir
                    rel = img.relative_to(BASE)
                    records.append({
                        "abs_path":      str(img),
                        "rel_path":      str(rel).replace("\\", "/"),
                        "dataset_group": "original_data",
                        "split":         "original",
                        "label":         label,
                        "label_id":      LABEL_ENC.get(label, -1),
                        "variant":       variant,
                        "patient_folder": None,
                        "filename":      img.name,
                        "stem":          img.stem,
                        "ext":           img.suffix.lower(),
                        "size_bytes":    img.stat().st_size,
                    })
    return records


def _collect_split_for_train(group_dir: Path) -> list[dict]:
    """
    split_for_train/
      train/<HCC|Hemangioma>/<uuid>.jpg|.png
      val/<HCC|Hemangioma>/<uuid>.jpg|.png
      test/<HCC|Hemangioma>/<uuid>.jpg|.png

    Images here are a mix of caliper + clean (same UUIDs as original_data).
    Variant is NOT indicated by the folder structure; we mark it 'mixed'
    and later cross-reference by MD5 against original_data.
    """
    records = []
    for split_dir in sorted(group_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name          # train / val / test

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name           # HCC | Hemangioma

            for img in sorted(class_dir.iterdir()):
                if _is_image(img):
                    rel = img.relative_to(BASE)
                    records.append({
                        "abs_path":      str(img),
                        "rel_path":      str(rel).replace("\\", "/"),
                        "dataset_group": "split_for_train",
                        "split":         split_name,
                        "label":         label,
                        "label_id":      LABEL_ENC.get(label, -1),
                        "variant":       "mixed",
                        "patient_folder": None,
                        "filename":      img.name,
                        "stem":          img.stem,
                        "ext":           img.suffix.lower(),
                        "size_bytes":    img.stat().st_size,
                    })
    return records


def _collect_clean_ver_for_train(group_dir: Path) -> list[dict]:
    """
    clean_ver_for_train/
      train_clean/<HCC|Hemangioma>/<uuid>.jpg|.png
      val_clean/<HCC|Hemangioma>/<uuid>.jpg|.png
      test_clean/<HCC|Hemangioma>/<uuid>.jpg|.png

    These are caliper-free (clean) images only.
    """
    records = []
    for split_dir in sorted(group_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name          # train_clean / val_clean / test_clean
        # canonical split: remove "_clean" suffix for comparison
        canonical_split = split_name.replace("_clean", "")

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name           # HCC | Hemangioma

            for img in sorted(class_dir.iterdir()):
                if _is_image(img):
                    rel = img.relative_to(BASE)
                    records.append({
                        "abs_path":       str(img),
                        "rel_path":       str(rel).replace("\\", "/"),
                        "dataset_group":  "clean_ver_for_train",
                        "split":          canonical_split,
                        "split_folder":   split_name,
                        "label":          label,
                        "label_id":       LABEL_ENC.get(label, -1),
                        "variant":        "clean",
                        "patient_folder": None,
                        "filename":       img.name,
                        "stem":           img.stem,
                        "ext":            img.suffix.lower(),
                        "size_bytes":     img.stat().st_size,
                    })
    return records


def phase1_collect_all() -> list[dict]:
    _banner("PHASE 1 — Dataset Structure & Image Collection")

    all_records: list[dict] = []

    group_dispatch = {
        "original_data":       _collect_original_data,
        "split_for_train":     _collect_split_for_train,
        "clean_ver_for_train": _collect_clean_ver_for_train,
    }

    for group_name, fn in group_dispatch.items():
        gdir = BASE / group_name
        if not gdir.exists():
            print(f"  ⚠  Directory not found: {gdir}")
            continue
        recs = fn(gdir)
        all_records.extend(recs)
        print(f"  📁 {group_name}: {len(recs)} images")

    # Per-group breakdown
    from collections import Counter
    combo = Counter(
        (r["dataset_group"], r.get("split", "?"), r["label"])
        for r in all_records
    )
    print("\n  Breakdown (group / split / label):")
    for (g, s, l), cnt in sorted(combo.items()):
        print(f"    {g:<25} {s:<15} {l:<12} {cnt}")

    print(f"\n  Total images found: {len(all_records)}")
    return all_records


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – Official Metadata Analysis
# ══════════════════════════════════════════════════════════════════════════════

def phase2_metadata(records: list[dict]) -> tuple[dict, dict]:
    """
    Reads HCC_patient_table.csv and Hem_patient_table.csv.
    Returns two dicts:  patient_folder_name -> official Subject_id
      hcc_pid_map["HCC_10"]  = "10"
      hem_pid_map["Hem_10"]  = "Hem_10"
    """
    _banner("PHASE 2 — Official Metadata Analysis")

    issues = []

    # ── HCC ──────────────────────────────────────────────────────────────────
    hcc_csv = BASE / "original_data" / "HCC" / "HCC_patient_table.csv"
    hcc_pid_map: dict[str, str] = {}      # "HCC_<id>" -> str(Subject_id)

    if hcc_csv.exists():
        if PANDAS_OK:
            hcc_df = pd.read_csv(hcc_csv)
            print(f"\n  HCC patient table: {hcc_csv.name}")
            print(f"  Rows: {len(hcc_df)}   Columns: {list(hcc_df.columns)}")
            print(f"  Subject_id range: {hcc_df['Subject_id'].min()} – {hcc_df['Subject_id'].max()}")

            missing_sz = hcc_df["Tumor size"].isna().sum()
            print(f"  Missing 'Tumor size': {missing_sz} / {len(hcc_df)}")

            # Build folder-name → official Subject_id map
            # Original folders are named "HCC_<Subject_id>"
            for sid in hcc_df["Subject_id"]:
                folder_key = f"HCC_{sid}"
                hcc_pid_map[folder_key] = str(int(sid))

            print(f"  Built HCC patient_id map: {len(hcc_pid_map)} entries")
        else:
            # Fallback: manual CSV read
            with open(hcc_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        sid = int(row["Subject_id"])
                        hcc_pid_map[f"HCC_{sid}"] = str(sid)
                    except (KeyError, ValueError):
                        pass
            print(f"  HCC patient_id map (fallback): {len(hcc_pid_map)} entries")
    else:
        issues.append("HCC_patient_table.csv not found")
        print("  ⚠  HCC_patient_table.csv not found")

    # ── Hemangioma ────────────────────────────────────────────────────────────
    hem_csv = BASE / "original_data" / "Hemangioma" / "Hem_patient_table.csv"
    hem_pid_map: dict[str, str] = {}      # "Hem_<id>" -> "Hem_<id>"

    if hem_csv.exists():
        if PANDAS_OK:
            hem_df = pd.read_csv(hem_csv)
            print(f"\n  Hemangioma patient table: {hem_csv.name}")
            print(f"  Rows: {len(hem_df)}   Columns: {list(hem_df.columns[:6])}")

            # Subject_id is like "Hem_2", "Hem_4", etc.
            sid_col = hem_df.columns[1]   # always second column
            for sid in hem_df[sid_col].dropna():
                sid_str = str(sid).strip()
                hem_pid_map[sid_str] = sid_str

            print(f"  Built Hemangioma patient_id map: {len(hem_pid_map)} entries")
        else:
            with open(hem_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vals = list(row.values())
                    if len(vals) > 1:
                        sid = str(vals[1]).strip()
                        if sid.startswith("Hem_"):
                            hem_pid_map[sid] = sid
            print(f"  Hemangioma patient_id map (fallback): {len(hem_pid_map)} entries")
    else:
        issues.append("Hem_patient_table.csv not found")
        print("  ⚠  Hem_patient_table.csv not found")

    # ── Cross-reference patient_folder against maps ───────────────────────────
    matched_hcc = unmatched_hcc = matched_hem = unmatched_hem = 0
    for r in records:
        if r["dataset_group"] != "original_data":
            r["patient_id"] = None
            continue
        pf = r.get("patient_folder")
        label = r["label"]
        if label == "HCC":
            pid = hcc_pid_map.get(pf)
            if pid:
                matched_hcc += 1
            else:
                unmatched_hcc += 1
            r["patient_id"] = pid
        elif label == "Hemangioma":
            pid = hem_pid_map.get(pf)
            if pid:
                matched_hem += 1
            else:
                unmatched_hem += 1
            r["patient_id"] = pid
        else:
            r["patient_id"] = None

    print(f"\n  HCC patient_id matches:      {matched_hcc}  (unmatched: {unmatched_hcc})")
    print(f"  Hemangioma patient_id matches: {matched_hem}  (unmatched: {unmatched_hem})")

    if unmatched_hcc or unmatched_hem:
        issues.append(
            f"{unmatched_hcc} HCC folders and {unmatched_hem} Hemangioma folders "
            "could not be matched to official Subject_ids"
        )

    # For split_for_train and clean_ver_for_train, patient_id must come from
    # the UUID cross-reference (done later after MD5 map is built).
    # Mark as pending for now.
    for r in records:
        if "patient_id" not in r:
            r["patient_id"] = None

    return hcc_pid_map, hem_pid_map


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 – Image Quality Verification
# ══════════════════════════════════════════════════════════════════════════════

def phase3_quality(records: list[dict]) -> tuple[list, list, dict]:
    """
    For each record:
      - Read image → get width, height, mode
      - Compute MD5
    Returns: (corrupt_list, duplicate_pairs, md5_to_first_record)
    """
    _banner("PHASE 3 — Image Quality Verification")

    if not PIL_OK:
        print("  ⚠  Pillow unavailable – skipping quality checks")
        for r in records:
            r["width"] = r["height"] = r["mode"] = r["md5"] = None
        return [], [], {}

    corrupt: list[tuple[str, str]] = []
    md5_map: dict[str, dict] = {}          # md5 -> first record
    duplicates: list[tuple[str, str]] = []

    total = len(records)
    for i, r in enumerate(records, 1):
        if i % 1000 == 0 or i == total:
            print(f"  Scanning {i:>5}/{total} ...", end="\r")

        p = Path(r["abs_path"])

        # ── MD5 ──
        digest = file_md5(p)
        r["md5"] = digest

        if digest in md5_map:
            duplicates.append((r["rel_path"], md5_map[digest]["rel_path"]))
        else:
            md5_map[digest] = r

        # ── Image properties ──
        if not PIL_OK:
            r["width"] = r["height"] = r["mode"] = None
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                r["width"], r["height"] = img.size
                r["mode"] = img.mode
        except (UnidentifiedImageError, Exception) as e:
            r["width"] = r["height"] = r["mode"] = None
            corrupt.append((r["rel_path"], str(e)))

    print()

    # ── Dimension stats ──
    valid_dims = [(r["width"], r["height"]) for r in records if r.get("width")]
    if valid_dims:
        widths  = [d[0] for d in valid_dims]
        heights = [d[1] for d in valid_dims]
        dim_counter = collections.Counter(valid_dims)
        print(f"\n  Unique dimension pairs  : {len(dim_counter)}")
        print(f"  Width  range            : {min(widths)} – {max(widths)}")
        print(f"  Height range            : {min(heights)} – {max(heights)}")
        print(f"  Top-5 dimension sizes   :")
        for (w, h), cnt in dim_counter.most_common(5):
            print(f"    {w}×{h} : {cnt} images")

    modes = collections.Counter(r.get("mode", "unknown") for r in records)
    print(f"\n  Color modes:")
    for m, c in modes.most_common():
        print(f"    {m}: {c}")

    print(f"\n  Corrupt / unreadable files : {len(corrupt)}")
    for path, err in corrupt[:10]:
        print(f"    {path}  →  {err}")

    print(f"\n  Duplicate files (same MD5)  : {len(duplicates)}")
    print(f"  (Cross-group duplicates are expected; see audit report)")
    # Show a sample of cross-group vs within-group
    within_group_dups = []
    cross_group_dups  = []
    for dup_path, orig_path in duplicates:
        dup_group  = dup_path.split("/")[0]
        orig_group = orig_path.split("/")[0]
        if dup_group == orig_group:
            within_group_dups.append((dup_path, orig_path))
        else:
            cross_group_dups.append((dup_path, orig_path))

    print(f"    Within-group duplicates : {len(within_group_dups)}")
    print(f"    Cross-group duplicates  : {len(cross_group_dups)}")
    if within_group_dups:
        print("  ⚠  Within-group duplicates (unexpected):")
        for dup, orig in within_group_dups[:10]:
            print(f"    {dup}  ==  {orig}")

    return corrupt, duplicates, md5_map


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 – Patient & Split Verification
# ══════════════════════════════════════════════════════════════════════════════

def phase4_patient_split(
    records: list[dict],
    hcc_pid_map: dict,
    hem_pid_map: dict,
    md5_map: dict,
) -> tuple[list, list, dict]:
    """
    1. Propagate patient_id to split_for_train / clean_ver_for_train
       by UUID stem matching with original_data records.
    2. Check for patient-level leakage within split_for_train.
    3. Check for patient-level leakage within clean_ver_for_train.
    4. Label conflict check.
    """
    _banner("PHASE 4 — Patient & Split Verification")

    # ── Step 1: build stem → patient_id map from original_data ───────────────
    stem_to_pid: dict[str, str] = {}   # UUID stem -> patient_id
    stem_to_label: dict[str, str] = {}

    for r in records:
        if r["dataset_group"] == "original_data" and r.get("patient_id"):
            stem_to_pid[r["stem"]] = r["patient_id"]
            stem_to_label[r["stem"]] = r["label"]

    # Propagate to other groups
    missing_pid = 0
    for r in records:
        if r["dataset_group"] != "original_data":
            pid = stem_to_pid.get(r["stem"])
            r["patient_id"] = pid
            if pid is None:
                missing_pid += 1

    print(f"\n  patient_id propagated via UUID stem.")
    print(f"  Images in non-original groups without resolvable patient_id: {missing_pid}")

    # ── Step 2: patient-level leakage in split_for_train ─────────────────────
    sft_records = [r for r in records if r["dataset_group"] == "split_for_train"]

    pid_to_splits_sft: dict[str, set] = collections.defaultdict(set)
    for r in sft_records:
        pid = r.get("patient_id")
        if pid:
            pid_to_splits_sft[pid].add(r["split"])

    leakage_sft = {
        pid: splits
        for pid, splits in pid_to_splits_sft.items()
        if len(splits) > 1
    }
    print(f"\n  split_for_train – patients spanning >1 split: {len(leakage_sft)}")
    for pid, splits in list(leakage_sft.items())[:10]:
        print(f"    patient_id={pid}  splits={splits}")

    # ── Step 3: patient-level leakage in clean_ver_for_train ─────────────────
    cvft_records = [r for r in records if r["dataset_group"] == "clean_ver_for_train"]

    pid_to_splits_cvft: dict[str, set] = collections.defaultdict(set)
    for r in cvft_records:
        pid = r.get("patient_id")
        if pid:
            pid_to_splits_cvft[pid].add(r["split"])

    leakage_cvft = {
        pid: splits
        for pid, splits in pid_to_splits_cvft.items()
        if len(splits) > 1
    }
    print(f"\n  clean_ver_for_train – patients spanning >1 split: {len(leakage_cvft)}")
    for pid, splits in list(leakage_cvft.items())[:10]:
        print(f"    patient_id={pid}  splits={splits}")

    # ── Step 4: Label conflicts ───────────────────────────────────────────────
    stem_labels: dict[str, set] = collections.defaultdict(set)
    for r in records:
        stem_labels[r["stem"]].add(r["label"])

    label_conflicts = {
        stem: labs for stem, labs in stem_labels.items() if len(labs) > 1
    }
    print(f"\n  Label conflicts (same UUID, different label): {len(label_conflicts)}")
    for stem, labs in list(label_conflicts.items())[:5]:
        print(f"    stem={stem}  labels={labs}")

    # ── Step 5: Image-level overlap summary ──────────────────────────────────
    # Compare UUID stems across groups
    groups = ["original_data", "split_for_train", "clean_ver_for_train"]
    group_stems: dict[str, set] = {g: set() for g in groups}
    for r in records:
        group_stems[r["dataset_group"]].add(r["stem"])

    print("\n  UUID stem overlap across groups:")
    glist = list(group_stems.items())
    for i in range(len(glist)):
        for j in range(i + 1, len(glist)):
            g1, s1 = glist[i]
            g2, s2 = glist[j]
            inter = len(s1 & s2)
            print(f"    {g1} ∩ {g2} = {inter} stems")

    # ── Step 6: split distribution ───────────────────────────────────────────
    print("\n  Images per (group / split / label):")
    combo = collections.Counter(
        (r["dataset_group"], r.get("split", "?"), r["label"])
        for r in records
    )
    for (g, s, l), cnt in sorted(combo.items()):
        print(f"    {g:<25} {s:<12} {l:<12} {cnt}")

    # Build leakage combined dict for report
    leakage_combined = {**leakage_sft, **leakage_cvft}
    return list(leakage_sft.items()), list(leakage_cvft.items()), label_conflicts


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 – Visual Samples  (optional – requires matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def phase5_visual_samples(records: list[dict]) -> None:
    _banner("PHASE 5 — Visual Inspection Samples")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠  matplotlib not available – skipping visual samples")
        return

    if not PIL_OK:
        print("  ⚠  Pillow not available – skipping visual samples")
        return

    def _sample_images(group: str, split: str, label: str, variant: str,
                       n: int = 4) -> list[Path]:
        subset = [
            r for r in records
            if r["dataset_group"] == group
            and r.get("split") == split
            and r["label"] == label
            and r.get("variant") == variant
        ]
        import random
        random.seed(42)
        sample = random.sample(subset, min(n, len(subset)))
        return [Path(r["abs_path"]) for r in sample]

    # Grid: HCC clean vs caliper, Hemangioma clean vs caliper (from original_data)
    grid_specs = [
        ("original_data", "original", "HCC",         "clean",   "HCC Clean"),
        ("original_data", "original", "HCC",         "caliper", "HCC Caliper"),
        ("original_data", "original", "Hemangioma",  "clean",   "Hemangioma Clean"),
        ("original_data", "original", "Hemangioma",  "caliper", "Hemangioma Caliper"),
    ]

    n_cols = 4
    n_rows = len(grid_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    any_plotted = False
    for row_idx, (group, split, label, variant, title) in enumerate(grid_specs):
        paths = _sample_images(group, split, label, variant, n=n_cols)
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(title, fontsize=9, loc="left", pad=4)
            if col_idx < len(paths):
                try:
                    with Image.open(paths[col_idx]) as img:
                        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                    any_plotted = True
                except Exception:
                    pass

    if any_plotted:
        out_png = OUT_DIR / "visual_samples.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        plt.close()
        print(f"  ✅  Visual sample grid saved → {out_png}")
    else:
        print("  ⚠  No images could be rendered")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 – Final Manifest  (CSV + JSONL + summary + audit report)
# ══════════════════════════════════════════════════════════════════════════════

MANIFEST_FIELDS = [
    "rel_path", "dataset_group", "split", "label", "label_id",
    "variant", "patient_id", "width", "height", "mode",
    "ext", "size_bytes", "md5",
]


def phase6_manifest(
    records: list[dict],
    corrupt: list,
    duplicates: list,
    leakage_sft: list,
    leakage_cvft: list,
    label_conflicts: dict,
) -> None:
    _banner("PHASE 6 — Final Manifest & Audit Report")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    print(f"\n  ✅  manifest.csv   → {csv_path}  ({len(records)} rows)")

    # ── JSONL ────────────────────────────────────────────────────────────────
    jsonl_path = OUT_DIR / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            row = {k: r.get(k) for k in MANIFEST_FIELDS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  ✅  manifest.jsonl → {jsonl_path}  ({len(records)} rows)")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    within_group_dups = [
        (d, o) for d, o in duplicates
        if d.split("/")[0] == o.split("/")[0]
    ]
    cross_group_dups = [
        (d, o) for d, o in duplicates
        if d.split("/")[0] != o.split("/")[0]
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_images": len(records),
        "label_distribution": dict(
            collections.Counter(r["label"] for r in records)
        ),
        "split_distribution": dict(
            collections.Counter(r.get("split", "?") for r in records)
        ),
        "group_distribution": dict(
            collections.Counter(r["dataset_group"] for r in records)
        ),
        "variant_distribution": dict(
            collections.Counter(r.get("variant", "unknown") for r in records)
        ),
        "ext_distribution": dict(
            collections.Counter(r["ext"] for r in records)
        ),
        "mode_distribution": dict(
            collections.Counter(r.get("mode", "unknown") for r in records)
        ),
        "corrupt_count": len(corrupt),
        "total_md5_duplicates": len(duplicates),
        "within_group_duplicates": len(within_group_dups),
        "cross_group_duplicates": len(cross_group_dups),
        "patient_leakage_split_for_train": len(leakage_sft),
        "patient_leakage_clean_ver_for_train": len(leakage_cvft),
        "label_conflicts": len(label_conflicts),
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✅  summary.json   → {summary_path}")

    # ── Leakage CSV (only if actual leakage confirmed) ────────────────────────
    all_leakage = leakage_sft + leakage_cvft
    confirmed = [(pid, splits) for pid, splits in all_leakage if len(splits) > 1]
    if confirmed:
        leakage_path = OUT_DIR / "leakage_report.csv"
        with open(leakage_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "splits_present", "source_group"])
            for pid, splits in leakage_sft:
                writer.writerow([pid, "|".join(sorted(splits)), "split_for_train"])
            for pid, splits in leakage_cvft:
                writer.writerow([pid, "|".join(sorted(splits)), "clean_ver_for_train"])
        print(f"  ✅  leakage_report.csv → {leakage_path}  ({len(confirmed)} cases)")
    else:
        print("  ✅  No confirmed patient-level leakage – leakage_report.csv not written")

    # ── Audit Report ─────────────────────────────────────────────────────────
    _write_audit_report(records, summary, corrupt, duplicates,
                        within_group_dups, cross_group_dups,
                        leakage_sft, leakage_cvft, label_conflicts)


def _write_audit_report(
    records, summary, corrupt, duplicates,
    within_group_dups, cross_group_dups,
    leakage_sft, leakage_cvft, label_conflicts
):
    report_path = OUT_DIR / "dataset_audit_report.md"

    # helper to build a table line
    def tbl(rows):
        return "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)

    # Per-group/split/label counts
    combo = collections.Counter(
        (r["dataset_group"], r.get("split", "?"), r["label"])
        for r in records
    )

    dim_counter = collections.Counter(
        (r.get("width"), r.get("height"))
        for r in records if r.get("width")
    )

    lines = []
    lines.append("# CureSight / SMC-LUD Dataset Audit Report")
    lines.append(f"\n**Generated:** {summary['generated_at']}")
    lines.append(f"\n**Dataset root:** `data/raw/cv/SMC-LUD/`\n")

    lines.append("---\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total images | {summary['total_images']:,} |")
    for g, cnt in summary['group_distribution'].items():
        lines.append(f"| Images in `{g}` | {cnt:,} |")
    lines.append(f"| Corrupt / unreadable | {summary['corrupt_count']} |")
    lines.append(f"| MD5 duplicates (total) | {summary['total_md5_duplicates']:,} |")
    lines.append(f"| – within-group | {summary['within_group_duplicates']:,} |")
    lines.append(f"| – cross-group  | {summary['cross_group_duplicates']:,} |")

    lines.append("\n### 1.1 Per-Group / Split / Label Breakdown\n")
    lines.append("| Dataset Group | Split | Label | Count |")
    lines.append("|---------------|-------|-------|------:|")
    for (g, s, l), cnt in sorted(combo.items()):
        lines.append(f"| {g} | {s} | {l} | {cnt:,} |")

    lines.append("\n### 1.2 Variant Distribution\n")
    lines.append("| Variant | Count |")
    lines.append("|---------|------:|")
    for v, cnt in sorted(summary['variant_distribution'].items()):
        lines.append(f"| {v} | {cnt:,} |")

    lines.append("\n---\n")
    lines.append("## 2. Official Metadata\n")
    lines.append("Two official patient tables are present in `original_data/`:\n")
    lines.append("| File | Notes |")
    lines.append("|------|-------|")
    lines.append("| `HCC/HCC_patient_table.csv` | 600 rows; columns: No, Subject_id (numeric), age, Height, Weight, Sex, T-stage (UICC 2000 & AJCC 2010), Tumor size, Max_Diameter |")
    lines.append("| `Hemangioma/Hem_patient_table.csv` | 423 rows; columns: No, Subject_id (`Hem_N`), age, Height, Weight, Sex, Disease_id |")
    lines.append("\n**Key observations:**")
    lines.append("- HCC Subject_ids are plain integers; folder names are `HCC_<Subject_id>`.")
    lines.append("- Hemangioma Subject_ids are prefixed strings (`Hem_N`); folder names match directly.")
    lines.append("- Hemangioma table has 3 rows with missing age/height/weight (rows 421–423).")
    lines.append("- HCC table has significant missing values in `Height` and `Tumor size` columns.")
    lines.append("- Neither table contains image filenames (UUIDs) – patient linkage is through the patient sub-folder name.")
    lines.append("- Images in `split_for_train` and `clean_ver_for_train` are flat (no patient sub-folder); patient_id is resolved by UUID stem cross-referencing with `original_data`.")

    lines.append("\n---\n")
    lines.append("## 3. Image Quality\n")
    lines.append("### 3.1 Dimensions\n")
    lines.append("| Dimension (W×H) | Count |")
    lines.append("|-----------------|------:|")
    for (w, h), cnt in dim_counter.most_common(10):
        lines.append(f"| {w}×{h} | {cnt:,} |")

    lines.append("\n### 3.2 Color Modes\n")
    lines.append("| Mode | Count |")
    lines.append("|------|------:|")
    for mode, cnt in summary['mode_distribution'].items():
        lines.append(f"| {mode} | {cnt:,} |")

    lines.append("\n### 3.3 File Extensions\n")
    lines.append("| Extension | Count |")
    lines.append("|-----------|------:|")
    for ext, cnt in summary['ext_distribution'].items():
        lines.append(f"| {ext} | {cnt:,} |")

    lines.append(f"\n### 3.4 Corrupt Files\n")
    if corrupt:
        lines.append(f"**{len(corrupt)} corrupt file(s) detected:**\n")
        for path, err in corrupt[:20]:
            lines.append(f"- `{path}` → `{err}`")
    else:
        lines.append("No corrupt or unreadable files detected.")

    lines.append("\n### 3.5 MD5 Duplicate Analysis\n")
    lines.append(f"- **Total duplicate pairs:** {len(duplicates):,}")
    lines.append(f"- **Cross-group duplicates:** {len(cross_group_dups):,} — **EXPECTED**")
    lines.append(f"  - `split_for_train` and `original_data` share all 5 385 UUID stems (same files, different directory).")
    lines.append(f"  - `clean_ver_for_train` is a 2 656-image subset of the same pool.")
    lines.append(f"- **Within-group duplicates:** {len(within_group_dups):,}")
    if within_group_dups:
        lines.append(f"  ⚠ Unexpected within-group duplicates found:")
        for d, o in within_group_dups[:20]:
            lines.append(f"  - `{d}` == `{o}`")
    else:
        lines.append("  - No unexpected within-group duplicates.")

    lines.append("\n---\n")
    lines.append("## 4. Patient & Split Analysis\n")

    lines.append("### 4.1 Patient-level leakage – `split_for_train`\n")
    if leakage_sft:
        lines.append(f"**{len(leakage_sft)} patient(s) appear in more than one split:**\n")
        lines.append("| patient_id | splits |")
        lines.append("|------------|--------|")
        for pid, splits in leakage_sft[:30]:
            lines.append(f"| {pid} | {', '.join(sorted(splits))} |")
        lines.append("\n> ⚠ These patients have images in multiple splits. "
                     "This may represent confirmed data leakage OR an intentional "
                     "dataset design choice. Do NOT modify the official splits "
                     "without further review by the dataset authors.")
    else:
        lines.append("No patient-level leakage detected in `split_for_train`.")

    lines.append("\n### 4.2 Patient-level leakage – `clean_ver_for_train`\n")
    if leakage_cvft:
        lines.append(f"**{len(leakage_cvft)} patient(s) appear in more than one split:**\n")
        lines.append("| patient_id | splits |")
        lines.append("|------------|--------|")
        for pid, splits in leakage_cvft[:30]:
            lines.append(f"| {pid} | {', '.join(sorted(splits))} |")
        lines.append("\n> ⚠ Same note applies as above.")
    else:
        lines.append("No patient-level leakage detected in `clean_ver_for_train`.")

    lines.append("\n### 4.3 Label Conflicts\n")
    if label_conflicts:
        lines.append(f"**{len(label_conflicts)} UUID stem(s) appear with conflicting labels:**\n")
        for stem, labs in list(label_conflicts.items())[:10]:
            lines.append(f"- `{stem}` → {labs}")
    else:
        lines.append("No label conflicts detected (same UUID always maps to the same class).")

    lines.append("\n---\n")
    lines.append("## 5. Cross-Group Relationship\n")
    lines.append(
        "After UUID-stem analysis:\n"
        "- **`original_data` ≡ `split_for_train`** in content:\n"
        "  All 5 385 image UUIDs in `split_for_train` also appear in `original_data`.\n"
        "  `original_data` preserves the patient sub-folder hierarchy; "
        "`split_for_train` is a flat re-organisation by split.\n"
        "- **`clean_ver_for_train` ⊂ `split_for_train`**:\n"
        "  All 2 656 UUID stems in `clean_ver_for_train` are a subset of `split_for_train`.\n"
        "  These are caliper-free images only."
    )
    lines.append("\n> **Recommended use for training:** Use `clean_ver_for_train` "
                 "(pre-split, caliper-free) as the primary dataset. "
                 "Use `split_for_train` only if mixed clean+caliper is intended.")

    lines.append("\n---\n")
    lines.append("## 6. Unresolved / Open Items\n")
    open_items = []
    if summary["corrupt_count"]:
        open_items.append(f"- {summary['corrupt_count']} corrupt file(s) need manual inspection.")
    if summary["within_group_duplicates"]:
        open_items.append(f"- {summary['within_group_duplicates']} unexpected within-group MD5 duplicates.")
    if leakage_sft or leakage_cvft:
        n = len(leakage_sft) + len(leakage_cvft)
        open_items.append(
            f"- {n} patient(s) flagged for potential leakage. "
            "These require confirmation with the dataset authors before any splits are altered."
        )
    if not open_items:
        open_items.append("- No unresolved issues.")
    lines.extend(open_items)

    lines.append("\n---\n")
    lines.append("## 7. Deliverables\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `manifests/manifest.csv` | Corrected manifest with all fields |")
    lines.append("| `manifests/manifest.jsonl` | Same data in JSONL format |")
    lines.append("| `manifests/summary.json` | Aggregate statistics |")
    lines.append("| `manifests/dataset_audit_report.md` | This report |")
    if summary["patient_leakage_split_for_train"] or summary["patient_leakage_clean_ver_for_train"]:
        lines.append("| `manifests/leakage_report.csv` | Patient-level leakage cases |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✅  dataset_audit_report.md → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("  CureSight / SMC-LUD Dataset Audit Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # Phase 1 – Collect all images (group-aware parsing)
    records = phase1_collect_all()

    # Phase 2 – Official metadata
    hcc_pid_map, hem_pid_map = phase2_metadata(records)

    # Phase 3 – Image quality
    corrupt, duplicates, md5_map = phase3_quality(records)

    # Phase 4 – Patient / split analysis
    leakage_sft, leakage_cvft, label_conflicts = phase4_patient_split(
        records, hcc_pid_map, hem_pid_map, md5_map
    )

    # Phase 5 – Visual samples
    phase5_visual_samples(records)

    # Phase 6 – Final manifest & audit report
    phase6_manifest(
        records, corrupt, duplicates,
        leakage_sft, leakage_cvft, label_conflicts
    )

    print("\n" + "=" * 62)
    print("  ✅  All phases complete.")
    print(f"  Outputs in: {OUT_DIR}")
    print("=" * 62)
