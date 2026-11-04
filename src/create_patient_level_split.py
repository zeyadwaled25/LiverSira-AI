"""Create a reproducible patient-level split for SMC-LUD clean images.

This phase intentionally reads only the clean image folders under
``original_data``. It does not modify, copy, resize, or preprocess images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from curesight.common.paths import CV_MANIFEST_DIR, CV_RAW_DIR

BASE = CV_RAW_DIR
ORIGINAL = BASE / "original_data"
MANIFEST_DIR = CV_MANIFEST_DIR
OUTPUT_MANIFEST = MANIFEST_DIR / "patient_level_manifest.csv"
OUTPUT_SUMMARY = MANIFEST_DIR / "patient_level_split_summary.json"
REPORT = MANIFEST_DIR / "dataset_audit_report.md"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_IDS = {"HCC": 0, "Hemangioma": 1}
CSV_FIELDS = [
    "rel_path",
    "dataset_group",
    "split",
    "label",
    "label_id",
    "variant",
    "patient_id",
    "patient_group_id",
    "official_metadata_available",
    "width",
    "height",
    "mode",
    "ext",
    "size_bytes",
    "md5",
    "stem",
]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_subject_ids(path: Path, label: str) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if "Subject_id" not in (rows.fieldnames or []):
            raise ValueError(f"{path} does not contain Subject_id")
        values = set()
        for row in rows:
            value = (row.get("Subject_id") or "").strip()
            if label == "HCC" and value.isdigit():
                value = str(int(value))
            if value:
                values.add(value)
        return values


def load_metadata_ids() -> dict[str, set[str]]:
    return {
        "HCC": read_subject_ids(
            ORIGINAL / "HCC" / "HCC_patient_table.csv", "HCC"
        ),
        "Hemangioma": read_subject_ids(
            ORIGINAL / "Hemangioma" / "Hem_patient_table.csv", "Hemangioma"
        ),
    }


def image_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None, None

    try:
        with Image.open(path) as image:
            return image.width, image.height, image.mode
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read image {path}: {exc}") from exc


def collect_clean_images(metadata_ids: dict[str, set[str]]) -> list[dict[str, object]]:
    records = []
    variant_dirs = {
        "HCC": ORIGINAL / "HCC" / "HCC_clean (wo caliper)",
        "Hemangioma": ORIGINAL / "Hemangioma" / "Hem_clean (wo caliper)",
    }
    for label, variant_dir in variant_dirs.items():
        if not variant_dir.is_dir():
            raise FileNotFoundError(f"Missing clean image directory: {variant_dir}")
        for patient_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
            patient_group_id = patient_dir.name
            expected_prefix = "HCC_" if label == "HCC" else "Hem_"
            if not patient_group_id.startswith(expected_prefix):
                raise ValueError(
                    f"Unexpected {label} patient folder: {patient_group_id}"
                )
            patient_id = (
                patient_group_id.removeprefix("HCC_")
                if label == "HCC"
                else patient_group_id
            )
            available = patient_id in metadata_ids[label]
            if not available and patient_group_id != "Hem_87":
                raise ValueError(
                    f"Missing official metadata for unexpected patient {patient_group_id}"
                )
            for path in sorted(patient_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                width, height, mode = image_dimensions(path)
                records.append(
                    {
                        "rel_path": path.relative_to(BASE).as_posix(),
                        "dataset_group": "original_data",
                        "split": "original",
                        "label": label,
                        "label_id": LABEL_IDS[label],
                        "variant": "clean",
                        "patient_id": patient_id if available else "",
                        "patient_group_id": patient_group_id,
                        "official_metadata_available": str(available).lower(),
                        "width": width if width is not None else "",
                        "height": height if height is not None else "",
                        "mode": mode or "",
                        "ext": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                        "md5": md5(path),
                        "stem": path.stem,
                    }
                )
    return records


def allocate_counts(count: int) -> dict[str, int]:
    quotas = {name: count * ratio for name, ratio in RATIOS.items()}
    counts = {name: int(quota) for name, quota in quotas.items()}
    remaining = count - sum(counts.values())
    order = sorted(
        RATIOS,
        key=lambda name: (quotas[name] - counts[name], RATIOS[name]),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def make_patient_assignments(records: list[dict[str, object]]) -> dict[str, str]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[str(record["patient_group_id"])].add(str(record["label"]))
    conflicting = {group: labels for group, labels in groups.items() if len(labels) != 1}
    if conflicting:
        raise ValueError(f"Patients have conflicting labels: {conflicting}")

    assignments: dict[str, str] = {"Hem_87": "train"}
    randomizer = random.Random(SEED)
    for label in ("HCC", "Hemangioma"):
        patient_groups = sorted(
            group for group in groups if group != "Hem_87" and next(iter(groups[group])) == label
        )
        randomizer.shuffle(patient_groups)
        counts = allocate_counts(len(patient_groups))
        start = 0
        for split in ("train", "val", "test"):
            for group in patient_groups[start : start + counts[split]]:
                assignments[group] = split
            start += counts[split]
    if set(assignments) != set(groups):
        raise AssertionError("Not every patient group received a split")
    return assignments


def verify_and_write(records: list[dict[str, object]], assignments: dict[str, str]) -> dict:
    for record in records:
        record["patient_level_split"] = assignments[str(record["patient_group_id"])]

    split_groups = {
        split: {
            str(record["patient_group_id"])
            for record in records
            if record["patient_level_split"] == split
        }
        for split in RATIOS
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[first] & split_groups[second]
        if overlap:
            raise AssertionError(f"Patient leakage in {first}/{second}: {sorted(overlap)}")

    split_uuids = {
        split: {str(record["stem"]) for record in records if record["patient_level_split"] == split}
        for split in RATIOS
    }
    split_md5 = {
        split: {str(record["md5"]) for record in records if record["patient_level_split"] == split}
        for split in RATIOS
    }
    uuid_overlaps = {}
    md5_overlaps = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        uuid_overlaps[f"{first}_vs_{second}"] = len(split_uuids[first] & split_uuids[second])
        md5_overlaps[f"{first}_vs_{second}"] = len(split_md5[first] & split_md5[second])
    if any(uuid_overlaps.values()) or any(md5_overlaps.values()):
        raise AssertionError("Image UUID or MD5 leakage detected")
    if len(records) != len({str(record["rel_path"]) for record in records}):
        raise AssertionError("A selected image was collected more than once")

    output_fields = CSV_FIELDS + ["patient_level_split"]
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(records)

    patient_distribution = {
        split: {
            label: sum(
                1
                for group, assigned in assignments.items()
                if assigned == split
                and next(
                    str(record["label"])
                    for record in records
                    if str(record["patient_group_id"]) == group
                )
                == label
            )
            for label in LABEL_IDS
        }
        for split in RATIOS
    }
    image_distribution = {
        split: {
            label: sum(
                1
                for record in records
                if record["patient_level_split"] == split and record["label"] == label
            )
            for label in LABEL_IDS
        }
        for split in RATIOS
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "original_data clean images only",
        "random_seed": SEED,
        "split_strategy": "stratified patient-level (70/20/10)",
        "total_patients": len(assignments),
        "total_images": len(records),
        "patient_distribution": patient_distribution,
        "image_distribution": image_distribution,
        "hem_87_handling": {
            "patient_group_id": "Hem_87",
            "official_patient_metadata_available": False,
            "assigned_split": "train",
        },
        "leakage_verification": {
            "patient_intersections": {key: 0 for key in ("train_val", "train_test", "val_test")},
            "uuid_intersections": uuid_overlaps,
            "md5_intersections": md5_overlaps,
            "unique_image_assignments": len(records),
        },
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def append_report(summary: dict) -> None:
    report_section = f"""

## Phase 7: Patient-Level Dataset Splitting

**Generated:** {summary["generated_at"]}  
**Source:** `original_data/HCC/HCC_clean (wo caliper)` and `original_data/Hemangioma/Hem_clean (wo caliper)` only  
**Random seed:** `{SEED}`

### Methodology

Images were grouped by their verified patient folder (`patient_group_id`) before
assignment. Patients were stratified by class and assigned approximately 70% to
train, 20% to validation, and 10% to test. No image was copied or modified.
`patient_id` is populated only when the corresponding official patient table
contains the patient; the grouping key remains the folder ID.

### Patient counts

| Split | HCC | Hemangioma | Total |
|---|---:|---:|---:|
| Train | {summary["patient_distribution"]["train"]["HCC"]} | {summary["patient_distribution"]["train"]["Hemangioma"]} | {sum(summary["patient_distribution"]["train"].values())} |
| Validation | {summary["patient_distribution"]["val"]["HCC"]} | {summary["patient_distribution"]["val"]["Hemangioma"]} | {sum(summary["patient_distribution"]["val"].values())} |
| Test | {summary["patient_distribution"]["test"]["HCC"]} | {summary["patient_distribution"]["test"]["Hemangioma"]} | {sum(summary["patient_distribution"]["test"].values())} |

### Image counts

| Split | HCC | Hemangioma | Total |
|---|---:|---:|---:|
| Train | {summary["image_distribution"]["train"]["HCC"]} | {summary["image_distribution"]["train"]["Hemangioma"]} | {sum(summary["image_distribution"]["train"].values())} |
| Validation | {summary["image_distribution"]["val"]["HCC"]} | {summary["image_distribution"]["val"]["Hemangioma"]} | {sum(summary["image_distribution"]["val"].values())} |
| Test | {summary["image_distribution"]["test"]["HCC"]} | {summary["image_distribution"]["test"]["Hemangioma"]} | {sum(summary["image_distribution"]["test"].values())} |

### Leakage verification

- Train ∩ Validation patients: **0**
- Train ∩ Test patients: **0**
- Validation ∩ Test patients: **0**
- Cross-split UUID intersections: **0**
- Cross-split MD5 intersections: **0**
- Selected images with exactly one assignment: **{summary["total_images"]} / {summary["total_images"]}**

### Hem_87 handling

`Hem_87` is represented by `patient_group_id = Hem_87`, while
`official_metadata_available = false` and `patient_id` remains empty. All five
clean images from this group were assigned to **train** only. No patient
information was inferred or invented.

### Phase 7 deliverables

- `manifests/patient_level_manifest.csv`
- `manifests/patient_level_split_summary.json`
"""
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    marker = "## Phase 7: Patient-Level Dataset Splitting"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    REPORT.write_text(existing + report_section.lstrip(), encoding="utf-8")


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    metadata_ids = load_metadata_ids()
    records = collect_clean_images(metadata_ids)
    assignments = make_patient_assignments(records)
    summary = verify_and_write(records, assignments)
    append_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
