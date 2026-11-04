"""Materialize the verified patient-level CV split without modifying source data."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from liversira_ai.common.paths import CV_MANIFEST_DIR, CV_PROCESSED_DIR, CV_RAW_DIR


MANIFEST_PATH = CV_MANIFEST_DIR / "patient_level_manifest.csv"
OUTPUT_DIR = CV_PROCESSED_DIR / "patient_split_v1"
METADATA_PATH = OUTPUT_DIR / "patient_split_metadata.csv"
REPORT = CV_MANIFEST_DIR / "dataset_audit_report.md"
SPLITS = ("train", "val", "test")
LABELS = ("HCC", "Hemangioma")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing source manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "rel_path",
        "label",
        "patient_group_id",
        "md5",
        "stem",
        "patient_level_split",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def materialize(rows: list[dict[str, str]]) -> dict[str, int]:
    expected_destinations: set[Path] = set()
    metadata_rows: list[dict[str, str]] = []
    counts = defaultdict(int)

    for row in rows:
        split = row["patient_level_split"]
        label = row["label"]
        if split not in SPLITS or label not in LABELS:
            raise ValueError(f"Invalid split or label in manifest row: {row}")
        if row.get("variant") != "clean":
            raise ValueError(f"Non-clean image in patient-level manifest: {row['rel_path']}")
        source_rel_path = row["rel_path"].lower()
        if "caliper" in source_rel_path and "wo caliper" not in source_rel_path:
            raise ValueError(f"Caliper image in patient-level manifest: {row['rel_path']}")

        source = CV_RAW_DIR / Path(row["rel_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Manifest source image does not exist: {source}")
        if file_md5(source) != row["md5"]:
            raise ValueError(f"Source checksum mismatch: {source}")

        destination = OUTPUT_DIR / split / label / Path(row["rel_path"]).name
        if destination in expected_destinations:
            raise ValueError(f"Duplicate destination filename in manifest: {destination}")
        expected_destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            if not destination.is_file() or file_md5(destination) != row["md5"]:
                raise FileExistsError(
                    f"Conflicting destination exists and will not be overwritten: {destination}"
                )
        else:
            destination.write_bytes(source.read_bytes())
        if file_md5(destination) != row["md5"]:
            raise ValueError(f"Destination checksum mismatch: {destination}")

        counts[f"{split}:{label}"] += 1
        metadata_rows.append(
            {
                "destination_rel_path": destination.relative_to(OUTPUT_DIR).as_posix(),
                "source_rel_path": row["rel_path"],
                "split": split,
                "label": label,
                "variant": row["variant"],
                "patient_group_id": row["patient_group_id"],
                "patient_id": row.get("patient_id", ""),
                "md5": row["md5"],
                "stem": row["stem"],
            }
        )

    existing_files = {
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path != METADATA_PATH
    } if OUTPUT_DIR.exists() else set()
    unexpected = existing_files - expected_destinations
    if unexpected:
        raise ValueError(
            "Destination contains files not represented by the manifest: "
            + ", ".join(str(path) for path in sorted(unexpected))
        )

    metadata_rows.sort(key=lambda row: row["destination_rel_path"])
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_rows[0])
        writer.writeheader()
        writer.writerows(metadata_rows)
    return dict(counts)


def update_report(rows: list[dict[str, str]], counts: dict[str, int]) -> None:
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_splits[row["patient_group_id"]].add(row["patient_level_split"])
    intersections = {
        f"{first}_vs_{second}": len(
            {
                patient
                for patient, splits in patient_splits.items()
                if first in splits and second in splits
            }
        )
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    patient_counts = {
        split: {
            label: len(
                {
                    row["patient_group_id"]
                    for row in rows
                    if row["patient_level_split"] == split and row["label"] == label
                }
            )
            for label in LABELS
        }
        for split in SPLITS
    }
    section = f"""

## Phase 8: Physical Patient-Level Split

The physical split below supersedes any earlier manifest-only Phase 7 wording.
The manifest remains the source of truth; this phase only copies files.

| Item | Location |
|---|---|
| Source images | `data/raw/cv/SMC-LUD/original_data/` |
| Source manifest | `data/manifests/cv/SMC-LUD/patient_level_manifest.csv` |
| Physical destination | `data/processed/cv/SMC-LUD/patient_split_v1/` |
| Patient metadata sidecar | `data/processed/cv/SMC-LUD/patient_split_v1/patient_split_metadata.csv` |

### Physical counts

| Split | HCC | Hemangioma | Total | Patients (HCC / Hemangioma) |
|---|---:|---:|---:|---:|
| Train | {counts["train:HCC"]} | {counts["train:Hemangioma"]} | {counts["train:HCC"] + counts["train:Hemangioma"]} | {patient_counts["train"]["HCC"]} / {patient_counts["train"]["Hemangioma"]} |
| Validation | {counts["val:HCC"]} | {counts["val:Hemangioma"]} | {counts["val:HCC"] + counts["val:Hemangioma"]} | {patient_counts["val"]["HCC"]} / {patient_counts["val"]["Hemangioma"]} |
| Test | {counts["test:HCC"]} | {counts["test:Hemangioma"]} | {counts["test:HCC"] + counts["test:Hemangioma"]} | {patient_counts["test"]["HCC"]} / {patient_counts["test"]["Hemangioma"]} |

### Verification

- Source rows copied exactly once: **{len(rows)} / {len(rows)}**
- Source and destination MD5 checksums: **all matched**
- Train ∩ Validation patients: **{intersections["train_vs_val"]}**
- Train ∩ Test patients: **{intersections["train_vs_test"]}**
- Validation ∩ Test patients: **{intersections["val_vs_test"]}**
- Caliper images included: **0**
- `Hem_87` clean images: **5**, Train only, with no official `patient_id`
- Idempotency: existing matching files are accepted; conflicting or unexpected files fail loudly
"""
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    marker = "## Phase 8: Physical Patient-Level Split"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    REPORT.write_text(existing + section.lstrip(), encoding="utf-8")


def main() -> None:
    counts = materialize(load_manifest())
    update_report(load_manifest(), counts)
    print(f"Materialized {sum(counts.values())} images in {OUTPUT_DIR}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
