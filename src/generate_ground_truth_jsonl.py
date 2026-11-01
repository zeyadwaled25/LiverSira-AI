"""Generate Source-Derived Ground Truth Benchmark JSONL.

Constructs initial GroundTruthRecord objects from the AbdomenAtlas 3.0 Mini CSV labels.
Initial status is strictly marked as PENDING verification, acknowledging that
source-derived labels must undergo consistency validation against narrative reports
before serving as an adjudicated clinical benchmark.

Guarantees:
- 9,262 records matching Master Record IDs 1:1.
- Strict consistency: has_lesion == False <==> lesion_count == 0 & largest_lesion is None.
- Preserves missing anatomical locations as None without inventing segments.
- Tracks unverified field paths for auditing.
"""

from pathlib import Path
import re
import pandas as pd
from schemas.ground_truth_schema import (
    Attenuation,
    GroundTruthLabels,
    GroundTruthRecord,
    LargestLesionGT,
    ValidationMetadata,
    ValidationStatus,
)

INPUT_CSV = Path("data/raw/AbdomenAtlas3.0MiniWithMeta.csv")
OUTPUT_JSONL = Path("data/processed/ground_truth.jsonl")


def generate_ground_truth_records(input_path: Path, output_path: Path) -> int:
    """Read CSV, build GroundTruthRecord for each row, and write JSONL."""
    print(f"Reading source CSV: {input_path}...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"Total source rows to process: {total_rows:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            record_id = str(row["BDMAP ID"])
            count = int(row["number of liver lesion instances"])
            has_lesion = count > 0

            if has_lesion:
                # Location segments
                loc_raw = row["largest liver lesion location (1-8)"]
                if pd.notna(loc_raw):
                    segs = [int(x) for x in re.findall(r"\d+", str(loc_raw))]
                else:
                    segs = None

                # Diameter
                diam = float(row["largest liver lesion diameter (cm)"])

                # Attenuation
                att_raw = row[
                    "largest liver lesion attenuation "
                    "(hyperattenuating, isoattenuating, hypoattenuating)"
                ]
                att_enum = Attenuation(str(att_raw)) if pd.notna(att_raw) else None

                largest_lesion = LargestLesionGT(
                    location_segments=segs,
                    source_diameter_cm=diam,
                    attenuation=att_enum,
                )

                unverified = [
                    "has_lesion",
                    "lesion_count",
                    "largest_lesion.location_segments",
                    "largest_lesion.source_diameter_cm",
                    "largest_lesion.attenuation",
                ]
            else:
                largest_lesion = None
                unverified = ["has_lesion", "lesion_count"]

            record = GroundTruthRecord(
                id=record_id,
                ground_truth=GroundTruthLabels(
                    has_lesion=has_lesion,
                    lesion_count=count,
                    largest_lesion=largest_lesion,
                ),
                validation=ValidationMetadata(
                    status=ValidationStatus.PENDING,
                    verified_fields=[],
                    unverified_fields=unverified,
                    review_notes=[],
                ),
            )

            f_out.write(record.model_dump_json() + "\n")
            written_count += 1

            if written_count % 2000 == 0:
                print(f"Processed {written_count:,} / {total_rows:,} records...")

    print(f"\n[DONE] Successfully generated {written_count:,} ground truth records.")
    print(f"Output saved to: {output_path}")
    return written_count


def main() -> None:
    """Execute generator script."""
    generate_ground_truth_records(INPUT_CSV, OUTPUT_JSONL)


if __name__ == "__main__":
    main()

