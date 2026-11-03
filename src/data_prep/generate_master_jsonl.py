"""Generate Master Medical Records JSONL.

Transforms the full AbdomenAtlas 3.0 Mini dataset (all 37 columns, all 9,262 rows)
into validated MasterRecord objects and streams them to data/processed/master_records.jsonl.

Guarantees:
- 0 rows excluded (preserves all records regardless of missing narrative text).
- Preserves raw columns verbatim under source.raw_columns.
- Organizes findings for Liver, Spleen, Pancreas, and Kidneys.
- Isolates structured reports from narrative text to protect downstream NLP inputs.
"""

from pathlib import Path
import re
import pandas as pd
from schemas.master_schema import MasterRecord

INPUT_CSV = Path("data/raw/AbdomenAtlas3.0MiniWithMeta.csv")
OUTPUT_JSONL = Path("data/processed/master_records.jsonl")


def clean_val(v: object) -> object:
    """Return None for NaN/NA values so JSON serializes to null."""
    if pd.isna(v):
        return None
    return v


def parse_spacing(val: object) -> list[float] | None:
    """Parse spacing string like '[0.822 0.822 2.5]' into floats."""
    if pd.isna(val):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    return [float(x) for x in nums] if len(nums) == 3 else None


def parse_shape(val: object) -> list[int] | None:
    """Parse shape string like '(512, 512, 339)' into ints."""
    if pd.isna(val):
        return None
    nums = re.findall(r"\d+", str(val))
    return [int(x) for x in nums] if len(nums) == 3 else None


def clean_text(val: object) -> str | None:
    """Normalize whitespace and strip text."""
    if pd.isna(val):
        return None
    text = re.sub(r"\s+", " ", str(val)).strip()
    return text if text else None


def generate_master_records(input_path: Path, output_path: Path) -> int:
    """Read CSV, build MasterRecord for each row, and write JSONL."""
    print(f"Reading source CSV: {input_path}...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"Total source rows to process: {total_rows:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            # 1. Raw columns dictionary (NaN -> None)
            raw_cols = {col: clean_val(row[col]) for col in df.columns}

            # 2. Meta
            age_val = clean_val(row["age"])
            sex_val = clean_val(row["sex"])
            patient_age = float(age_val) if age_val is not None else None
            patient_sex = str(sex_val) if sex_val is not None else None

            # 3. Liver findings
            l_count = int(row["number of liver lesion instances"])
            l_largest = None
            if l_count > 0:
                loc_val = row["largest liver lesion location (1-8)"]
                segs = [int(x) for x in re.findall(r"\d+", str(loc_val))] if pd.notna(loc_val) else None
                l_largest = {
                    "location_segments": segs,
                    "source_diameter_cm": float(row["largest liver lesion diameter (cm)"]),
                    "attenuation": clean_val(
                        row["largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]
                    ),
                }

            # 4. Pancreas findings
            p_count = int(row["number of pancreatic lesion instances"])
            p_largest = None
            if p_count > 0:
                p_largest = {
                    "location": clean_val(row["largest pancreatic lesion location (head, body, tail)"]),
                    "source_diameter_cm": float(row["largest pancreatic lesion diameter (cm)"]),
                    "attenuation": clean_val(
                        row["largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]
                    ),
                }

            # 5. Kidney findings
            k_count = int(row["number of kidney lesion instances"])
            k_largest = None
            if k_count > 0:
                k_largest = {
                    "location": clean_val(row["largest kidney lesion location (left, right)"]),
                    "source_diameter_cm": float(row["largest kidney lesion diameter (cm)"]),
                    "attenuation": clean_val(
                        row["largest kidney lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]
                    ),
                }

            # 6. Colon findings
            c_count = int(row["number of colon lesion instances"])
            c_largest = None
            if c_count > 0:
                c_largest = {
                    "source_diameter_cm": float(row["largest colon lesion diameter (cm)"]),
                    "attenuation": clean_val(
                        row["largest colon lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)"]
                    ),
                }

            record = MasterRecord(
                id=str(row["BDMAP ID"]),
                source={
                    "dataset": "AbdomenAtlas3.0MiniWithMeta",
                    "schema_version": "1.0",
                    "raw_columns": raw_cols,
                },
                meta={
                    "scanner": clean_val(row["scanner"]),
                    "contrast": clean_val(row["contrast"]),
                    "patient": {"age": patient_age, "sex": patient_sex},
                    "study_spacing": parse_spacing(row["spacing"]),
                    "shape": parse_shape(row["shape"]),
                },
                organ_findings={
                    "liver": {
                        "volume_cc": clean_val(row["liver volume (cm^3)"]),
                        "total_lesion_volume_cc": float(row["total liver lesion volume (cm^3)"]),
                        "lesion_count": l_count,
                        "largest_lesion": l_largest,
                    },
                    "spleen": {
                        "volume_cc": clean_val(row["spleen volume (cm^3)"]),
                    },
                    "pancreas": {
                        "volume_cc": clean_val(row["pancreas volume (cm^3)"]),
                        "total_lesion_volume_cc": float(row["total pancreatic lesion volume (cm^3)"]),
                        "lesion_count": p_count,
                        "largest_lesion": p_largest,
                    },
                    "kidneys": {
                        "left_volume_cc": clean_val(row["left kidney volume (cm^3)"]),
                        "right_volume_cc": clean_val(row["right kidney volume (cm^3)"]),
                        "total_volume_cc": clean_val(row["kidney volume (cm^3)"]),
                        "total_lesion_volume_cc": float(row["total kidney lesion volume (cm^3)"]),
                        "lesion_count": k_count,
                        "largest_lesion": k_largest,
                    },
                    "colon": {
                        "total_lesion_volume_cc": float(row["total colon lesion volume (cm^3)"]),
                        "lesion_count": c_count,
                        "largest_lesion": c_largest,
                    },
                },
                reports={
                    "narrative_original": clean_val(row["narrative report"]),
                    "narrative_clean": clean_text(row["narrative report"]),
                    "structured_report": clean_val(row["structured report"]),
                    "fusion_structured_report": clean_val(row["fusion structured report"]),
                    "fusion_narrative_report": clean_val(row["fusion narrative report"]),
                    "radiologist_note": clean_val(row["radiologist note"]),
                },
            )

            # Dump as a single JSON line
            f_out.write(record.model_dump_json() + "\n")
            written_count += 1

            if written_count % 2000 == 0:
                print(f"Processed {written_count:,} / {total_rows:,} records...")

    print(f"\n[DONE] Successfully generated {written_count:,} master records.")
    print(f"Output saved to: {output_path}")
    return written_count


def main() -> None:
    """Execute generator script."""
    generate_master_records(INPUT_CSV, OUTPUT_JSONL)


if __name__ == "__main__":
    main()

