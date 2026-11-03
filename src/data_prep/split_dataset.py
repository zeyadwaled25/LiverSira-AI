"""Split dataset into Train (70%), Validation (15%), and Test (15%) partitions.

Methodology:
- Stratified Split based on `has_lesion_gt` (ensuring balanced positive/negative ratio across splits).
- Independent Study Units: Each BDMAP ID corresponds to an independent CT scan.
- Leakage Prevention: Narrative reports are strictly segregated across partitions.
- Saves partition indices/metadata to data/processed/splits.json.
"""

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

AUDIT_CSV = Path("data/processed/target_consistency_audit.csv")
OUTPUT_SPLITS_JSON = Path("data/processed/splits.json")
OUTPUT_SPLITS_CSV = Path("data/processed/dataset_splits.csv")


def generate_splits(
    audit_csv_path: Path,
    splits_json_path: Path,
    splits_csv_path: Path,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Execute stratified 70/15/15 dataset split and persist partition mappings."""
    print("=" * 60)
    print("DATASET PARTITIONING (TRAIN / VAL / TEST)")
    print("=" * 60)
    print(f"Reading audit records from: {audit_csv_path}")

    df = pd.read_csv(audit_csv_path)

    # Exclude record without usable narrative text (BDMAP_00005266)
    valid_df = df[df["discrepancy_details"] != "Missing narrative report"].copy()
    total_valid = len(valid_df)
    print(f"Total valid studies available for modeling: {total_valid:,}")

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        valid_df,
        test_size=0.30,
        random_state=random_state,
        stratify=valid_df["has_lesion_gt"],
    )

    # Second split: 15% Val, 15% Test (50/50 of Temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["has_lesion_gt"],
    )

    splits = {
        "train": sorted(train_df["id"].tolist()),
        "val": sorted(val_df["id"].tolist()),
        "test": sorted(test_df["id"].tolist()),
    }

    # Summary stats
    print("\n" + "-" * 60)
    print(f"{'Split':<12} | {'Count':<8} | {'Share (%)':<10} | {'Lesion Positive (%)':<20}")
    print("-" * 60)
    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        cnt = len(split_df)
        share = cnt / total_valid * 100
        pos_rate = split_df["has_lesion_gt"].mean() * 100
        print(f"{name:<12} | {cnt:<8} | {share:>8.1f}% | {pos_rate:>18.2f}%")
    print("-" * 60)

    # Save to JSON
    splits_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_json_path, "w", encoding="utf-8") as f_json:
        json.dump(splits, f_json, indent=2)

    # Save mapping CSV
    mapping_rows = []
    for split_name, ids in splits.items():
        for item_id in ids:
            mapping_rows.append({"id": item_id, "split": split_name})
    pd.DataFrame(mapping_rows).to_csv(splits_csv_path, index=False)

    print(f"\n[DONE] Partition IDs saved to:  {splits_json_path}")
    print(f"[DONE] Partition table saved to: {splits_csv_path}")
    print("=" * 60)

    return splits


def main() -> None:
    """CLI entry point."""
    generate_splits(AUDIT_CSV, OUTPUT_SPLITS_JSON, OUTPUT_SPLITS_CSV)


if __name__ == "__main__":
    main()

