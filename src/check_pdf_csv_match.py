"""Cross-check BDMAP IDs between the CSV metadata and PDF report files.

Reports:
- How many CSV IDs have a matching PDF and vice-versa.
- CSV IDs that are missing a PDF file.
- PDF files without a corresponding CSV row.
- IDs that map to more than one PDF file (duplicates).
"""

from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path("data/raw")
CSV_PATH = DATA_DIR / "AbdomenAtlas3.0MiniWithMeta.csv"


def collect_pdf_ids(data_dir: Path) -> dict[str, list[Path]]:
    """Scan *data_dir/pdf_reports/* and return ``{BDMAP_ID: [paths]}``."""
    pdf_ids: dict[str, list[Path]] = {}

    for pdf_path in data_dir.rglob("./pdf_reports/*.pdf"):
        match = re.search(r"(BDMAP_\d+)", pdf_path.stem, re.IGNORECASE)

        if match:
            pdf_id = match.group(1).upper()
            pdf_ids.setdefault(pdf_id, []).append(pdf_path)
        else:
            print(f"Filename without BDMAP ID: {pdf_path}")

    return pdf_ids


def main() -> None:
    """Run the CSV <-> PDF matching report."""
    df = pd.read_csv(CSV_PATH)

    print("=" * 50)
    print("CSV AUDIT")
    print("=" * 50)

    print(f"CSV rows: {len(df)}")
    print(f"Unique BDMAP IDs: {df['BDMAP ID'].nunique()}")
    print(f"Duplicate BDMAP IDs: {df['BDMAP ID'].duplicated().sum()}")

    pdf_ids = collect_pdf_ids(DATA_DIR)
    pdf_files = [p for paths in pdf_ids.values() for p in paths]

    print(f"\nPDF files found: {len(pdf_files)}")

    csv_ids = set(df["BDMAP ID"].dropna().astype(str).str.upper())
    pdf_id_set = set(pdf_ids.keys())

    missing_pdfs = sorted(csv_ids - pdf_id_set)
    pdfs_without_csv = sorted(pdf_id_set - csv_ids)

    duplicate_pdf_ids = {
        pdf_id: paths for pdf_id, paths in pdf_ids.items() if len(paths) > 1
    }

    print("\n" + "=" * 50)
    print("CSV <-> PDF MATCHING")
    print("=" * 50)

    print(f"CSV IDs with matching PDF: {len(csv_ids & pdf_id_set)}")
    print(f"CSV IDs without PDF: {len(missing_pdfs)}")
    print(f"PDF IDs without CSV row: {len(pdfs_without_csv)}")
    print(f"IDs with multiple PDF files: {len(duplicate_pdf_ids)}")

    if missing_pdfs:
        print("\nFirst 10 CSV IDs without PDFs:")
        for item in missing_pdfs[:10]:
            print(item)

    if pdfs_without_csv:
        print("\nFirst 10 PDF IDs without CSV rows:")
        for item in pdfs_without_csv[:10]:
            print(item)

    if duplicate_pdf_ids:
        print("\nIDs with multiple PDF files:")
        for pdf_id, paths in list(duplicate_pdf_ids.items())[:10]:
            print(pdf_id, "->", [str(p) for p in paths])

    print("\nCheck completed.")


if __name__ == "__main__":
    main()
