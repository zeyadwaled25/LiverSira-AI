"""Repository-relative paths shared by the CV and NLP pipelines."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

CV_RAW_DIR = DATA_DIR / "raw" / "cv" / "SMC-LUD"
CV_MANIFEST_DIR = DATA_DIR / "manifests" / "cv" / "SMC-LUD"
CV_PROCESSED_DIR = DATA_DIR / "processed" / "cv" / "SMC-LUD"

NLP_RAW_DIR = DATA_DIR / "raw" / "nlp"
NLP_PROCESSED_DIR = DATA_DIR / "processed" / "nlp"
