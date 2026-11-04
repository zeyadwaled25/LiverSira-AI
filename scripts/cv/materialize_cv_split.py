"""Run the CV physical split materializer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from liversira_ai.cv.data.materialize_patient_split import main


if __name__ == "__main__":
    main()
