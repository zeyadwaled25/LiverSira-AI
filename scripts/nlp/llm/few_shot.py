"""Load the few-shot examples (IDs and hand-written outputs in few_shot_v1.json) with their liver-only text.

Checks, every time they are loaded:
- every ID is a Train study (never Validation or Test; the Challenge Sets come from Validation);
- each output is a valid ``LLMOutput`` and passes the evidence check with nothing rejected;
- each verifiable value agrees with the study's source labels.
"""

import csv
import json
from pathlib import Path
from typing import Any

from liversira_ai.nlp import llm_extraction as le
from liversira_ai.nlp.extraction_targets import targets_from_source

HERE = Path(__file__).resolve().parent
FEW_SHOT = HERE / "few_shot_v1.json"
LIVER_ONLY_CSV = Path("data/processed/nlp/liver_nlp_liver_only.csv")
SPLITS = Path("data/processed/nlp/splits.json")
SOURCE = {"count": "number of liver lesion instances", "location": "largest liver lesion location (1-8)",
          "diameter": "largest liver lesion diameter (cm)",
          "attenuation": "largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)"}


def _int(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0


def load_examples(path: Path = FEW_SHOT) -> list[dict[str, Any]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    ids = [e["id"] for e in spec["examples"]]
    if not set(ids) <= set(splits["train"]) or set(ids) & (set(splits["val"]) | set(splits["test"])):
        raise ValueError("few-shot examples must come from Train only")
    rows = {r["BDMAP ID"]: r for r in csv.DictReader(LIVER_ONLY_CSV.open(encoding="utf-8-sig")) if r["BDMAP ID"] in ids}
    examples = []
    for example in spec["examples"]:
        row = rows[example["id"]]
        text = row["narrative_clean"]
        output = le.LLMOutput.model_validate(example["output"])
        verified = le.verify(output, text)
        if verified.rejected:
            raise ValueError(f"{example['id']}: few-shot evidence rejected for {verified.rejected}")
        source = targets_from_source(text, _int(row[SOURCE["count"]]), row[SOURCE["location"]],
                                     row[SOURCE["diameter"]], row[SOURCE["attenuation"]])
        written = example["output"]
        dims = written["reported_dimensions_cm"]["value"]
        checks = {
            "count": (source["count"], None if written["lesion_count"]["value"] is None else str(written["lesion_count"]["value"])),
            "segments": (source["segments"], written["location_segments"]["value"]),
            "attenuation": (source["attenuation"], written["attenuation"]["value"]),
            "diameter_cm": (source["diameter_cm"], max(dims) if dims else None),
        }
        for name, (src, mine) in checks.items():
            if src is not None and src != mine:
                raise ValueError(f"{example['id']}: {name} {mine!r} disagrees with the source label {src!r}")
        examples.append({"id": example["id"], "text": text, "output": written})
    return examples


if __name__ == "__main__":
    loaded = load_examples()
    p_hash = le.prompt_hash(le.SYSTEM_PROMPT, [{"text": e["text"], "output": e["output"]} for e in loaded],
                            le.OUTPUT_SCHEMA)
    print(f"{len(loaded)} examples OK; prompt {le.PROMPT_VERSION} hash {p_hash}")
