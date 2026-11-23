"""Write the taxonomy of phrasing families in the 6.2 Train-aug data.

Lists only what is in Train-aug (extraction_train_v1, extraction_train_lesion_v1):
each family's construction, the fields it carries, how many items each set has,
and examples taken from the data. Nothing here describes Challenge Sets or
reserves families for Challenge-B. The Challenge-B writer sees this file only
after their items are frozen, and uses it to mark each item as a Train-aug
family or "new". Writes docs/nlp/extraction_train_aug_taxonomy.{md,json}.
"""

import json
from collections import Counter
from pathlib import Path

from liversira_ai.common.provenance import sha256_file
from scripts.nlp.transformers.extraction.train import read_jsonl

DATA = Path("data/processed/nlp/extraction")
SETS = ("extraction_train_v1", "extraction_train_lesion_v1")
OUT = Path("docs/nlp/extraction_train_aug_taxonomy")
EXAMPLES = 2

# Construction of each generated family, read from the generator templates.
FAMILIES: dict[str, tuple[str, str]] = {
    "count/present": ("count, segments, size", "\"<N> <att> lesions are present in the liver.\" then a separate "
                      "\"The largest, in <seg>, measures <size>.\" (N as digits or a word)."),
    "count/in_total": ("count", "\"... lesions, <N> in total; the largest lies in <seg> with a size of <size>.\""),
    "count/listed": ("count, order", "Labelled list \"Lesion 1: <seg>, <size>, <att>. Lesion 2: ...\" (2-5 lesions, "
                     "shuffled; count = number of labels)."),
    "count/another": ("count", "\"One <att> lesion is present in <seg> (<size>). Another lesion ... A further lesion "
                      "... Yet another lesion ...\" (2-4 lesions; count = number of sentences)."),
    "count/only_one": ("count", "\"Only one <att> lesion is present ...\" or \"There is a lone <att> lesion ...\"."),
    "count/vague": ("count (unstated)", "\"Innumerable / A few / Scattered / Various <att> lesions are present "
                    "throughout the liver; the largest, in <seg>, has a size of <size>.\" Count is unstated."),
    "size/single_mm": ("size", "One dimension in millimetres: \"A <att> lesion of <NN> mm lies in <seg>.\""),
    "size/cm_by_cm": ("size", "\"<a> cm by <b> cm\"."),
    "size/greatest_diameter": ("size", "\"... has a greatest diameter of <a> cm.\""),
    "size/three_in_size": ("size", "Three dimensions with the unit after each: \"<a> cm x <b> cm x <c> cm in size\"."),
    "size/reaching": ("size", "\"..., reaching <a> cm, ...\" or \"as large as <a> cm\"."),
    "segment/roman": ("segments", "Roman numerals: \"in liver segment VI\", \"(segment VI)\"."),
    "segment/subsegment": ("segments", "\"segment 4A/4B\", \"subsegment IVa/IVb\" (= segment 4)."),
    "segment/couinaud": ("segments", "\"in segment <n> according to Couinaud\"."),
    "segment/ordinal": ("segments", "\"in the 5th segment of the liver\", \"within segment five\", "
                        "\"in the fifth segment of the liver\"."),
    "segment/range": ("segments", "\"extending from segment <n> into segment <n+1>\", \"spanning segments <n> to <n+1>\"."),
    "segment/lobe": ("segments (unstated)", "Lobe only: \"in the right/left liver lobe\", \"within the right/left "
                     "lobe\". Segments are unstated."),
    "attenuation/appearing": ("attenuation", "\"hypodense / hyperdense / isodense\": \"A lesion appearing hypodense "
                              "...\", \"<Seg> contains a hypodense focus of <size>.\""),
    "attenuation/of_density": ("attenuation", "\"of low/high/equal density\", \"with low/high/equal attenuation\"."),
    "attenuation/relative": ("attenuation", "Comparison with the liver: \"darker/brighter than the adjacent hepatic "
                             "tissue\", \"less dense than / denser than background liver\", \"as dense as ...\", "
                             "\"indistinguishable in density from background liver\"."),
    "attenuation/separate": ("attenuation", "In its own sentence after the lesion: \"It appears hypoattenuating.\", "
                             "\"Its attenuation is low/high.\", \"Its attenuation matches the liver.\""),
    "order/size_list": ("order, count", "One sentence listing \"<seg> (<size>)\" for 2-4 lesions, ascending or "
                        "shuffled: \"<Att> lesions are seen in segment 1 (...) and segments 3/6 (...).\""),
    "order/larger_of_two": ("order, count", "Smaller lesion first, then \"The larger of the two, <att>, lies in <seg> "
                            "with a size of <size>.\""),
    "order/there_is": ("order, count", "\"There are lesions in <seg>, a <size> <att> lesion; in <seg>, a <size> "
                       "<att> lesion; ...\" (2-4, shuffled)."),
    "paraphrase/single": ("all", "Other nouns and verbs for one lesion: \"<Seg> is the site of a <size> <att> focus / "
                          "tumor / space-occupying lesion.\", \"A <att> <noun> sits in <seg>; size <size>.\""),
    "paraphrase/multiple": ("all", "\"The liver holds <N> <att> foci/tumors. The most prominent / greatest / principal "
                            "of these sits in <seg>, sized <size>; the others are under <x> cm.\""),
    "repeat/single": ("count, size", "One lesion, restated in a tail: \"There is a <att> lesion of <size> at <seg>. "
                      "IMPRESSION: <att> lesion, <seg>, <size>.\" (count stays 1)."),
    "repeat/multiple": ("count, size", "Listed lesions, then \"IMPRESSION: <N> lesions, of which the largest is <size>, "
                        "<seg>.\""),
    "non_lesion/length": ("count, size", "A liver length beside one lesion: \"Liver length is <x> cm.\" "
                          "(not a lesion size)."),
    "non_lesion/duct": ("count, size", "\"The common hepatic duct measures <x> cm.\" beside one lesion."),
    "non_lesion/node": ("count, size", "\"Near the liver, a lymph node of <x> cm is noted.\" beside one lesion."),
}
SPECIAL = {
    "real": "Real Train reports (liver-only text), unchanged; targets come from the source labels.",
    "ocr": "Character noise (swapped, dropped or replaced letters, digit/letter confusions) applied to real "
           "Train reports and to generated items of the families above (`ocr` subtype = the family it was applied to).",
}


def main() -> None:
    counts: dict[str, Counter] = {name: Counter() for name in SETS}
    examples: dict[str, list[str]] = {}
    for name in SETS:
        for item in read_jsonl(DATA / f"{name}.jsonl"):
            family = item["category"] if item["category"] in ("real", "ocr") else f"{item['category']}/{item['subtype']}"
            counts[name][family] += 1
            if family in FAMILIES and len(examples.setdefault(family, [])) < EXAMPLES:
                examples[family].append(item["inserted_text"])
    missing = (set(counts[SETS[0]]) | set(counts[SETS[1]])) - set(FAMILIES) - set(SPECIAL)
    if missing:
        raise SystemExit(f"families without a description: {sorted(missing)}")

    data = {
        "version": "extraction_train_aug_taxonomy_v1",
        "scope": "6.2 Train-aug only. No Challenge Set family is listed or reserved.",
        "families": {f: {"fields": fields, "construction": text, "examples": examples.get(f, []),
                         **{n: counts[n][f] for n in SETS}} for f, (fields, text) in FAMILIES.items()},
        "special": {f: {"description": d, **{n: counts[n][f] for n in SETS}} for f, d in SPECIAL.items()},
        "inputs_sha256": {(DATA / f"{n}.jsonl").as_posix(): sha256_file(DATA / f"{n}.jsonl") for n in SETS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 6.2 Train-aug phrasing taxonomy",
        "",
        "Every phrasing family the 6.2 training data contains (`extraction_train_v1` = first model, "
        "`extraction_train_lesion_v1` = per-lesion model; parser v1.2 rules were written from the same data). "
        "Nothing outside Train-aug is listed. Generated with `build_taxonomy.py`; edit the script, not this file.",
        "",
        "## Using it to mark Challenge-B items",
        "",
        "Open this file only after the Challenge-B items are frozen (their hash recorded). For each item, write the "
        "family id whose construction the item uses for the field it tests, or `new`:",
        "",
        "- Same construction with other values (numbers, segments, sizes, attenuation, word order of the "
        "surrounding report) = that family.",
        "- A construction this list does not describe = `new`, even if the words are close.",
        "- If an item uses two constructions, mark the one for the field it tests; if unsure, `new`.",
        "",
        "## Families",
        "",
        "| Family | Fields it carries | Construction | train_v1 | train_lesion_v1 |",
        "|---|---|---|---|---|",
        *[f"| `{f}` | {fields} | {text} | {counts[SETS[0]][f]} | {counts[SETS[1]][f]} |"
          for f, (fields, text) in FAMILIES.items()],
        *[f"| `{f}` | all | {d} | {counts[SETS[0]][f]} | {counts[SETS[1]][f]} |" for f, d in SPECIAL.items()],
        "",
        "## Examples from the data",
        "",
    ]
    for family in FAMILIES:
        lines += [f"**`{family}`**", "", *[f"- {e}" for e in examples.get(family, [])], ""]
    lines += ["## Inputs", "", *[f"- `{k}`: `{v}`" for k, v in data["inputs_sha256"].items()], ""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
