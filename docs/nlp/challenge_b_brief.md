# Challenge-B: brief for the author

Challenge-B is the held-out test of liver lesion extraction. It is used **once**, in Phase 8, to choose between the parsers, the models and the hybrid. To keep it independent, one team member writes it alone, from this brief only.

## 1. Independence

Before you start:

- **Do not look at:**
  - the 6.2 training data (`data/processed/nlp/extraction/`);
  - the Train-aug taxonomy (`docs/nlp/extraction_train_aug_taxonomy.*`);
  - the parser code or rules (`src/liversira_ai/nlp/parser_*.py`);
  - the Challenge Sets v1–v3 (`data/processed/nlp/challenge/`, `results/nlp/transformers/challenge/`);
  - any model prediction.
- **If you have already seen any of them**, say so when you freeze. This is recorded; it does not disqualify you.
- **Never run any parser or model** on your items, and do not show your items to anyone who runs them. The only tool you run is the validator.
- **Do not use Test reports** in any way, as a base or as a source of wording. A Train or Validation report may serve as the surrounding text; put its study id in `base_id`.

## 2. What each item is

An item is the **liver part of a CT report** in English: a few sentences to a paragraph, written the way a radiologist might write it. It may mention other organs; the labels are about the liver only.

There are three categories:

| Category | Items | What it is | `targets` |
|---|---|---|---|
| `clean` | **60** | A plain, routine description of one or more liver lesions, where all four values are stated in ordinary wording. | all four fields |
| `negative` | **60** | A liver with no lesion. The text **must say so** ("no focal liver lesion", "the liver is unremarkable"); a report that simply never mentions lesions is out of scope. | `["count"]` |
| `targeted` | **280** | A description written to test how a field is phrased: counts, segments, sizes, attenuation, which lesion is the largest, measurements that are not lesions, and so on. | the field(s) the item tests |

The choice of phrasings is yours. That is the point: write the ways you expect real reports to say these things. Group your targeted items into **subtypes**: your own short `snake_case` names, each with **at least 10 items**. Each field must be tested by **at least 80** targeted items, and one item may test more than one field. Include multi-lesion reports. Sizes should range from a few millimetres to about 20 cm.

## 3. Labels (exact label space)

Every item has all four labels, even for fields it does not test. Label what the **text states**; never guess from medical knowledge.

| Field | Allowed values | Rules |
|---|---|---|
| `count` | `"0"` … `"9"`, `"10+"`, or `null` | The number of liver lesions, when the text gives it: a number ("three", "3"), "single / solitary / only one", or lesions listed one by one so they can be counted. "Multiple / several / numerous" with no number is `null`. `negative` items are `"0"`. |
| `segments` | Sorted list of integers 1–8, or `null` | The Couinaud segments of the **largest** lesion, as written (digits, Roman numerals, words). Subsegments 4a / 4b are `4`. A lobe name only, the caudate lobe included, is `null`. |
| `attenuation` | `"hypoattenuating"`, `"isoattenuating"`, `"hyperattenuating"`, or `null` | The attenuation of the **largest** lesion, in whatever words the text uses ("hypodense", "low attenuation", "darker than the liver" are all hypoattenuating). `null` if not stated for the largest lesion. |
| `diameter_cm` | Number in cm with at most one decimal (0.1–30), or `null` | The longest dimension of the **largest** lesion. Millimetres are divided by 10. `null` if no size is stated for it. |

**The largest lesion** is the one the text says is largest. If the text does not say, it is the one with the longest stated dimension. Sizes of things that are not liver lesions (the liver itself, vessels, ducts, lymph nodes, other organs) are never lesion sizes. A `null` in a targeted field is allowed: it tests that a system leaves an unstated value empty.

When `count` is `"0"`, the other three labels are `null`.

## 4. File format

Write one JSON object per line to `data/processed/nlp/challenge/challenge_b/challenge_b_items.jsonl`. This folder is not tracked by git; keep it out of chats and shared drives that candidate runners can open.

```json
{"id": "B0001", "category": "targeted", "subtype": "count_in_words", "targets": ["count"],
 "text": "...", "labels": {"count": "3", "segments": [6], "attenuation": "hypoattenuating", "diameter_cm": 4.2},
 "base_id": null, "notes": "optional, free text"}
```

- `id`: `B0001` … `B0400`, unique.
- `base_id`: `null`, or the Train / Validation study id whose report you used as the surrounding text.
- No two items may have the same text.

## 5. Freezing (two steps; neither can be undone)

**Step 1, items.** When all 400 items are written, run:

```
python -m scripts.nlp.transformers.extraction.validate_challenge_b items --author "<your initials>" --seen-train-aug no --seen-parser-rules no --seen-challenge-sets no
```

Answer each `--seen-…` flag truthfully. The validator checks the format, the label space, the numbers above and the `base_id` rule. It prints item ids, never text. When everything passes, it records the file's hash in `results/nlp/phase8/challenge_b_freeze.json`. Commit that file; it holds no item text. From this point **the items file must not change**.

**Step 2, coverage.** Now open `docs/nlp/extraction_train_aug_taxonomy.md`. For every item, decide whether the construction it uses for the field it tests is one of the Train-aug families listed there (write the family id) or not (write `new`). The taxonomy explains how to decide. Write one line per item to `challenge_b_coverage.jsonl` in the same folder:

```json
{"id": "B0001", "coverage": "count/present"}
{"id": "B0002", "coverage": "new"}
```

Then run:

```
python -m scripts.nlp.transformers.extraction.validate_challenge_b coverage
```

This checks that the items are unchanged and adds the coverage hash. Commit the freeze file again. Challenge-B is now complete.

## 6. After the freeze

Nobody edits either file. Phase 8 checks both hashes before it runs (`validate_challenge_b check`) and stops if anything changed. The analysis that will be run is fixed in `docs/nlp/phase8_analysis_plan.md`. You may read that plan, since it describes no phrasings.
