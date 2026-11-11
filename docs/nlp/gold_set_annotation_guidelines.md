# Gold Set Annotation Guidelines

These guidelines are for the people who fill `annotation_sheet_A.csv` and
`annotation_sheet_B.csv` in `data/processed/nlp/gold_set/`.

## The question you answer

For each report, record **what the report states** about liver lesions. Do not
use any other source: no CT images, no dataset columns, no knowledge of how the
report was produced. If the report does not state something, say so with
`UNDOCUMENTED`; never estimate it.

## Rules

- Work **independently**. Do not discuss cases with the other annotator until
  both sheets are complete.
- Do not open `gold_set_key.csv` or the source dataset. They contain the
  labels you are checking.
- Fill every row. Do not change `BDMAP ID` or `report_text`, and do not
  reorder or delete rows.
- Use your own stable `reviewer_id` (for example your initials) on every row.
- `reviewed_at` is an ISO-8601 time, for example `2026-11-12T14:30:00+02:00`.

## Fields

| Column | Allowed values | Meaning |
|---|---|---|
| `review_presence` | `YES`, `NO`, `UNCLEAR` | Does the report state at least one liver lesion (lesion, mass, nodule or tumor in the liver)? `NO` only when the report says there is none **or** describes the liver as normal / unremarkable without any lesion. `UNCLEAR` when the wording is genuinely ambiguous (for example "cannot be excluded"); a clinician will decide. |
| `reviewed_count` | integer ≥ 1, `UNDOCUMENTED`; `0` when presence is `NO` | Number of liver lesions stated. "Multiple" or "several" without a number is `UNDOCUMENTED`. "A total of 25 lesions" is `25`. |
| `reviewed_location` | Couinaud segments `1`–`8`, comma-separated (`5,8`); `UNDOCUMENTED`; `NOT_APPLICABLE` when `NO` | Segments stated for the **largest** liver lesion only. |
| `reviewed_diameter_cm` | number in cm; `UNDOCUMENTED`; `NOT_APPLICABLE` when `NO` | Largest stated dimension of the **largest** liver lesion. "3.1 x 1.6 cm" is `3.1`. Convert mm to cm (`12 mm` is `1.2`). |
| `reviewed_attenuation` | `hypoattenuating`, `isoattenuating`, `hyperattenuating`, `UNDOCUMENTED`; `NOT_APPLICABLE` when `NO` | Attenuation stated for the largest liver lesion. If only a group of lesions is described ("multiple hypoattenuating lesions"), use that value. |
| `evidence_quote` | text | Copy the shortest verbatim phrase from the report that supports your presence decision. |
| `review_notes` | text, required | Anything unusual: ambiguity, conflicting sentences, lesions in other organs you deliberately ignored. Write `ok` if nothing applies. |
| `reviewer_id` | text, required | Your stable ID. |
| `reviewed_at` | ISO-8601 time, required | When you completed the row. |

## Common cases

- **Lesions in other organs** (pancreas, kidney) are not liver lesions. Ignore
  their sizes and attenuation; note them in `review_notes` if they could confuse.
- **Findings and impression disagree**: use the more specific statement and
  explain in `review_notes`.
- **Largest lesion not identified**: if the report gives several lesions but
  does not say which is largest, use the one with the largest stated dimension
  and say so in `review_notes`.
- **Cysts and other benign findings** described as liver lesions count as
  lesions; record what the report calls them in `review_notes`.

## After both sheets are complete

1. `python scripts/nlp/gold_set.py agree` computes agreement (Cohen's kappa)
   and writes `adjudication_sheet.csv`. Rows where you both agree on every field
   are filled in as consensus.
2. A clinician completes the rows marked `needs_clinician = YES`, using the
   same fields plus their own `reviewer_id`, `reviewed_at` and `review_notes`.
   The sheet shows both annotations side by side, but not the source labels.
3. `python scripts/nlp/gold_set.py finalize` and then `apply` write
   `data/processed/nlp/gold_standard.jsonl`.
4. `python scripts/nlp/reliability_report.py` scores both systems against the
   gold standard.

## How the sample was drawn

About 230 Validation and Test studies, sampled per split from five strata:
every study whose source label conflicts with its text, plus fixed numbers of
single-lesion, multi-lesion, explicitly negative and not-stated negative
studies. The strata deliberately over-represent hard cases, so gold-set
metrics are not population estimates unless weighted by the
`inclusion_probability` in the key.

Test-split gold labels are for final reporting only. Use Validation gold labels
for parser and model development.
