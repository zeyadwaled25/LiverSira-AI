# Parser v1.2 on 6.2 Train-aug

Parser v1.2.0 = v1.1 plus 14 rules written from Train-aug only (`extraction_train_v1`, `extraction_train_lesion_v1`). Challenge Sets, Validation and Test were not read. Presence logic is unchanged. v1.1 stays frozen.

## Rules and their evidence

Each rule has one Train-aug item on which v1.2 is right and v1.2 without that rule is wrong.

| Rule | What it does | Evidence item | Field | Target | v1.1 | v1.2 | v1.2 without rule |
|---|---|---|---|---|---|---|---|
| R1 | An elliptical clause that continues a lesion statement is part of it ("The largest, in segment 6, measures ...", "; size 8.2 x 6.8 cm"). | `aug:count:present:BDMAP_00005631:059723` | segments | [6] | None | [6] | None |
| R3 | "N in total" states the count. | `aug:count:in_total:BDMAP_00001937:826725` | count | 6 | None | 6 | None |
| R4 | A labelled list ("Lesion 1: ... Lesion 2: ...") is counted by its highest label. | `aug:count:listed:BDMAP_00002550:372864` | count | 5 | None | 5 | None |
| R5 | A number followed by ".<digit>" after a segment list is a size, not a segment. | `aug:count:listed:BDMAP_00002550:372864` | segments | [2, 8] | [2, 6, 8] | [2, 8] | [2, 6, 8] |
| R6 | "Another lesion" and "A further lesion" each add one lesion. | `aug:count:another:BDMAP_00005815:239600` | count | 3 | 1 | 3 | 1 |
| R7 | Subsegments ("4A", "IVa") map to their segment. | `aug:segment:subsegment:BDMAP_00003845:231586` | segments | [4] | None | [4] | None |
| R8 | Ordinal and word segment names ("the second segment", "segment eight"). | `aug:segment:ordinal:BDMAP_00003333:473223` | segments | [2] | None | [2] | None |
| R9 | "segments 2 to 3" is a range of segments. | `aug:segment:range:BDMAP_00003237:113871` | segments | [2, 3] | [2] | [2, 3] | [2] |
| R10 | "of low/high/equal density" and "with low/high/equal attenuation". | `aug:attenuation:of_density:BDMAP_00004586:460124` | attenuation | isoattenuating | None | isoattenuating | None |
| R11 | Comparisons with the liver: "darker/less dense than", "brighter/denser than", "as dense as", "indistinguishable in density". | `aug:attenuation:relative:BDMAP_00000881:711770` | attenuation | hyperattenuating | None | hyperattenuating | None |
| R12 | A sentence about the previous lesion's attenuation ("It appears hyperattenuating.", "Its attenuation is low."). | `aug:attenuation:separate:BDMAP_00001252:554569` | attenuation | hyperattenuating | None | hyperattenuating | None |
| R13 | "focus" and "foci" are lesion nouns. | `aug:attenuation:appearing:BDMAP_00008962:339767` | count | 1 | None | 1 | None |
| R14 | A bare plural with no unquantified word ("lesions are seen in segment 1 (1.3 x 1.1 cm) and ...") is counted from the sizes it lists. | `aug:order:size_list:BDMAP_00005472:950414` | count | 2 | None | 2 | None |
| R15 | "the larger of the two" states two lesions. | `aug:order:larger_of_two:BDMAP_00007427:855788` | count | 2 | 1 | 2 | 1 |

R2 (more words for the largest lesion) was written and dropped: switching it off changed no Train-aug item.

Real Train reports (6,482 in each set): field-level changes v1.1 → v1.2: 2 better, 0 worse.

## Accuracy by Train-aug family

Cells: v1.1 → v1.2 (bold where it changed). Almost all errors left in generated families have presence unknown: the inserted lesion sentence names no organ and follows a base sentence about several organs, so the unchanged presence logic does not attribute it to the liver. `real` count is low for both because a report that never mentions a liver lesion stays unknown, not 0.

### extraction_train_v1

| Family | n | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|---|
| attenuation/appearing | 125 | 0.432 → **0.976** | 0.432 → **0.976** | 0.432 → **0.976** | 0.432 → **0.976** |
| attenuation/of_density | 125 | 0.960 | 0.960 | 0.000 → **0.960** | 0.960 |
| attenuation/relative | 125 | 1.000 | 1.000 | 0.000 → **1.000** | 1.000 |
| attenuation/separate | 125 | 0.976 | 0.976 | 0.000 → **0.976** | 0.976 |
| count/another | 83 | 0.000 → **0.940** | 0.940 | 0.940 | 0.940 |
| count/in_total | 84 | 0.000 → **1.000** | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| count/listed | 83 | 0.000 → **1.000** | 0.181 → **1.000** | 1.000 | 1.000 |
| count/only_one | 83 | 0.964 | 0.964 | 0.964 | 0.964 |
| count/present | 84 | 1.000 | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| count/vague | 83 | 1.000 | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| ocr | 1000 | 0.683 → **0.787** | 0.627 → **0.739** | 0.701 → **0.763** | 0.745 → **0.840** |
| order/larger_of_two | 167 | 0.000 → **0.964** | 0.060 → **0.964** | 0.270 → **0.964** | 0.000 → **0.964** |
| order/size_list | 167 | 0.000 → **0.958** | 0.958 | 0.958 | 0.958 |
| order/there_is | 166 | 0.000 → **0.940** | 0.940 | 0.940 | 0.940 |
| paraphrase/multiple | 250 | 0.536 → **1.000** | 0.000 → **1.000** | 0.536 → **1.000** | 0.000 → **1.000** |
| paraphrase/single | 250 | 0.584 → **0.948** | 0.584 → **0.948** | 0.584 → **0.948** | 0.304 → **0.948** |
| real | 6482 | 0.446 | 0.986 → **0.986** | 0.993 | 0.997 |
| segment/couinaud | 83 | 0.976 | 0.976 | 0.976 | 0.976 |
| segment/lobe | 83 | 0.964 | 1.000 | 0.964 | 0.964 |
| segment/ordinal | 83 | 0.976 | 0.000 → **0.976** | 0.976 | 0.976 |
| segment/range | 83 | 0.940 | 0.470 → **0.940** | 0.940 | 0.940 |
| segment/roman | 84 | 0.988 | 0.988 | 0.988 | 0.988 |
| segment/subsegment | 84 | 1.000 | 0.000 → **1.000** | 1.000 | 1.000 |
| size/cm_by_cm | 100 | 0.990 | 0.990 | 0.990 | 0.990 |
| size/greatest_diameter | 100 | 0.990 | 0.990 | 0.990 | 0.990 |
| size/reaching | 100 | 1.000 | 1.000 | 1.000 | 1.000 |
| size/single_mm | 100 | 1.000 | 1.000 | 1.000 | 1.000 |
| size/three_in_size | 100 | 0.950 | 0.950 | 0.950 | 0.950 |

### extraction_train_lesion_v1

| Family | n | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|---|
| attenuation/appearing | 125 | 0.536 → **0.992** | 0.536 → **0.992** | 0.536 → **0.992** | 0.536 → **0.992** |
| attenuation/of_density | 125 | 0.984 | 0.984 | 0.000 → **0.984** | 0.984 |
| attenuation/relative | 125 | 1.000 | 1.000 | 0.000 → **1.000** | 1.000 |
| attenuation/separate | 125 | 0.928 | 0.928 | 0.000 → **0.928** | 0.928 |
| count/another | 83 | 0.000 → **0.988** | 0.988 | 0.988 | 0.988 |
| count/in_total | 84 | 0.000 → **1.000** | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| count/listed | 83 | 0.000 → **0.964** | 0.084 → **0.964** | 0.964 | 0.964 |
| count/only_one | 83 | 0.964 | 0.964 | 0.964 | 0.964 |
| count/present | 84 | 1.000 | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| count/vague | 83 | 1.000 | 0.000 → **1.000** | 1.000 | 0.000 → **1.000** |
| non_lesion/duct | 167 | 0.952 | 0.952 | 0.952 | 0.952 |
| non_lesion/length | 167 | 0.976 | 0.976 | 0.976 | 0.976 |
| non_lesion/node | 166 | 0.982 | 0.982 | 0.982 | 0.982 |
| ocr | 1000 | 0.713 → **0.800** | 0.637 → **0.742** | 0.712 → **0.759** | 0.783 → **0.857** |
| order/larger_of_two | 167 | 0.000 → **0.976** | 0.060 → **0.976** | 0.270 → **0.976** | 0.000 → **0.976** |
| order/size_list | 167 | 0.000 → **0.964** | 0.964 | 0.964 | 0.964 |
| order/there_is | 166 | 0.000 → **0.964** | 0.964 | 0.964 | 0.964 |
| paraphrase/multiple | 250 | 0.500 → **1.000** | 0.000 → **1.000** | 0.500 → **1.000** | 0.000 → **1.000** |
| paraphrase/single | 250 | 0.672 → **0.980** | 0.672 → **0.980** | 0.672 → **0.980** | 0.320 → **0.980** |
| real | 6482 | 0.446 | 0.986 → **0.986** | 0.993 | 0.997 |
| repeat/multiple | 250 | 0.976 → **0.984** | 0.984 | 0.968 | 0.984 |
| repeat/single | 250 | 0.964 | 0.968 → **0.972** | 0.972 | 0.972 |
| segment/couinaud | 83 | 0.976 | 0.976 | 0.976 | 0.976 |
| segment/lobe | 83 | 0.988 | 1.000 | 0.988 | 0.988 |
| segment/ordinal | 83 | 0.976 | 0.000 → **0.976** | 0.976 | 0.976 |
| segment/range | 83 | 0.976 | 0.602 → **0.976** | 0.976 | 0.976 |
| segment/roman | 84 | 0.988 | 0.988 | 0.988 | 0.988 |
| segment/subsegment | 84 | 1.000 | 0.000 → **1.000** | 1.000 | 1.000 |
| size/cm_by_cm | 100 | 0.990 | 0.990 | 0.990 | 0.990 |
| size/greatest_diameter | 100 | 0.970 | 0.970 | 0.970 | 0.970 |
| size/reaching | 100 | 0.930 | 0.930 | 0.930 | 0.930 |
| size/single_mm | 100 | 0.980 | 0.980 | 0.980 | 0.980 |
| size/three_in_size | 100 | 0.980 | 0.980 | 0.980 | 0.980 |

## Code hashes

- `src/liversira_ai/nlp/parser_v1_1.py`: `1ececcd2969e44e2106f500cfd3ed7dc478afc664ba7ac2ec362c86b32fbb797`
- `src/liversira_ai/nlp/parser_v1_2.py`: `c4181fea66ebc9557c38931f9272f5bb6031f57cc13ff5e6a8e6d648c1ecf112`
- `src/liversira_ai/nlp/parser_v1_2_rules.py`: `9cbec7d7babcc589c13850dbdcfcef2065aa4eebf2dfec162cadc6b9ab9568f6`
- `src/liversira_ai/nlp/baseline.py`: `9d85bdfc40a812a152b40e3a8dadc393248ad6ada8ab83351c93cfec07b6747b`
