# 6.2 Train-aug phrasing taxonomy

Every phrasing family the 6.2 training data contains (`extraction_train_v1` = first model, `extraction_train_lesion_v1` = per-lesion model; parser v1.2 rules were written from the same data). Nothing outside Train-aug is listed. Generated with `build_taxonomy.py`; edit the script, not this file.

## Using it to mark Challenge-B items

Open this file only after the Challenge-B items are frozen (their hash recorded). For each item, write the family id whose construction the item uses for the field it tests, or `new`:

- Same construction with other values (numbers, segments, sizes, attenuation, word order of the surrounding report) = that family.
- A construction this list does not describe = `new`, even if the words are close.
- If an item uses two constructions, mark the one for the field it tests; if unsure, `new`.

## Families

| Family | Fields it carries | Construction | train_v1 | train_lesion_v1 |
|---|---|---|---|---|
| `count/present` | count, segments, size | "<N> <att> lesions are present in the liver." then a separate "The largest, in <seg>, measures <size>." (N as digits or a word). | 84 | 84 |
| `count/in_total` | count | "... lesions, <N> in total; the largest lies in <seg> with a size of <size>." | 84 | 84 |
| `count/listed` | count, order | Labelled list "Lesion 1: <seg>, <size>, <att>. Lesion 2: ..." (2-5 lesions, shuffled; count = number of labels). | 83 | 83 |
| `count/another` | count | "One <att> lesion is present in <seg> (<size>). Another lesion ... A further lesion ... Yet another lesion ..." (2-4 lesions; count = number of sentences). | 83 | 83 |
| `count/only_one` | count | "Only one <att> lesion is present ..." or "There is a lone <att> lesion ...". | 83 | 83 |
| `count/vague` | count (unstated) | "Innumerable / A few / Scattered / Various <att> lesions are present throughout the liver; the largest, in <seg>, has a size of <size>." Count is unstated. | 83 | 83 |
| `size/single_mm` | size | One dimension in millimetres: "A <att> lesion of <NN> mm lies in <seg>." | 100 | 100 |
| `size/cm_by_cm` | size | "<a> cm by <b> cm". | 100 | 100 |
| `size/greatest_diameter` | size | "... has a greatest diameter of <a> cm." | 100 | 100 |
| `size/three_in_size` | size | Three dimensions with the unit after each: "<a> cm x <b> cm x <c> cm in size". | 100 | 100 |
| `size/reaching` | size | "..., reaching <a> cm, ..." or "as large as <a> cm". | 100 | 100 |
| `segment/roman` | segments | Roman numerals: "in liver segment VI", "(segment VI)". | 84 | 84 |
| `segment/subsegment` | segments | "segment 4A/4B", "subsegment IVa/IVb" (= segment 4). | 84 | 84 |
| `segment/couinaud` | segments | "in segment <n> according to Couinaud". | 83 | 83 |
| `segment/ordinal` | segments | "in the 5th segment of the liver", "within segment five", "in the fifth segment of the liver". | 83 | 83 |
| `segment/range` | segments | "extending from segment <n> into segment <n+1>", "spanning segments <n> to <n+1>". | 83 | 83 |
| `segment/lobe` | segments (unstated) | Lobe only: "in the right/left liver lobe", "within the right/left lobe". Segments are unstated. | 83 | 83 |
| `attenuation/appearing` | attenuation | "hypodense / hyperdense / isodense": "A lesion appearing hypodense ...", "<Seg> contains a hypodense focus of <size>." | 125 | 125 |
| `attenuation/of_density` | attenuation | "of low/high/equal density", "with low/high/equal attenuation". | 125 | 125 |
| `attenuation/relative` | attenuation | Comparison with the liver: "darker/brighter than the adjacent hepatic tissue", "less dense than / denser than background liver", "as dense as ...", "indistinguishable in density from background liver". | 125 | 125 |
| `attenuation/separate` | attenuation | In its own sentence after the lesion: "It appears hypoattenuating.", "Its attenuation is low/high.", "Its attenuation matches the liver." | 125 | 125 |
| `order/size_list` | order, count | One sentence listing "<seg> (<size>)" for 2-4 lesions, ascending or shuffled: "<Att> lesions are seen in segment 1 (...) and segments 3/6 (...)." | 167 | 167 |
| `order/larger_of_two` | order, count | Smaller lesion first, then "The larger of the two, <att>, lies in <seg> with a size of <size>." | 167 | 167 |
| `order/there_is` | order, count | "There are lesions in <seg>, a <size> <att> lesion; in <seg>, a <size> <att> lesion; ..." (2-4, shuffled). | 166 | 166 |
| `paraphrase/single` | all | Other nouns and verbs for one lesion: "<Seg> is the site of a <size> <att> focus / tumor / space-occupying lesion.", "A <att> <noun> sits in <seg>; size <size>." | 250 | 250 |
| `paraphrase/multiple` | all | "The liver holds <N> <att> foci/tumors. The most prominent / greatest / principal of these sits in <seg>, sized <size>; the others are under <x> cm." | 250 | 250 |
| `repeat/single` | count, size | One lesion, restated in a tail: "There is a <att> lesion of <size> at <seg>. IMPRESSION: <att> lesion, <seg>, <size>." (count stays 1). | 0 | 250 |
| `repeat/multiple` | count, size | Listed lesions, then "IMPRESSION: <N> lesions, of which the largest is <size>, <seg>." | 0 | 250 |
| `non_lesion/length` | count, size | A liver length beside one lesion: "Liver length is <x> cm." (not a lesion size). | 0 | 167 |
| `non_lesion/duct` | count, size | "The common hepatic duct measures <x> cm." beside one lesion. | 0 | 167 |
| `non_lesion/node` | count, size | "Near the liver, a lymph node of <x> cm is noted." beside one lesion. | 0 | 166 |
| `real` | all | Real Train reports (liver-only text), unchanged; targets come from the source labels. | 6482 | 6482 |
| `ocr` | all | Character noise (swapped, dropped or replaced letters, digit/letter confusions) applied to real Train reports and to generated items of the families above (`ocr` subtype = the family it was applied to). | 1000 | 1000 |

## Examples from the data

**`count/present`**

- Seven hypoattenuating lesions are present in the liver. The largest, in segment 6, measures 6.3 x 3.3 cm.
- 3 hypoattenuating lesions are present in the liver. The largest, in segments 3/5, measures 8.3 x 5.0 cm.

**`count/in_total`**

- The liver shows hypoattenuating lesions, 6 in total; the largest lies in segment 1 with a size of 4.1 x 3.6 cm.
- The liver shows hypoattenuating lesions, 4 in total; the largest lies in segment 5 with a size of 5.1 x 4.4 cm.

**`count/listed`**

- Lesion 1: segment 3, 0.9 x 0.7 cm, hyperattenuating. Lesion 2: segment 6, 2.9 x 2.5 cm, hypoattenuating. Lesion 3: segments 2/8, 6.1 x 5.4 cm, hypoattenuating. Lesion 4: segment 6, 4.5 x 3.6 cm, hypoattenuating. Lesion 5: segment 4, 2.5 x 2.1 cm, hyperattenuating.
- Lesion 1: segment 2, 2.5 x 2.1 cm, hypoattenuating. Lesion 2: segment 4, 1.6 x 1.2 cm, hypoattenuating.

**`count/another`**

- One hypoattenuating lesion is present in segment 6 (4.0 x 3.0 cm). Another lesion is present in segment 1 (2.0 x 1.0 cm). A further lesion is present in segment 4 (1.5 x 0.9 cm).
- One hyperattenuating lesion is present in segments 2/7 (4.4 x 3.8 cm). Another lesion is present in segment 5 (1.2 x 0.6 cm).

**`count/only_one`**

- There is a lone hypoattenuating lesion in segment 7 of 7.7 x 6.3 cm.
- There is a lone hyperattenuating lesion in segment 1 of 8.7 x 4.6 cm.

**`count/vague`**

- Various hyperattenuating lesions are present throughout the liver; the largest, in segments 4/8, has a size of 3.3 x 2.9 cm.
- A few isoattenuating lesions are present throughout the liver; the largest, in segment 2, has a size of 5.8 x 3.7 cm.

**`size/single_mm`**

- A hypoattenuating lesion of 48 mm lies in segment 5.
- An isoattenuating lesion of 80 mm lies in segments 2/6.

**`size/cm_by_cm`**

- An isoattenuating lesion of 2.9 cm by 2.0 cm is present in segment 7.
- A hypoattenuating lesion of 2.6 cm by 1.5 cm is present in segments 3/4.

**`size/greatest_diameter`**

- A hypoattenuating lesion located at segment 4 has a greatest diameter of 3.2 cm.
- A hypoattenuating lesion located at segment 8 has a greatest diameter of 6.7 cm.

**`size/three_in_size`**

- There is a hyperattenuating lesion at segment 2, 6.6 cm x 4.7 cm x 2.0 cm in size.
- There is a hypoattenuating lesion at segment 2, 4.9 cm x 4.4 cm x 1.9 cm in size.

**`size/reaching`**

- A hyperattenuating lesion at segments 1/8, reaching 8.3 cm, is present.
- A hyperattenuating lesion at segment 2, reaching 3.7 cm, is present.

**`segment/roman`**

- A hyperattenuating lesion of 4.2 x 3.8 cm is present in liver segment II.
- A hypoattenuating lesion of 7.7 x 6.8 cm is present (segment I).

**`segment/subsegment`**

- A hypoattenuating lesion of 3.6 x 2.0 cm is present in subsegment IVa.
- A hypoattenuating lesion of 4.0 x 2.9 cm is present in subsegment IVa.

**`segment/couinaud`**

- A hypoattenuating lesion of 4.0 x 2.2 cm is present in segment 2 according to Couinaud.
- An isoattenuating lesion of 3.0 x 1.7 cm is present in segment 4 according to Couinaud.

**`segment/ordinal`**

- A hyperattenuating lesion of 2.3 x 2.0 cm is present within segment eight.
- A hypoattenuating lesion of 8.0 x 4.0 cm is present in the 1st segment of the liver.

**`segment/range`**

- A hyperattenuating lesion of 7.3 x 4.0 cm is present extending from segment 3 into segment 4.
- A hyperattenuating lesion of 3.0 x 2.6 cm is present extending from segment 6 into segment 7.

**`segment/lobe`**

- A hypoattenuating lesion of 8.4 x 5.2 cm is present within the left lobe.
- A hypoattenuating lesion of 9.0 x 6.8 cm is present in the right liver lobe.

**`attenuation/appearing`**

- A lesion appearing isodense is present in segments 1/5, with a size of 4.1 x 3.1 cm.
- Segments 4/5 contains a hyperdense focus of 4.9 x 2.7 cm.

**`attenuation/of_density`**

- A lesion of equal density is present in segment 6, with a size of 3.4 x 2.3 cm.
- A lesion of high density is present in segments 1/7, with a size of 2.1 x 1.4 cm.

**`attenuation/relative`**

- In segment 3 there is a 5.0 x 3.6 cm lesion, less dense than background liver.
- In segment 5 there is a 3.8 x 2.5 cm lesion, brighter than the adjacent hepatic tissue.

**`attenuation/separate`**

- There is a lesion in segments 3/6 with a size of 5.3 x 4.5 cm. It appears hyperattenuating.
- There is a lesion in segment 4 with a size of 2.7 x 2.1 cm. Its attenuation is low.

**`order/size_list`**

- Hypoattenuating lesions are seen in segment 1 (1.3 x 1.1 cm) and segments 3/6 (2.3 x 1.6 cm).
- Hypoattenuating lesions are seen in segment 4 (1.9 x 1.0 cm), segment 3 (3.3 x 2.0 cm), and segment 7 (1.1 x 0.8 cm).

**`order/larger_of_two`**

- An isoattenuating lesion of 3.3 x 2.6 cm lies in segment 8. The larger of the two, hyperattenuating, lies in segment 4 with a size of 5.8 x 3.7 cm.
- A hypoattenuating lesion of 4.7 x 2.6 cm lies in segment 4. The larger of the two, isoattenuating, lies in segment 5 with a size of 5.9 x 3.2 cm.

**`order/there_is`**

- There are lesions in segments 2/4, a 7.1 x 3.7 cm hypoattenuating lesion; in segment 1, a 2.4 x 2.0 cm hyperattenuating lesion; in segment 2, a 5.3 x 3.0 cm hyperattenuating lesion.
- There are lesions in segment 5, a 0.4 x 0.3 cm hyperattenuating lesion; in segment 5, a 1.6 x 0.9 cm hypoattenuating lesion.

**`paraphrase/single`**

- A hyperattenuating space-occupying lesion sits in segments 2/7; size 8.2 x 6.8 cm.
- A hypoattenuating focus sits in segment 2; size 7.2 x 5.9 cm.

**`paraphrase/multiple`**

- The liver holds two hyperattenuating foci. The most prominent of these sits in segment 6, sized 7.6 x 6.1 cm; the others are under 7.2 cm.
- The liver holds four isoattenuating tumors. The greatest of these sits in segments 2/8, sized 5.7 x 3.4 cm; the others are under 5.3 cm.

**`repeat/single`**

- There is a hyperattenuating lesion of 8.3 x 5.2 cm at segment 2. IMPRESSION: hyperattenuating lesion, segment 2, 8.3 x 5.2 cm.
- There is a hypoattenuating lesion of 8.9 x 4.5 cm at segment 5. IMPRESSION: hypoattenuating lesion, segment 5, 8.9 x 4.5 cm.

**`repeat/multiple`**

- Hypoattenuating lesions are seen in segments 3/6 (3.2 x 2.3 cm) and segment 1 (2.0 x 1.6 cm). IMPRESSION: 2 lesions, of which the largest is 3.2 x 2.3 cm, segments 3/6.
- Hyperattenuating lesions are seen in segments 5/6 (2.5 x 1.7 cm), segment 1 (2.0 x 1.0 cm), segment 7 (1.2 x 1.0 cm), and segment 5 (0.4 x 0.3 cm). IMPRESSION: 4 lesions, of which the largest is 2.5 x 1.7 cm, segments 5/6.

**`non_lesion/length`**

- Liver length is 17.3 cm. Segment 8 contains a hyperattenuating lesion of 3.7 x 2.9 cm.
- Liver length is 18.5 cm. Segment 6 contains a hypoattenuating lesion of 3.0 x 1.5 cm.

**`non_lesion/duct`**

- Segment 7 contains a hypoattenuating lesion of 3.4 x 2.2 cm. The common hepatic duct measures 0.9 cm.
- Segments 3/7 contains a hyperattenuating lesion of 6.7 x 5.4 cm. The common hepatic duct measures 0.9 cm.

**`non_lesion/node`**

- Near the liver, a lymph node of 13.7 cm is noted. Segment 3 contains an isoattenuating lesion of 8.8 x 7.4 cm.
- Near the liver, a lymph node of 6.1 cm is noted. Segments 4/8 contains a hyperattenuating lesion of 3.6 x 2.8 cm.

## Inputs

- `data/processed/nlp/extraction/extraction_train_v1.jsonl`: `d794e543934e3bbd49e337886aeb9118ddbb825702607053df4fbed86d4f2be5`
- `data/processed/nlp/extraction/extraction_train_lesion_v1.jsonl`: `23e4e8755aace28572601d1c148403d04b36a104bf072726918dd892d1824bab`
