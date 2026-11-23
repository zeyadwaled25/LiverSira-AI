"""The Challenge-B validator: format and label-space checks, and that it loads no candidate."""

import copy
import subprocess
import sys

from scripts.nlp.transformers.extraction import validate_challenge_b as vb

TEST_IDS, KNOWN = {"BDMAP_T1"}, {"BDMAP_A1", "BDMAP_A2"}


def _row(i: int, category: str, subtype: str, targets: list[str], labels: dict) -> dict:
    size = labels["diameter_cm"]
    text = f"Item {i}. " + (f"A lesion of {size} cm." if size is not None else "No focal liver lesion.")
    return {"id": f"B{i:04d}", "text": text, "category": category, "subtype": subtype, "targets": targets,
            "labels": labels, "base_id": None}


def valid_set() -> list[dict]:
    rows, i = [], 1
    lesion = {"count": "2", "segments": [4, 8], "attenuation": "hypoattenuating", "diameter_cm": 3.5}
    for _ in range(60):
        rows.append(_row(i, "clean", "routine", list(vb.FIELDS), dict(lesion)))
        i += 1
    for _ in range(60):
        rows.append(_row(i, "negative", "stated_normal", ["count"],
                         {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None}))
        i += 1
    for k in range(280):
        field, other = vb.FIELDS[k % 4], vb.FIELDS[(k + 1) % 4]
        rows.append(_row(i, "targeted", f"probe_{field}", [field, other], dict(lesion)))
        i += 1
    return rows


def errors(rows: list[dict]) -> list[str]:
    return vb.validate_items(rows, TEST_IDS, KNOWN)[0]


def test_valid_set_passes():
    assert errors(valid_set()) == []
    assert vb.summary(valid_set())["targeted_items_per_field"] == {f: 140 for f in vb.FIELDS}


def test_label_space_is_enforced():
    rows = valid_set()
    bad = copy.deepcopy(rows)
    bad[0]["labels"]["count"] = "11"
    bad[1]["labels"]["segments"] = [8, 4]
    bad[2]["labels"]["attenuation"] = "hypodense"
    bad[3]["labels"]["diameter_cm"] = 3.55
    found = errors(bad)
    assert any("B0001: count" in e for e in found) and any("B0002: segments" in e for e in found)
    assert any("B0003: attenuation" in e for e in found) and any("B0004: diameter_cm" in e for e in found)


def test_category_rules_and_test_bases():
    bad = valid_set()
    bad[0]["labels"]["attenuation"] = None  # clean must state all four
    bad[60]["labels"]["count"] = "1"  # negative must be 0
    bad[200]["base_id"] = "BDMAP_T1"  # a Test report
    bad[201]["base_id"] = "BDMAP_X9"  # unknown id
    found = errors(bad)
    assert any(e.startswith("B0001: a clean item") for e in found)
    assert any(e.startswith("B0061: a negative item") for e in found)
    assert any("B0201: base_id is a Test study" in e for e in found)
    assert any("B0202: base_id 'BDMAP_X9'" in e for e in found)


def test_sizes_per_category_field_and_subtype():
    rows = valid_set()[:-1]  # 279 targeted
    assert any("targeted: 279 items" in e for e in errors(rows))
    rows = valid_set()
    for r in rows[120:]:
        r["targets"] = ["count"]
        r["subtype"] = "probe_count"
    assert any(e.startswith("segments: tested by 0") for e in errors(rows))
    rows = valid_set()
    rows[-1]["subtype"] = "lonely"
    assert any("targeted/lonely: 1 items" in e for e in errors(rows))


def test_duplicates_and_keys():
    rows = valid_set()
    rows[1]["id"] = rows[0]["id"]
    rows[2]["text"] = rows[3]["text"]
    del rows[4]["base_id"]
    found = errors(rows)
    assert any("used 2 times" in e for e in found) and any("repeat another item's text" in e for e in found)
    assert any(e.startswith("B0005: keys") for e in found)


def test_coverage_ids_and_values():
    ids = {"B0001", "B0002"}
    ok = [{"id": "B0001", "coverage": "count/present"}, {"id": "B0002", "coverage": "new"}]
    assert vb.validate_coverage(ok, ids, {"count/present"}) == []
    bad = [{"id": "B0001", "coverage": "count/imagined"}]
    found = vb.validate_coverage(bad, ids, {"count/present"})
    assert any("count/imagined" in e for e in found) and any("1 missing" in e for e in found)


def test_validator_loads_no_candidate():
    code = ("import sys; import scripts.nlp.transformers.extraction.validate_challenge_b; "
            "bad = [m for m in sys.modules if m.startswith(('torch', 'transformers', 'liversira_ai'))]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
