# roundtrip_aggregate_test.py
import json

import roundtrip_aggregate as A


def test_load_flattens_comparisons_and_tags_leg(tmp_path):
    doc = {"leg": "smol135.native.inference", "hf": "x", "produce": [],
           "comparisons": [{"comparison": "input_vs_lql.passthrough",
                            "file": "_manifest", "bijective": True}]}
    (tmp_path / "roundtrip-smol135.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = A.load(str(tmp_path / "roundtrip-*.json"))
    assert len(rows) == 1
    assert rows[0]["leg"] == "smol135.native.inference"
    assert rows[0]["comparison"] == "input_vs_lql.passthrough"


def test_load_includes_missing_dir_rows(tmp_path):
    doc = {"leg": "leg2", "comparisons": [
        {"comparison": "input_vs_cli.passthrough", "file": "_manifest",
         "bijective": False, "dir_a_exists": True, "dir_b_exists": False}]}
    (tmp_path / "roundtrip-leg2.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = A.load(str(tmp_path / "roundtrip-*.json"))
    assert len(rows) == 1
    assert rows[0]["dir_b_exists"] is False
    assert rows[0]["leg"] == "leg2"


def test_load_skips_malformed_json(tmp_path):
    (tmp_path / "roundtrip-bad.json").write_text("{not json", encoding="utf-8")
    good = {"leg": "ok", "comparisons": [{"comparison": "c", "file": "f"}]}
    (tmp_path / "roundtrip-good.json").write_text(json.dumps(good), encoding="utf-8")
    rows = A.load(str(tmp_path / "roundtrip-*.json"))
    assert len(rows) == 1
    assert rows[0]["leg"] == "ok"


def test_load_skips_leg_error_doc_without_comparisons(tmp_path):
    doc = {"leg": "errleg", "error": "produce crashed"}
    (tmp_path / "roundtrip-errleg.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = A.load(str(tmp_path / "roundtrip-*.json"))
    assert rows == []


def test_main_writes_json_and_markdown(tmp_path):
    doc = {"leg": "leg1", "comparisons": [
        {"comparison": "input_vs_lql.passthrough", "driver": "lql",
         "mode": "passthrough", "insert_form": None, "file": "_manifest",
         "bijective": True}]}
    (tmp_path / "roundtrip-leg1.json").write_text(json.dumps(doc), encoding="utf-8")
    out_json = tmp_path / "cat.json"
    out_md = tmp_path / "cat.md"
    A.main(str(tmp_path / "roundtrip-*.json"), str(out_json), str(out_md))
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["leg"] == "leg1"
    md = out_md.read_text(encoding="utf-8")
    assert "|" in md and "model.safetensors" not in md
