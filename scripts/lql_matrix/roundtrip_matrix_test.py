# roundtrip_matrix_test.py
import json

import roundtrip_matrix as M


def test_registry_defaults_to_smol135_only():
    active = M.active_variants()
    ids = {a[0] for a in active}
    assert ids == {"smol135"}
    variants = {a[1] for a in active}
    assert variants == {"instruct-bf16", "base-f32"}
    # everything else deactivated but present (just-works-on-re-add)
    assert any(m["id"] == "bitnet2b" and m["active"] is False for m in M.REGISTRY)


def test_enumerate_comparisons_covers_lattice():
    rows = M.enumerate_comparisons("smol135", "instruct-bf16")
    comps = {r["comparison"] for r in rows}
    assert {"input_vs_A", "lqlA_vs_cliA", "B_vs_A", "lqlB_vs_cliB"} <= comps
    # B rows carry an insert_form; A rows do not
    assert all(r["insert_form"] in ("knn", "compose")
               for r in rows if r["mode"] == "B")
    assert all(r["insert_form"] is None for r in rows if r["mode"] == "A")


def test_diff_model_dirs_combines_structural_and_value(tmp_path):
    import numpy as np, struct, json as _json
    da = tmp_path / "a"; db = tmp_path / "b"; da.mkdir(); db.mkdir()
    (da / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    (db / "config.json").write_text('{"architectures": ["Gemma3ForCausalLM"]}')

    def _st(path, dt, arr):
        raw = arr.tobytes()
        hdr = {"w": {"dtype": dt, "shape": list(arr.shape), "data_offsets": [0, len(raw)]}}
        blob = _json.dumps(hdr).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw)

    _st(da / "model.safetensors", "F32", np.array([1.0, 2.0], dtype=np.float32))
    _st(db / "model.safetensors", "BF16", np.array([0x3F80, 0x4000], dtype="<u2"))

    out = M.diff_model_dirs(str(da), str(db))
    assert out["manifest"]["bijective"] is True
    assert out["files"]["config.json"]["json"]["changed"]["architectures"] == \
        [["LlamaForCausalLM"], ["Gemma3ForCausalLM"]]
    st = out["files"]["model.safetensors"]
    assert st["sha256_equal"] is False
    assert st["header"]["dtype_changes"] == {"w": ["F32", "BF16"]}
    assert st["values"]["w"]["max_abs_diff"] == 0.0


def test_to_rows_carries_meta_and_measurements_no_verdicts():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": False,
                    "header": {"dtype_changes": {"w": ["F32", "BF16"]}, "order_equal": True,
                               "metadata_equal": False, "tensor_only_a": [], "tensor_only_b": [],
                               "shape_changes": {}},
                    "values": {"w": {"comparable": True, "n_total": 2, "n_differing": 0,
                                     "max_abs_diff": 0.0, "l2": 0.0}, "_bytes_equal": False}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A",
            "in_format_eq_out_format": False}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "model.safetensors"][0]
    assert r["model"] == "smol135" and r["comparison"] == "input_vs_A"
    assert r["sha256_equal"] is False
    assert r["in_format_eq_out_format"] is False
    # measured-only: no interpretation fields
    for banned in ("expected", "matches_expected", "cause", "verdict", "category"):
        assert banned not in r


def test_to_rows_propagates_header_and_value_errors():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": True,
                    "header": {"error_a": "bad hdr", "error_b": None},
                    "values": {"error": "bad val"}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "model.safetensors"][0]
    assert r["header_error_a"] == "bad hdr"
    assert r["values_error"] == "bad val"


def test_to_rows_propagates_json_error():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"config.json": {
                    "sha256_equal": False,
                    "json": {"error_a": "bad json"}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "config.json"][0]
    assert r["json_error_a"] == "bad json"


def test_to_rows_emits_l2_and_ntotal():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": False,
                    "header": {"dtype_changes": {}, "order_equal": True,
                               "metadata_equal": True, "tensor_only_a": [], "tensor_only_b": [],
                               "shape_changes": {}},
                    "values": {"w": {"comparable": True, "n_total": 4, "n_differing": 1,
                                     "max_abs_diff": 0.5, "l2": 0.7}, "_bytes_equal": False}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "model.safetensors"][0]
    assert r["values_l2_total"] == 0.7
    assert r["values_n_total_total"] == 4


def test_render_markdown_is_plain_table():
    rows = [{"model": "smol135", "variant": "base-f32", "comparison": "input_vs_A",
             "file": "model.safetensors", "sha256_equal": False}]
    md = M.render_markdown(rows)
    assert "model.safetensors" in md and "|" in md


def test_to_rows_carries_other_file_sizes():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"merges.txt": {"sha256_equal": False,
                                          "size_a": 100, "size_b": 120}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "merges.txt"][0]
    assert r["size_a"] == 100
    assert r["size_b"] == 120


def test_main_degrades_on_missing_dir(tmp_path):
    rc = M.main(["--a", str(tmp_path / "nope"), "--b", str(tmp_path / "nope2"),
                 "--meta", "{}", "--out", str(tmp_path / "o.jsonl")])
    assert isinstance(rc, int) and rc != 0
    lines = (tmp_path / "o.jsonl").read_text(encoding="utf-8").strip().splitlines()
    recs = [json.loads(l) for l in lines]
    assert any("error" in r for r in recs)


def test_main_degrades_on_malformed_meta(tmp_path):
    dir_a = tmp_path / "a"; dir_b = tmp_path / "b"
    dir_a.mkdir(); dir_b.mkdir()
    rc = M.main(["--a", str(dir_a), "--b", str(dir_b),
                 "--meta", "not-json", "--out", str(tmp_path / "o.jsonl")])
    assert isinstance(rc, int) and rc != 0
    lines = (tmp_path / "o.jsonl").read_text(encoding="utf-8").strip().splitlines()
    recs = [json.loads(l) for l in lines]
    assert any("error" in r for r in recs)


def _st_header(**over):
    h = {"dtype_changes": {}, "shape_changes": {}, "order_equal": True,
         "metadata_equal": True, "metadata_a": None, "metadata_b": None,
         "tensor_only_a": [], "tensor_only_b": []}
    h.update(over)
    return h


def test_to_rows_surfaces_incomparable_tensors():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": False, "header": _st_header(),
                    "values": {
                        "ok": {"comparable": True, "n_total": 2, "n_differing": 0,
                               "max_abs_diff": 0.0, "l2": 0.0},
                        "bad_shape": {"comparable": False, "shape_a": [2], "shape_b": [3]},
                        "bad_decode": {"comparable": False, "decode_error": "boom"},
                        "_bytes_equal": False}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    row = [r for r in M.to_rows(meta, dir_diff) if r.get("file") == "model.safetensors"][0]
    inc = row["values_incomparable"]
    assert set(inc) == {"bad_shape", "bad_decode"}
    assert inc["bad_shape"]["shape_a"] == [2] and inc["bad_shape"]["shape_b"] == [3]
    assert inc["bad_decode"]["decode_error"] == "boom"
    assert row["values_n_total_total"] == 2  # aggregates count only the comparable tensor


def test_to_rows_surfaces_differing_metadata():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": False,
                    "header": _st_header(metadata_equal=False,
                                         metadata_a={"format": "pt"}, metadata_b=None),
                    "values": {"_bytes_equal": False}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    row = [r for r in M.to_rows(meta, dir_diff) if r.get("file") == "model.safetensors"][0]
    assert row["header_metadata_a"] == {"format": "pt"}
    assert row["header_metadata_b"] is None


def test_to_rows_omits_metadata_values_when_equal():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": True, "header": _st_header(),
                    "values": {"_bytes_equal": True}}}}
    meta = {"model": "smol135", "variant": "x", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A"}
    row = [r for r in M.to_rows(meta, dir_diff) if r.get("file") == "model.safetensors"][0]
    assert "header_metadata_a" not in row
    assert row["values_incomparable"] == {}
