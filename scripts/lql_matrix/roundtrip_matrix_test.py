# roundtrip_matrix_test.py
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
