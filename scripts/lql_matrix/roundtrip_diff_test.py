import os, hashlib
import struct, json as _json
import roundtrip_diff as D

def test_file_manifest_hashes_and_sizes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "b.json").write_bytes(b"{}")
    man = D.file_manifest(str(tmp_path))
    assert set(man) == {"a.bin", "b.json"}
    assert man["a.bin"]["size"] == 5
    assert man["a.bin"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert man["b.json"]["size"] == 2

def test_manifest_bijection_reports_orphans():
    a = {"model.safetensors": {}, "config.json": {}, "lm_head.safetensors": {}}
    b = {"model.safetensors": {}, "config.json": {}}
    bij = D.manifest_bijection(a, b)
    assert bij["only_a"] == ["lm_head.safetensors"]
    assert bij["only_b"] == []
    assert bij["in_both"] == ["config.json", "model.safetensors"]
    assert bij["bijective"] is False

def _write_safetensors(path, header):
    blob = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 64)  # dummy data buffer

def test_read_safetensors_header_parses_tensors_metadata_order(tmp_path):
    p = tmp_path / "model.safetensors"
    _write_safetensors(str(p), {
        "b.weight": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "a.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
        "__metadata__": {"format": "pt"},
    })
    h = D.read_safetensors_header(str(p))
    assert h["tensors"]["a.weight"]["dtype"] == "BF16"
    assert h["metadata"] == {"format": "pt"}
    assert h["order"] == ["a.weight", "b.weight"]  # by data_offsets[0]

def test_read_safetensors_header_malformed_returns_error(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"\x02\x00")  # too short for u64 length
    h = D.read_safetensors_header(str(p))
    assert "error" in h

def test_read_safetensors_header_non_dict_header_returns_error(tmp_path):
    p = tmp_path / "list_header.safetensors"
    blob = _json.dumps([1, 2, 3]).encode("utf-8")
    with open(str(p), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    h = D.read_safetensors_header(str(p))
    assert "error" in h

def test_read_safetensors_header_non_dict_tensor_spec_returns_error(tmp_path):
    p = tmp_path / "bad_spec.safetensors"
    _write_safetensors(str(p), {"a.weight": 5})
    h = D.read_safetensors_header(str(p))
    assert "error" in h

def test_read_safetensors_header_oversized_length_returns_error(tmp_path):
    p = tmp_path / "oversized.safetensors"
    with open(str(p), "wb") as f:
        f.write(struct.pack("<Q", 2**60))
        f.write(b"{}")
    h = D.read_safetensors_header(str(p))
    assert "error" in h

def test_header_diff_reports_dtype_order_metadata_changes():
    ha = {"tensors": {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                      "lm": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]}},
          "metadata": {"format": "pt"}, "order": ["w", "lm"]}
    hb = {"tensors": {"w": {"dtype": "BF16", "shape": [4], "data_offsets": [8, 16]}},
          "metadata": None, "order": ["w"]}
    d = D.header_diff(ha, hb)
    assert d["tensor_only_a"] == ["lm"]
    assert d["tensor_only_b"] == []
    assert d["dtype_changes"] == {"w": ["F32", "BF16"]}
    assert d["order_equal"] is False
    assert d["metadata_equal"] is False

def test_header_diff_propagates_parse_error():
    d = D.header_diff({"error": "bad"}, {"tensors": {}, "metadata": None, "order": []})
    assert d["error_a"] == "bad"

def test_json_structural_diff_flatten_and_byte_identical(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text('{"architectures": ["LlamaForCausalLM"], "n": {"x": 1, "y": 2}}')
    b.write_text('{"architectures": ["Gemma3ForCausalLM"], "n": {"x": 1}, "z": 3}')
    d = D.json_structural_diff(str(a), str(b))
    assert d["changed"]["architectures"] == [["LlamaForCausalLM"], ["Gemma3ForCausalLM"]]
    assert d["only_a_paths"] == ["n.y"]
    assert d["only_b_paths"] == ["z"]
    assert d["byte_identical"] is False

def test_json_structural_diff_identical(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text('{"k": 1}'); b.write_text('{"k": 1}')
    d = D.json_structural_diff(str(a), str(b))
    assert d["byte_identical"] is True
    assert d["changed"] == {}
