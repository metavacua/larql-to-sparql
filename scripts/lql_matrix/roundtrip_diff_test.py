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
