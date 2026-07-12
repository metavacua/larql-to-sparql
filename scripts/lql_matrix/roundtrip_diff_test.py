import os, hashlib
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
