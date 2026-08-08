import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import tensor_presence as T

ATTN_ONLY = {"attn_weights.bin": 1000, "gate_vectors.bin": 500, "embeddings.bin": 200, "index.json": 10}
FULL      = {"attn_weights.bin": 1000, "up_weights.bin": 800, "down_weights.bin": 800,
             "gate_vectors.bin": 500, "embeddings.bin": 200}
EMPTY     = {"index.json": 10, "gate_vectors.bin": 0}   # 0-byte gate = absent


def test_presence_attention_only_flags_ffn_unwrap_risk():
    p = T.presence(ATTN_ONLY)
    assert p["has_attn"] is True and p["has_ffn"] is False
    assert p["ffn_unwrap_risk"] is True

def test_presence_full_has_ffn_no_risk():
    p = T.presence(FULL)
    assert p["has_attn"] and p["has_ffn"] and p["has_gate"]
    assert p["ffn_unwrap_risk"] is False

def test_presence_empty_no_attn_no_risk():
    p = T.presence(EMPTY)
    assert p["has_attn"] is False and p["ffn_unwrap_risk"] is False

def test_presence_tolerates_none_and_nonint():
    assert T.presence(None)["n_files"] == 0
    assert T.presence({"attn_weights.bin": "big"})["has_attn"] is False  # non-int size ignored

def test_collect_maps_leg_to_presence(tmp_path):
    d = tmp_path / "results-qwen05.native.attention" / "manifest-qwen05.native.attention"
    d.mkdir(parents=True)
    (d / "listing.json").write_text(json.dumps(ATTN_ONLY), encoding="utf-8")
    d2 = tmp_path / "results-qwen05.native.all" / "manifest-qwen05.native.all"
    d2.mkdir(parents=True)
    (d2 / "listing.json").write_text(json.dumps(FULL), encoding="utf-8")
    got = T.collect(str(tmp_path / "results-*/manifest-*/listing.json"))
    assert got["qwen05.native.attention"]["ffn_unwrap_risk"] is True
    assert got["qwen05.native.all"]["ffn_unwrap_risk"] is False

def test_load_listing_raises_on_missing(tmp_path):
    # A missing listing must NOT read as "a vindex with no files". It used to
    # return {}, which made a harness failure look like a product finding.
    import pytest
    with pytest.raises(OSError):
        T.load_listing(str(tmp_path / "nope.json"))


def test_load_listing_raises_on_non_object(tmp_path):
    p = tmp_path / "listing.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        T.load_listing(str(p))


def test_collect_records_unreadable_listing_per_leg(tmp_path):
    # One bad listing is reported as an error row for THAT leg; the others
    # still yield presence data.
    good = tmp_path / "results-a" / "manifest-a"
    bad = tmp_path / "results-b" / "manifest-b"
    good.mkdir(parents=True); bad.mkdir(parents=True)
    (good / "listing.json").write_text(json.dumps({"gate_vectors.bin": 10}), encoding="utf-8")
    (bad / "listing.json").write_text("{not json", encoding="utf-8")
    rows = T.collect(str(tmp_path / "results-*" / "manifest-*" / "listing.json"))
    assert rows["a"]["has_gate"] is True
    assert "error" in rows["b"] and "path" in rows["b"]

