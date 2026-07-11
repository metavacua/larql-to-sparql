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
