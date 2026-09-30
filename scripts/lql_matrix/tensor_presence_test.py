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

def test_load_listing_tolerates_missing(tmp_path):
    assert T.load_listing(str(tmp_path / "nope.json")) == {}

def test_resolve_q8_biconditional_and_counterexample():
    pres = {
        "qwen05.native.attention":   T.presence(ATTN_ONLY),   # risk=True
        "qwen05.native.all":         T.presence(FULL),        # risk=False
        "bitnet2b.native.attention": T.presence(EMPTY),       # risk=False (no attn wt)
    }
    panic = {  # observed panic@infer (from conformance no-crash)
        "qwen05.native.attention":   True,
        "qwen05.native.all":         False,
        "bitnet2b.native.attention": False,   # refuses gracefully — matches risk=False
    }
    r = T.resolve_q8(pres, panic)
    assert r["biconditional_holds"] is True
    assert r["counterexamples"] == []
    # now inject a violation: risk=True but no panic
    panic["qwen05.native.attention"] = False
    r2 = T.resolve_q8(pres, panic)
    assert r2["biconditional_holds"] is False
    assert "qwen05.native.attention" in [c["leg"] for c in r2["counterexamples"]]


def test_render_q8_none_violations_is_inconclusive():
    pres = {"m.native.attention": T.presence(ATTN_ONLY)}
    res = T.render_q8(pres, None)
    assert res["status"] == "inconclusive"
    assert res["biconditional_holds"] is None

def test_render_q8_empty_presence_is_inconclusive():
    res = T.render_q8({}, [])
    assert res["status"] == "inconclusive"
    assert res["biconditional_holds"] is None

def test_render_q8_excludes_produce_crash_from_panic():
    # one genuine risk leg that agrees (panicked at an actual corpus cell), plus
    # a non-risk leg whose ONLY no-crash violation is at produce-time — that
    # produce crash must NOT be counted as a panic, so it still agrees (risk=False).
    pres = {
        "m.native.attention": T.presence(ATTN_ONLY),   # risk=True
        "m.native.all":       T.presence(FULL),         # risk=False
    }
    violations = [
        {"invariant": "no-crash", "leg": "m.native.attention", "cell": "infer.top5"},
        {"invariant": "no-crash", "leg": "m.native.all", "cell": "produce"},
    ]
    res = T.render_q8(pres, violations)
    assert res["status"] == "holds"
    assert res["counterexamples"] == []
    assert "m.native.all" not in [r["leg"] for r in res["counterexamples"]]

def test_render_q8_genuine_refutation():
    # risk=True leg with no observed corpus panic at all -> real disagreement.
    pres = {"m.native.attention": T.presence(ATTN_ONLY)}
    res = T.render_q8(pres, [])
    assert res["status"] == "refuted"
    assert "m.native.attention" in [c["leg"] for c in res["counterexamples"]]

def test_render_q8_no_risk_no_panic_is_inconclusive():
    pres = {"m.native.all": T.presence(FULL)}  # risk=False
    res = T.render_q8(pres, [])
    assert res["status"] == "inconclusive"

def test_q8_markdown_inconclusive_does_not_raise():
    res = T.render_q8({}, None)
    md = T.q8_markdown(res)
    assert "inconclusive" in md
