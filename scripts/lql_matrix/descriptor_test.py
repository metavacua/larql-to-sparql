import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import descriptor as D


def test_describe_v2_matches_quant():
    idx = {"family": "qwen2", "dtype": "f16", "quant": "q4k",
           "has_model_weights": True, "num_layers": 24, "hidden_size": 896}
    d = D.describe_v2(idx, expect="q4k")
    assert d["generation"] == "v2"
    assert d["observed_quant"] == "q4k"
    assert d["quant_match"] is True


def test_describe_v2_bitnet_layout_overrides_quant_field():
    idx = {"family": "bitnet", "quant": "q4k", "bitnet_layout": {"scheme": "i2s"}}
    d = D.describe_v2(idx, expect="ternary")
    assert d["bitnet_layout"] is True
    assert d["observed_quant"] == "ternary"
    assert d["quant_match"] is True


def test_describe_v3_reads_representations_not_quant_scalar():
    idx = {
        "version": 4, "family": "llama", "hidden_size": 576, "num_layers": 30,
        "representations": {
            "obj.a@BF16": {"object": "a", "encoding": "BF16", "segment": "s1",
                           "tensor_count": 1, "payload_bytes": 10,
                           "payload_sha256": "x", "segment_sha256": "y"},
            "obj.b@BF16": {"object": "b", "encoding": "BF16", "segment": "s2",
                           "tensor_count": 1, "payload_bytes": 20,
                           "payload_sha256": "x", "segment_sha256": "y"},
        },
        "profiles": [{"name": "default"}],
    }
    d = D.describe_v3(idx, expect="none")
    assert d["generation"] == "v3"
    assert d["family"] == "llama"
    assert d["observed_quant"] == "BF16"
    assert d["num_representations"] == 2
    assert d["num_profiles"] == 1
    assert d["has_model_weights"] is True
    assert d["quant_match"] is True


def test_describe_v3_hollow_container_has_no_representations():
    idx = {"version": 4, "family": "llama", "hidden_size": 576, "num_layers": 30,
           "representations": {}, "profiles": []}
    d = D.describe_v3(idx, expect="none")
    assert d["num_representations"] == 0
    assert d["has_model_weights"] is False
    assert d["quant_match"] is True  # hollowness is completeness's finding, not quant's
    assert d["observed_quant"] is None


def test_describe_v3_quant_expectation_other_than_none_mismatches():
    d = D.describe_v3({"version": 4, "representations": {}, "profiles": []}, expect="q4k")
    assert d["quant_match"] is False
