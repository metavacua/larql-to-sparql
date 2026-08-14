# gen_legs_test.py
import gen_legs as G


def test_unfiltered_matrix_is_full(monkeypatch):
    monkeypatch.delenv("LQL_MATRIX_ONLY", raising=False)
    legs = G._apply_model_filter(G.build_legs())
    ids = {G._model_id(lg["name"]) for lg in legs}
    # every declared model is present when no filter is set
    assert {"qwen05", "smol135", "smol135base", "smol360", "qwen15",
            "granite1b", "bitnet2b", "bitnetgguf"} <= ids


def test_filter_keeps_only_named_models(monkeypatch):
    monkeypatch.setenv("LQL_MATRIX_ONLY", "smol135,smol135base")
    legs = G._apply_model_filter(G.build_legs())
    ids = {G._model_id(lg["name"]) for lg in legs}
    assert ids == {"smol135", "smol135base"}
    # the round-trip legs survive the narrowing (that's the point of the run)
    rt = sorted(lg["name"] for lg in legs if lg["roundtrip"])
    assert rt == ["smol135.native.all", "smol135.native.inference",
                  "smol135base.native.all", "smol135base.native.inference"]


def test_filter_is_reversible_default(monkeypatch):
    # empty / unset value is a no-op (full matrix), so removing the CI env var
    # restores full coverage
    monkeypatch.setenv("LQL_MATRIX_ONLY", "")
    assert G._apply_model_filter(G.build_legs()) == G.build_legs()
