import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import dtype_parity as DP


def _write_descriptor(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")


def test_find_pairs_matches_native_and_lql_same_model_and_level(tmp_path):
    d = tmp_path / "results-a"
    _write_descriptor(d / "descriptor-smol135.native.inference.json", dtype="f16")
    _write_descriptor(d / "descriptor-smol135.lql.inference.json", dtype="f32")
    pairs = DP.find_pairs(str(tmp_path / "results-*" / "descriptor-*.json"))
    assert list(pairs.keys()) == ["smol135.inference"]
    assert pairs["smol135.inference"]["native"].endswith("smol135.native.inference.json")
    assert pairs["smol135.inference"]["lql"].endswith("smol135.lql.inference.json")


def test_find_pairs_drops_lone_native_leg(tmp_path):
    # Known, deferred behavior: a leg with only one side present (e.g. the lql
    # extract crashed and left no descriptor) produces no row at all -- see
    # the open matrix-audit finding on whether a missing side should instead
    # get an explicit null/"missing" marker. This test pins the CURRENT
    # behavior so a future change to it is deliberate, not accidental.
    d = tmp_path / "results-a"
    _write_descriptor(d / "descriptor-smol135.native.inference.json", dtype="f16")
    pairs = DP.find_pairs(str(tmp_path / "results-*" / "descriptor-*.json"))
    assert pairs == {}


def test_find_pairs_ignores_unrelated_filenames(tmp_path):
    d = tmp_path / "results-a"
    _write_descriptor(d / "descriptor-smol135.native.inference.json", dtype="f16")
    _write_descriptor(d / "descriptor-smol135.lql.inference.json", dtype="f32")
    (d / "manifest-smol135.native.inference").mkdir(parents=True)
    (d / "manifest-smol135.native.inference" / "listing.json").write_text("{}", encoding="utf-8")
    pairs = DP.find_pairs(str(tmp_path / "results-*" / "descriptor-*.json"))
    assert list(pairs.keys()) == ["smol135.inference"]


def test_find_pairs_dedupes_flat_and_recursive_glob(tmp_path):
    d = tmp_path / "results-a"
    _write_descriptor(d / "descriptor-smol135.native.inference.json", dtype="f16")
    _write_descriptor(d / "descriptor-smol135.lql.inference.json", dtype="f32")
    # A flat-glob pattern already matches these files directly under tmp_path/results-a;
    # the recursive fallback in _glob_all must not produce duplicate/second rows.
    pairs = DP.find_pairs(str(tmp_path / "results-*" / "descriptor-*.json"))
    assert len(pairs) == 1


def test_leg_summary_extracts_fixed_shape():
    s = DP.leg_summary({"dtype": "f16", "produced": True, "extra": "ignored"})
    assert s == {"dtype": "f16", "produced": True, "error": None}


def test_leg_summary_tolerates_error_only_descriptor():
    s = DP.leg_summary({"error": "JSONDecodeError: boom"})
    assert s["dtype"] is None and s["error"] == "JSONDecodeError: boom"


def test_main_writes_output_and_reports_zero_pairs_on_no_match(tmp_path, capsys):
    out = tmp_path / "dtype-parity.json"
    old_argv = sys.argv
    sys.argv = ["dtype_parity.py", str(tmp_path / "nope-*" / "descriptor-*.json"), str(out)]
    try:
        DP.main()
    finally:
        sys.argv = old_argv
    assert json.loads(out.read_text(encoding="utf-8")) == []
    assert "0 pairs" in capsys.readouterr().out


def test_main_reports_dtype_mismatch(tmp_path, capsys):
    d = tmp_path / "results-a"
    _write_descriptor(d / "descriptor-qwen05.native.default.json", dtype="f16", produced=True)
    _write_descriptor(d / "descriptor-qwen05.lql.default.json", dtype="f32", produced=True)
    out = tmp_path / "dtype-parity.json"
    old_argv = sys.argv
    sys.argv = ["dtype_parity.py", str(tmp_path / "results-*" / "descriptor-*.json"), str(out)]
    try:
        DP.main()
    finally:
        sys.argv = old_argv
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows == [{
        "pair": "qwen05.default",
        "cli": {"dtype": "f16", "produced": True, "error": None},
        "lql": {"dtype": "f32", "produced": True, "error": None},
    }]
    printed = capsys.readouterr().out
    assert "qwen05.default: cli=f16 lql=f32" in printed
