import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import conformance as C


def make_leg(name="lg", cells=None, descriptor=None, produce=None, meta=None):
    return C.Leg(name=name, meta=meta or {"level": name},
                 cells=cells or {}, descriptor=descriptor or {},
                 produce=produce or {})


def test_load_reads_meta_cells_descriptor_produce(tmp_path):
    d = tmp_path / "results-lgA"
    d.mkdir()
    (d / "results-lgA.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lgA"}) + "\n" +
        json.dumps({"level": "lgA", "id": "stats", "bucket": "ok",
                    "exit_code": 0, "stdout_head": "(24 layers, 5K features, m)"}) + "\n",
        encoding="utf-8")
    (d / "descriptor-lgA.json").write_text(json.dumps({"name": "lgA", "family": "qwen2"}), encoding="utf-8")
    (d / "produce-lgA.json").write_text(json.dumps({"name": "lgA", "op": "extract", "bucket": "ok"}), encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert set(legs) == {"lgA"}
    assert legs["lgA"].cells["stats"]["bucket"] == "ok"
    assert legs["lgA"].descriptor["family"] == "qwen2"
    assert legs["lgA"].produce["op"] == "extract"


def test_run_with_empty_registry_is_green(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [])
    legs = {"lg": make_leg()}
    monkeypatch.setattr(C, "load", lambda g: legs)
    code = C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=True)
    assert code == 0
    assert json.loads((tmp_path / "c.json").read_text())["violations"] == []


def _cell(cid, stdout="", bucket="ok", exit_code=0, err_signal=0, err_line="", stderr=""):
    return {"id": cid, "bucket": bucket, "exit_code": exit_code, "err_signal": err_signal,
            "err_line": err_line, "stdout_head": stdout, "stderr_head": stderr}


def test_feature_count_parses_banner_with_suffix():
    lg = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 116.7K features, m)")})
    assert C.feature_count(lg) == 116700
    lg0 = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 0 features, m)")})
    assert C.feature_count(lg0) == 0


def test_completeness_flags_hollow_vindex():
    hollow = make_leg("granite", cells={"stats": _cell("stats", "(24 layers, 0 features, m)")})
    healthy = make_leg("qwen", cells={"stats": _cell("stats", "(24 layers, 5K features, m)")})
    vs = C.inv_completeness({"granite": hollow, "qwen": healthy})
    assert [v.leg for v in vs] == ["granite"]
    assert vs[0].invariant == "completeness"


def test_produce_failure_is_distinct_produce_violation_not_hollow():
    # fp4-like: produce FAILED (exit 1, err bucket), descriptor knows produced=False,
    # and every corpus cell errored so there is no feature banner. This must be a
    # definitive `produce` violation, NOT a hedged completeness/hollow one.
    fp4 = make_leg("qwen15.xform.fp4",
                   cells={"stats": _cell("stats", "Error: vindex not found", err_signal=1)},
                   descriptor={"produced": False, "quant_match": False},
                   produce={"op": "quantize-fp4", "bucket": "err", "exit_code": 1})
    # a genuinely produced-but-hollow leg (granite-like) must STAY a completeness/hollow
    hollow = make_leg("granite",
                      cells={"stats": _cell("stats", "(24 layers, 0 features, m)")},
                      descriptor={"produced": True},
                      produce={"op": "extract", "bucket": "ok", "exit_code": 0})
    legs = {"qwen15.xform.fp4": fp4, "granite": hollow}

    prod = C.inv_produce(legs)
    assert [v.leg for v in prod] == ["qwen15.xform.fp4"]
    assert prod[0].invariant == "produce"
    assert "1" in prod[0].detail  # the exit code is surfaced

    comp = C.inv_completeness(legs)
    # the produce-failed leg is NOT reported as hollow; only the real hollow is
    assert [v.leg for v in comp] == ["granite"]


def test_produce_crash_left_to_no_crash_not_double_reported():
    # a produce CRASH (137) is already handled by inv_no_crash; inv_produce must
    # not also report it (avoid double-counting the same failure).
    crash = make_leg("boom", descriptor={"produced": False},
                     produce={"op": "extract", "bucket": "crash", "exit_code": 137})
    legs = {"boom": crash}
    assert any(v.invariant == "no-crash" and v.cell == "produce" for v in C.inv_no_crash(legs))
    assert C.inv_produce(legs) == []


def test_feature_count_tolerates_malformed_banner():
    lg = make_leg(cells={"stats": _cell("stats", "(24 layers, . features, m)")})
    assert C.feature_count(lg) is None
    lg2 = make_leg(cells={"stats": _cell("stats", "(24 layers, 1.2.3 features, m)")})
    assert C.feature_count(lg2) is None


def test_no_crash_flags_panic_and_crash_not_graceful():
    legs = {
        "panic": make_leg("panic", cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101)}),
        "crash": make_leg("crash", cells={"x": _cell("x", bucket="crash", exit_code=137)}),
        "graceful": make_leg("graceful", cells={"infer.top5": _cell("infer.top5", bucket="ok", exit_code=0, err_signal=1, err_line="Error: requires model weights")}),
        "cleanerr": make_leg("cleanerr", cells={"y": _cell("y", bucket="err", exit_code=3)}),
        "produce_panic": make_leg("pp", produce={"bucket": "crash", "exit_code": 139}),
    }
    vs = C.inv_no_crash(legs)
    flagged = sorted({v.leg for v in vs})
    assert flagged == ["crash", "panic", "produce_panic"]
    assert all(v.invariant == "no-crash" for v in vs)


def test_descriptor_match_flags_quant_mismatch_and_generic_fallback():
    legs = {
        "ok": make_leg("ok", descriptor={"name": "ok", "quant_match": True, "family": "qwen2"}),
        "dequant": make_leg("dequant", descriptor={"name": "dequant", "quant_match": False,
                            "observed_quant": "none", "expect_quant": "q4k", "family": "qwen2"}),
        "generic": make_leg("generic", descriptor={"name": "generic", "quant_match": True, "family": "generic"}),
        "nodesc": make_leg("nodesc", descriptor={}),
    }
    vs = C.inv_descriptor_match(legs)
    assert sorted({v.leg for v in vs}) == ["dequant", "generic"]
    assert all(v.invariant == "descriptor-match" for v in vs)


SHOW_LAYERS_HOLLOW = ("Layer      Features  With Meta       Top Token\n"
                      "------------------------------------------------\n"
                      "L0                0          0\n"
                      "L1                0          0\n")
SHOW_LAYERS_OK = ("Layer      Features  With Meta       Top Token\n"
                  "------------------------------------------------\n"
                  "L0             2560          0\n"
                  "L1             2560          0\n")


def test_cross_check_flags_show_layers_zero_vs_stats_nonzero():
    bad = make_leg("bad", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_HOLLOW)})
    good = make_leg("good", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_OK)})
    vs = C.inv_cross_check({"bad": bad, "good": good})
    assert [v.leg for v in vs] == ["bad"]
    assert vs[0].invariant == "cross-check"


SHOW_LAYERS_MIXED = ("Layer      Features  With Meta       Top Token\n"
                     "------------------------------------------------\n"
                     "L0                0          0\n"
                     "L1             2.6K          0\n")


def test_cross_check_no_false_positive_on_ksuffix_rows():
    lg = make_leg("mixed", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_MIXED)})
    assert C.show_layers_total(lg) == 2600
    assert C.inv_cross_check({"mixed": lg}) == []


def test_diagnostic_flag_honored_and_message_attribution():
    silent_override = make_leg("q4kbrowse",
        produce={"op": "extract", "level": "browse", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    honored = make_leg("q4kall",
        produce={"op": "extract", "level": "all", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    misattrib = make_leg("mis", produce={"op": "extract", "level": "all", "flags": ""},
        cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101,
               err_line="panicked: FFN weight tensor missing — this is a `--compact` vindex")})
    vs = C.inv_diagnostic({"q4kbrowse": silent_override, "q4kall": honored, "mis": misattrib})
    invs = sorted((v.leg, v.invariant) for v in vs)
    assert ("mis", "diagnostic") in invs
    assert ("q4kbrowse", "diagnostic") in invs
    assert not any(v.leg == "q4kall" for v in vs)


def test_load_folds_produce_err_into_stderr_head(tmp_path):
    d = tmp_path / "results-lg1"
    d.mkdir()
    (d / "results-lg1.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lg1"}) + "\n", encoding="utf-8")
    (d / "produce-lg1.json").write_text(
        json.dumps({"name": "lg1", "op": "extract", "level": "browse", "flags": "--quant q4k"}),
        encoding="utf-8")
    (d / "produce-lg1.err").write_text("extract stderr, no override warning here", encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert "no override warning" in legs["lg1"].produce["stderr_head"]


def test_transformation_report_surfaces_gaps():
    legs = {
        "qwen.q4k": make_leg("qwen.q4k",
            produce={"op": "extract", "flags": "--quant q4k"},
            descriptor={"family": "qwen2", "observed_quant": "q4k"}),
        "bitnet.gguf": make_leg("bitnet.gguf",
            produce={"op": "gguf-to-vindex", "flags": ""},
            descriptor={"family": "bitnet", "observed_quant": "none"}),
    }
    md = "\n".join(C.transformation_report(legs))
    assert "Transformation" in md
    assert "qwen2" in md and "bitnet" in md
    # gaps surfaced verbatim
    assert "Ollama" in md and "GGUF" in md
    assert "no compiled" in md.lower() or "gap" in md.lower()


def test_run_strict_fails_on_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: [C.Violation("x", "lg", "", "boom")]])
    monkeypatch.setattr(C, "load", lambda g: {"lg": make_leg()})
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=True) == 1
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=False) == 0


def test_non_dict_sidecar_does_not_crash(tmp_path):
    d = tmp_path / "results-lg"
    d.mkdir()
    (d / "results-lg.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lg"}) + "\n", encoding="utf-8")
    (d / "produce-lg.json").write_text("[1, 2, 3]", encoding="utf-8")   # valid JSON, not a dict
    (d / "descriptor-lg.json").write_text("null", encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert legs["lg"].produce == {}
    assert legs["lg"].descriptor == {}
    # running the full oracle over this leg must not raise
    C.run(str(tmp_path / "results-*/results-*.jsonl"),
          str(tmp_path / "m.md"), str(tmp_path / "j.json"), strict=False)
