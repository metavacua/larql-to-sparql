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
