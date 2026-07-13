# roundtrip_compile_test.py
import json
import struct

import roundtrip_compile as RC


def test_plan_jobs_enumerates_8_expected_ids():
    jobs = RC.plan_jobs("smol135.native.inference", "/vindex/src", "/tmp/out")
    ids = [j["id"] for j in jobs]
    assert ids == [
        "lql.passthrough",
        "lql.knn.memitoff", "lql.knn.memiton",
        "lql.compose.memitoff", "lql.compose.memiton",
        "cli.passthrough", "cli.knn", "cli.compose",
    ]
    # every job carries the shared shape
    for j in jobs:
        assert j["driver"] in ("lql", "cli")
        assert j["mode"] in ("passthrough", "edit")
        assert j["vindex_copy"].startswith("/tmp/out")
        assert j["out_dir"].startswith("/tmp/out")


def test_plan_jobs_cli_edit_jobs_need_vlp_from_matching_lql_memitoff():
    jobs = RC.plan_jobs("leg", "/vindex/src", "/tmp/out")
    by_id = {j["id"]: j for j in jobs}
    assert by_id["cli.knn"]["needs_vlp_from"] == "lql.knn.memitoff"
    assert by_id["cli.compose"]["needs_vlp_from"] == "lql.compose.memitoff"
    # cli.passthrough gets its own (empty) vlp, no dependency
    assert by_id["cli.passthrough"]["needs_vlp_from"] is None
    assert by_id["cli.passthrough"]["vlp_path"] is not None


def test_lql_stmt_has_mode_clause():
    jobs = RC.plan_jobs("leg", "/vindex/src", "/tmp/out")
    by_id = {j["id"]: j for j in jobs}
    assert "MODE KNN" in by_id["lql.knn.memitoff"]["lql_stmt"]
    assert "MODE KNN" in by_id["lql.knn.memiton"]["lql_stmt"]
    assert "MODE COMPOSE" in by_id["lql.compose.memitoff"]["lql_stmt"]
    assert "MODE COMPOSE" in by_id["lql.compose.memiton"]["lql_stmt"]
    # passthrough has no INSERT at all
    assert "INSERT" not in by_id["lql.passthrough"]["lql_stmt"]
    # every edit job's statement is a single BEGIN PATCH ... SAVE PATCH ...
    # COMPILE batch (MEMIT only bakes in within one larql-lql process)
    for jid in ("lql.knn.memitoff", "lql.knn.memiton",
                "lql.compose.memitoff", "lql.compose.memiton"):
        stmt = by_id[jid]["lql_stmt"]
        assert "BEGIN PATCH" in stmt
        assert "SAVE PATCH" in stmt
        assert "COMPILE CURRENT INTO MODEL" in stmt
        assert stmt.index("BEGIN PATCH") < stmt.index("SAVE PATCH") < stmt.index("COMPILE")
    # USE opens the vindex, never USE VINDEX
    assert by_id["lql.passthrough"]["lql_stmt"].startswith('USE "')


def test_comparison_plan_covers_input_bva_driver_memit():
    jobs = RC.plan_jobs("leg", "/vindex/src", "/tmp/out")
    comps = RC.comparison_plan(jobs)

    input_vs = [c for c in comps if c["comparison"].startswith("input_vs_")]
    assert len(input_vs) == 8
    assert {c["slug_a"] for c in input_vs} == {"_input"}
    assert {c["slug_b"] for c in input_vs} == {j["id"] for j in jobs}

    bva = [c for c in comps
           if c["slug_a"] != "_input"
           and c["slug_b"] in ("lql.passthrough", "cli.passthrough")
           and c["driver"] in ("lql", "cli")]
    assert len(bva) == 6  # 4 lql edit jobs + 2 cli edit jobs

    driver_vs_driver = [c for c in comps if c["comparison"].startswith("driver_vs_driver:")]
    assert len(driver_vs_driver) == 5  # passthrough + 2 forms x 2 memit variants

    memit_effect = [c for c in comps if c["comparison"].startswith("memit_effect:")]
    assert len(memit_effect) == 2
    assert {c["insert_form"] for c in memit_effect} == {"knn", "compose"}

    # nothing dropped: 8 + 6 + 5 + 2
    assert len(comps) == 21


def test_run_one_comparison_degrades_on_missing_dir(tmp_path):
    meta = {"comparison": "input_vs_lql.passthrough", "driver": "lql",
            "mode": "passthrough", "insert_form": None, "memit": None,
            "slug_a": "_input", "slug_b": "lql.passthrough"}
    rows = RC.run_one_comparison(str(tmp_path / "nope_a"), str(tmp_path / "nope_b"), meta)
    assert len(rows) == 1
    r = rows[0]
    assert r["file"] == "_manifest"
    assert r["bijective"] is False
    assert r["dir_a_exists"] is False
    assert r["dir_b_exists"] is False
    assert r["comparison"] == "input_vs_lql.passthrough"


def _st(path, dt, values):
    import numpy as np
    arr = np.array(values, dtype="<u2" if dt == "BF16" else "<f4")
    raw = arr.tobytes()
    hdr = {"w": {"dtype": dt, "shape": list(arr.shape), "data_offsets": [0, len(raw)]}}
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw)


def test_run_one_comparison_delegates_to_diff_model_dirs(tmp_path):
    da = tmp_path / "a"
    db = tmp_path / "b"
    da.mkdir()
    db.mkdir()
    _st(da / "model.safetensors", "F32", [1.0, 2.0])
    _st(db / "model.safetensors", "F32", [1.0, 2.0])
    meta = {"comparison": "lql.knn.memitoff_vs_lql.passthrough", "driver": "lql",
            "mode": "edit", "insert_form": "knn", "memit": False,
            "slug_a": "lql.knn.memitoff", "slug_b": "lql.passthrough"}
    rows = RC.run_one_comparison(str(da), str(db), meta)
    manifest_row = [r for r in rows if r["file"] == "_manifest"][0]
    assert manifest_row["bijective"] is True
    st_row = [r for r in rows if r["file"] == "model.safetensors"][0]
    assert st_row["sha256_equal"] is True
    assert st_row["comparison"] == "lql.knn.memitoff_vs_lql.passthrough"
