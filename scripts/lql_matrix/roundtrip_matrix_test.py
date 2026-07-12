# roundtrip_matrix_test.py
import roundtrip_matrix as M


def test_registry_defaults_to_smol135_only():
    active = M.active_variants()
    ids = {a[0] for a in active}
    assert ids == {"smol135"}
    variants = {a[1] for a in active}
    assert variants == {"instruct-bf16", "base-f32"}
    # everything else deactivated but present (just-works-on-re-add)
    assert any(m["id"] == "bitnet2b" and m["active"] is False for m in M.REGISTRY)


def test_enumerate_comparisons_covers_lattice():
    rows = M.enumerate_comparisons("smol135", "instruct-bf16")
    comps = {r["comparison"] for r in rows}
    assert {"input_vs_A", "lqlA_vs_cliA", "B_vs_A", "lqlB_vs_cliB"} <= comps
    # B rows carry an insert_form; A rows do not
    assert all(r["insert_form"] in ("knn", "compose")
               for r in rows if r["mode"] == "B")
    assert all(r["insert_form"] is None for r in rows if r["mode"] == "A")
