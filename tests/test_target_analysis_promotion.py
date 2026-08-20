from pathlib import Path

import pytest

from scripts.target_analysis_common import error_sites, load_json
from scripts.target_analysis_promotion import (
    depth_advanced,
    no_std_scaffold_ok,
    serde_features_ok,
    stage_b_lib_rs_filenames,
    stage_promotes,
    workspace_members_ok,
)

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_serde_features_ok_is_false_for_default_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert serde_features_ok(unit_graph) is False


def test_serde_features_ok_is_true_for_patched_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_patched.json")
    assert serde_features_ok(unit_graph) is True


def test_serde_features_ok_is_false_when_serde_is_absent():
    assert serde_features_ok({"units": []}) is False


def test_workspace_members_ok_false_for_full_workspace():
    metadata = load_json(FIXTURES / "cargo_metadata_full_workspace.json")
    expected = ["larql-cli", "larql-boundary", "larql-vindex-spec"]
    assert workspace_members_ok(metadata, expected) is False


def test_workspace_members_ok_true_for_trimmed_workspace():
    metadata = load_json(FIXTURES / "cargo_metadata_trimmed_workspace.json")
    expected = ["larql-cli", "larql-boundary", "larql-vindex-spec"]
    assert workspace_members_ok(metadata, expected) is True


def test_workspace_members_ok_parses_real_modern_package_id_format():
    # Real cargo (this repo's toolchain: 1.97.1) package-id format is
    # "path+file:///abs/path/to/crate#version" -- no space anywhere, unlike the
    # older "name version (source)" format the fixtures previously (wrongly)
    # assumed, which is why this bug was invisible to this test file since Task 4.
    metadata = {
        "packages": [
            {"id": "path+file:///repo/crates/larql-cli#0.2.0", "name": "larql-cli"},
            {"id": "path+file:///repo/crates/larql-boundary#0.2.0", "name": "larql-boundary"},
        ],
        "workspace_members": [
            "path+file:///repo/crates/larql-cli#0.2.0",
            "path+file:///repo/crates/larql-boundary#0.2.0",
        ],
    }
    assert workspace_members_ok(metadata, ["larql-cli", "larql-boundary"]) is True


def test_no_std_scaffold_ok_requires_both_markers():
    assert no_std_scaffold_ok("//! docs\n#![no_std]\nextern crate alloc;\n") is True
    assert no_std_scaffold_ok("//! docs\n#![no_std]\n") is False
    assert no_std_scaffold_ok("//! docs\nextern crate alloc;\n") is False


def test_stage_promotes_serde_patch_transitions_false_to_true():
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is True


def test_stage_promotes_serde_patch_false_when_sibling_also_unpatched():
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is False


def test_stage_promotes_false_when_baseline_already_satisfies_postcondition():
    # A sibling can never promote by matching an already-true baseline —
    # promotion requires a genuine false-to-true transition, not just "true".
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is False


def test_depth_advanced_true_when_a_baseline_site_is_resolved():
    baseline = error_sites(load_json(FIXTURES / "compiler_messages_baseline.json"))
    sibling = error_sites(load_json(FIXTURES / "compiler_messages_sibling_progress.json"))
    # baseline has line 12 and 47; sibling has 47 and a new line 60.
    # Line 12 disappearing is real progress, even though a new site (60) appeared.
    assert depth_advanced(baseline, sibling) is True


def test_depth_advanced_false_when_nothing_baseline_present_is_resolved():
    baseline = error_sites(load_json(FIXTURES / "compiler_messages_baseline.json"))
    # sibling == baseline: identical wall, no resolution, no advancement.
    assert depth_advanced(baseline, baseline) is False


def test_depth_advanced_false_when_sibling_only_adds_new_sites():
    baseline = {("crates/larql-boundary/src/lib.rs", 12, "E0433")}
    sibling = baseline | {("crates/larql-boundary/src/lib.rs", 99, "E0999")}
    assert depth_advanced(baseline, sibling) is False


def test_stage_b_lib_rs_filenames_rejects_larql_cli():
    # larql-cli is bin-only (no src/lib.rs) and is never one of the crates
    # the Secondary-layer mutation job produces a baseline/sibling lib.rs
    # pair for. A caller asking for it by name gets a clear error here,
    # not a bare FileNotFoundError three layers downstream in a per-target
    # CI loop, across every target in every batch.
    with pytest.raises(ValueError, match="larql-cli"):
        stage_b_lib_rs_filenames("larql-cli")


def test_stage_b_lib_rs_filenames_default_is_larql_compute():
    baseline, sibling = stage_b_lib_rs_filenames()
    assert baseline == "baseline-lib-rs-larql-compute.txt"
    assert sibling == "sibling-lib-rs-larql-compute.txt"


def test_stage_b_lib_rs_filenames_accepts_any_real_mutated_crate():
    baseline, sibling = stage_b_lib_rs_filenames("larql-boundary")
    assert baseline == "baseline-lib-rs-larql-boundary.txt"
    assert sibling == "sibling-lib-rs-larql-boundary.txt"
