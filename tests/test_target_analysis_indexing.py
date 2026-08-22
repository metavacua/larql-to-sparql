import tomllib
from pathlib import Path

from scripts.target_analysis_common import load_json
from scripts.target_analysis_build_attempt import guaranteed_std_failure, skip_record
from scripts.target_analysis_indexing import (
    build_attempt_filenames,
    combined_record_for_target,
    count_errors_by_target,
    missing_artifacts,
    unexpected_clean_std_build,
)

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_count_errors_by_target_counts_only_error_level():
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    assert count_errors_by_target(messages) == {"larql-boundary": 2}


def test_unexpected_clean_std_build_flags_the_nvptx_canary_contradiction():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert unexpected_clean_std_build(target_spec, std_mode_errors=[]) is True


def test_unexpected_clean_std_build_does_not_flag_a_real_failure():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert unexpected_clean_std_build(target_spec, std_mode_errors=["some error"]) is False


def test_unexpected_clean_std_build_does_not_flag_targets_with_std():
    host_spec = {"metadata": {"std": True}}
    assert unexpected_clean_std_build(host_spec, std_mode_errors=[]) is False


def test_unexpected_clean_std_build_ignores_stray_top_level_std_key():
    # Real rustc target-spec-json nests std under metadata; a stray top-level
    # "std" key (the exact shape mismatch that made this check inert) must
    # never be mistaken for the real field.
    target_spec = {"std": False, "metadata": {}}
    assert unexpected_clean_std_build(target_spec, std_mode_errors=[]) is False


def test_missing_artifacts_returns_the_set_difference():
    expected = {"probe-a", "probe-b", "probe-c"}
    actual = {"probe-a", "probe-c"}
    assert missing_artifacts(expected, actual) == {"probe-b"}


def test_missing_artifacts_empty_when_nothing_missing():
    expected = {"probe-a"}
    assert missing_artifacts(expected, expected) == set()


def test_build_attempt_filenames_covers_all_six_real_combos():
    # The exact cmd x features cartesian product build-attempt's own bash
    # loop generates -- single source of truth so a future change to either
    # dimension can't silently desync the two.
    assert build_attempt_filenames("nvptx64-nvidia-cuda") == [
        "attempt-nvptx64-nvidia-cuda-none-check-default-features.json",
        "attempt-nvptx64-nvidia-cuda-none-check-no-default-features.json",
        "attempt-nvptx64-nvidia-cuda-none-clippy-default-features.json",
        "attempt-nvptx64-nvidia-cuda-none-clippy-no-default-features.json",
        "attempt-nvptx64-nvidia-cuda-none-build-default-features.json",
        "attempt-nvptx64-nvidia-cuda-none-build-no-default-features.json",
    ]


def test_build_attempt_filenames_first_combo_matches_the_representative_single_file():
    # indexing's per-target loop reads exactly one of these six combos
    # (check + default-features) as its representative real build result --
    # confirmed here to be the first entry, so both stay in sync by
    # construction rather than by two independently-hand-typed literals.
    assert build_attempt_filenames("t")[0] == "attempt-t-none-check-default-features.json"


def test_guaranteed_std_failure_true_when_target_capability_says_no_std():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert guaranteed_std_failure(target_spec, "armv7a-none-eabihf") is True


def test_guaranteed_std_failure_false_when_target_has_std():
    assert guaranteed_std_failure({"metadata": {"std": True}}, "armv7a-none-eabihf") is False


def test_guaranteed_std_failure_false_for_the_sp5_canary_target_even_though_it_lacks_std():
    # nvptx64-nvidia-cuda is Standing Principle 5's own reference canary
    # for unexpected_clean_std_build -- it must keep running a real
    # attempt so that check stays empirically reachable, never silently
    # short-circuited away just because its own metadata.std is False.
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert guaranteed_std_failure(target_spec, "nvptx64-nvidia-cuda") is False


def test_skip_record_carries_the_marker_key_and_target():
    record = skip_record("armv7a-none-eabihf")
    assert record["skipped_no_std_guaranteed_fail"] is True
    assert record["target"] == "armv7a-none-eabihf"
    assert "ring" in record["skip_reason"] or "std" in record["skip_reason"]


def test_combined_record_for_target_reports_skip_not_a_contradiction():
    # Load-bearing case: a skip must NEVER be reported as
    # unexpected_clean_std_build's contradiction (Standing Principle 5),
    # which specifically watches for zero errors despite an ACTUAL attempt.
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")  # metadata.std is False
    messages = [skip_record("armv7a-none-eabihf")]
    record = combined_record_for_target(messages, target_spec, missing_sorted=[])
    assert record["skipped_no_std_guaranteed_fail"] is True
    assert "contradictions" not in record
    assert "error_counts_by_target" not in record


def test_reqwest_remains_unconditional_in_larql_cli_cargo_toml():
    # guaranteed_std_failure's whole premise (scripts/target_analysis_build_attempt.py)
    # is that larql-cli's reqwest dependency is unconditional -- not optional,
    # not gated by any [features] entry -- so a target lacking std is a
    # guaranteed failure regardless of which of the 6 cmd/feature combos runs.
    # That was verified twice by direct human inspection (2026-08-21) but never
    # pinned by code. If this assumption is ever violated (e.g. reqwest becomes
    # optional behind a new feature), the guaranteed-failure skip would become
    # wrong and silently under-measure targets that could actually build --
    # this test exists to make that drift loud instead of silent.
    cargo_toml_path = (
        Path(__file__).parent.parent / "crates" / "larql-cli" / "Cargo.toml"
    )
    with open(cargo_toml_path, "rb") as handle:
        data = tomllib.load(handle)

    dependencies = data.get("dependencies", {})
    assert "reqwest" in dependencies, "reqwest must remain a direct dependency of larql-cli"

    reqwest_entry = dependencies["reqwest"]
    if isinstance(reqwest_entry, dict):
        assert reqwest_entry.get("optional") is not True, (
            "reqwest must not become optional -- that would let a build "
            "configuration exist where it's absent, invalidating the "
            "guaranteed-failure skip's premise"
        )
    # else: a plain version string is inherently non-optional.

    for feature_name, feature_deps in data.get("features", {}).items():
        assert "reqwest" not in feature_deps and "dep:reqwest" not in feature_deps, (
            f"feature '{feature_name}' must not gate reqwest -- that would make "
            "it conditional, invalidating the guaranteed-failure skip's premise"
        )


def test_combined_record_for_target_preserves_existing_behavior_when_not_skipped():
    # Exact regression check: a real (non-skip) message list must produce
    # the identical shape the pre-existing inline workflow logic always did.
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    target_spec = {"metadata": {"std": True}}
    record = combined_record_for_target(messages, target_spec, missing_sorted=["x"])
    assert record == {
        "error_counts_by_target": {"larql-boundary": 2},
        "contradictions": {"unexpected_clean_std_build": False},
        "missing_artifacts": ["x"],
    }
