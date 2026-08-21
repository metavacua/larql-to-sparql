from pathlib import Path

from scripts.target_analysis_common import load_json
from scripts.target_analysis_indexing import (
    build_attempt_filenames,
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
