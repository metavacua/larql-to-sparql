import json
import sys

import pytest

from scripts.target_analysis_discovery import (
    chunk_targets,
    expand_batch_json,
    filter_by_tier,
    main,
    parse_target_list,
    resolve_target_matrix,
)

RAW = "aarch64-apple-darwin\nnvptx64-nvidia-cuda\nwasm32v1-none\nx86_64-unknown-linux-gnu\n"


def test_parse_target_list_splits_and_strips_blank_lines():
    raw_with_trailing_blank = RAW + "\n"
    assert parse_target_list(raw_with_trailing_blank) == [
        "aarch64-apple-darwin",
        "nvptx64-nvidia-cuda",
        "wasm32v1-none",
        "x86_64-unknown-linux-gnu",
    ]


def test_resolve_target_matrix_with_no_request_returns_everything():
    targets = parse_target_list(RAW)
    assert resolve_target_matrix(targets, None) == targets


def test_resolve_target_matrix_with_valid_request_returns_singleton():
    targets = parse_target_list(RAW)
    assert resolve_target_matrix(targets, "nvptx64-nvidia-cuda") == ["nvptx64-nvidia-cuda"]


def test_resolve_target_matrix_with_invalid_request_raises():
    targets = parse_target_list(RAW)
    with pytest.raises(ValueError, match="not-a-real-target"):
        resolve_target_matrix(targets, "not-a-real-target")


def test_main_treats_empty_string_requested_target_as_none(tmp_path, capsys):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text(RAW, encoding="utf-8")
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
    ]
    assert main() == 0
    out = capsys.readouterr().out
    assert json.loads(out) == parse_target_list(RAW)


def test_expand_batch_json_turns_json_array_into_newline_list():
    # The exact transform every batched job's `$BATCH_TARGETS` env var (a
    # `toJSON(...)`-produced JSON array string) needs before its per-target
    # `while IFS= read` loop -- previously reimplemented inline 5 times
    # across the workflow YAML.
    assert expand_batch_json('["a", "b", "c"]') == "a\nb\nc"


def test_expand_batch_json_single_target():
    assert expand_batch_json('["nvptx64-nvidia-cuda"]') == "nvptx64-nvidia-cuda"


def test_chunk_targets_empty_list_returns_empty():
    assert chunk_targets([], max_size=3) == []


def test_chunk_targets_smaller_than_max_size_is_one_chunk():
    assert chunk_targets(["a", "b"], max_size=3) == [["a", "b"]]


def test_chunk_targets_exactly_max_size_is_one_chunk():
    assert chunk_targets(["a", "b", "c"], max_size=3) == [["a", "b", "c"]]


def test_chunk_targets_splits_into_multiple_chunks_with_remainder():
    targets = ["a", "b", "c", "d", "e", "f", "g"]
    assert chunk_targets(targets, max_size=3) == [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g"],
    ]


def test_chunk_targets_default_max_size_is_256():
    targets = [f"t{i}" for i in range(300)]
    chunks = chunk_targets(targets)
    assert len(chunks) == 2
    assert len(chunks[0]) == 256
    assert len(chunks[1]) == 44


def test_chunk_targets_rejects_non_positive_max_size():
    with pytest.raises(ValueError, match="max_size must be positive"):
        chunk_targets(["a"], max_size=0)


def test_main_writes_matrix_batches_and_batch_indices_to_github_output(tmp_path):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("\n".join(f"t{i}" for i in range(300)), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    matrix_line = next(line for line in output_text.splitlines() if line.startswith("matrix="))
    batches_line = next(line for line in output_text.splitlines() if line.startswith("batches="))
    batch_indices_line = next(line for line in output_text.splitlines() if line.startswith("batch-indices="))
    matrix = json.loads(matrix_line[len("matrix="):])
    batches = json.loads(batches_line[len("batches="):])
    batch_indices = json.loads(batch_indices_line[len("batch-indices="):])
    assert matrix == [f"t{i}" for i in range(300)]
    assert len(batches) == 2
    assert batch_indices == [0, 1]


def test_filter_by_tier_keeps_only_tier_at_or_below_max():
    targets_with_tiers = {"a": 1, "b": 2, "c": 3, "d": None}
    assert filter_by_tier(targets_with_tiers, max_tier=2) == ["a", "b"]


def test_filter_by_tier_excludes_null_tier():
    targets_with_tiers = {"a": None}
    assert filter_by_tier(targets_with_tiers, max_tier=3) == []


def test_filter_by_tier_max_tier_3_keeps_everything_with_a_tier():
    targets_with_tiers = {"a": 1, "b": 2, "c": 3}
    assert filter_by_tier(targets_with_tiers, max_tier=3) == ["a", "b", "c"]


def test_main_applies_tier_filter_when_no_target_requested(tmp_path):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("a\nb\nc\nd\n", encoding="utf-8")
    tiers_file = tmp_path / "tiers.json"
    tiers_file.write_text(json.dumps({"a": 1, "b": 2, "c": 3, "d": None}), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
        "--target-tiers-file", str(tiers_file),
        "--max-tier", "2",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    matrix_line = next(line for line in output_text.splitlines() if line.startswith("matrix="))
    matrix = json.loads(matrix_line[len("matrix="):])
    assert matrix == ["a", "b"]


def test_main_skips_tier_filter_when_specific_target_requested(tmp_path):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("a\nb\nc\n", encoding="utf-8")
    tiers_file = tmp_path / "tiers.json"
    tiers_file.write_text(json.dumps({"a": 1, "b": 2, "c": 3}), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "c",
        "--target-tiers-file", str(tiers_file),
        "--max-tier", "2",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    matrix_line = next(line for line in output_text.splitlines() if line.startswith("matrix="))
    matrix = json.loads(matrix_line[len("matrix="):])
    assert matrix == ["c"]
