import json

import pytest

from scripts.target_analysis_discovery import parse_target_list, resolve_target_matrix

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
    from scripts.target_analysis_discovery import main
    import sys

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
