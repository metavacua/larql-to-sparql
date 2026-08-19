#!/usr/bin/env python3
"""Structural extraction for the indexing job: error counts by cargo target
name, the unexpected-clean-std-build contradiction rule (Standing Principle
5 — the nvptx canary), and artifact-completeness checking (Standing
Principle 9 — the indexing job fails loudly on any missing expected
artifact rather than silently indexing whatever showed up)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.target_analysis_common import error_level_messages, load_json


def count_errors_by_target(compiler_messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry, _message in error_level_messages(compiler_messages):
        name = entry.get("target", {}).get("name", "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def unexpected_clean_std_build(target_spec: dict[str, Any], std_mode_errors: list[Any]) -> bool:
    return target_spec.get("std") is False and len(std_mode_errors) == 0


def missing_artifacts(expected: set[str], actual: set[str]) -> set[str]:
    return expected - actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-messages-file", required=True, type=Path)
    parser.add_argument("--target-spec-file", required=True, type=Path)
    parser.add_argument("--std-mode-errors-file", required=True, type=Path)
    parser.add_argument("--expected-artifacts-file", required=True, type=Path)
    parser.add_argument("--actual-artifacts-file", required=True, type=Path)
    args = parser.parse_args()

    compiler_messages = load_json(args.compiler_messages_file)
    target_spec = load_json(args.target_spec_file)
    std_mode_errors = load_json(args.std_mode_errors_file)
    expected = set(load_json(args.expected_artifacts_file))
    actual = set(load_json(args.actual_artifacts_file))

    missing = missing_artifacts(expected, actual)
    result = {
        "error_counts_by_target": count_errors_by_target(compiler_messages),
        "contradictions": {
            "unexpected_clean_std_build": unexpected_clean_std_build(target_spec, std_mode_errors),
        },
        "missing_artifacts": sorted(missing),
    }
    print(json.dumps(result, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
