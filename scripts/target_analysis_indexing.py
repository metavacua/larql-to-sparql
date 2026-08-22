#!/usr/bin/env python3
"""Structural extraction for the indexing job: error counts by cargo target
name, the unexpected-clean-std-build contradiction rule (Standing Principle
5 — the nvptx canary), and artifact-completeness checking (Standing
Principle 9 — the indexing job fails loudly on any missing expected
artifact rather than silently indexing whatever showed up)."""

from __future__ import annotations

from typing import Any

from scripts.target_analysis_common import error_level_messages, is_skip_record

# The exact cmd x features cartesian product build-attempt's own bash loop
# generates for every target -- single source of truth for both sides, so
# a future change to either dimension can't silently desync the workflow's
# generation loop from indexing's expected-artifact computation.
BUILD_ATTEMPT_CARGO_CMDS = ("check", "clippy", "build")
BUILD_ATTEMPT_FEATURE_MODES = ("default-features", "no-default-features")


def build_attempt_filenames(target: str) -> list[str]:
    return [
        f"attempt-{target}-none-{cmd}-{feat}.json"
        for cmd in BUILD_ATTEMPT_CARGO_CMDS
        for feat in BUILD_ATTEMPT_FEATURE_MODES
    ]


def count_errors_by_target(compiler_messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry, _message in error_level_messages(compiler_messages):
        name = entry.get("target", {}).get("name", "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def unexpected_clean_std_build(target_spec: dict[str, Any], std_mode_errors: list[Any]) -> bool:
    return target_spec.get("metadata", {}).get("std") is False and len(std_mode_errors) == 0


def missing_artifacts(expected: set[str], actual: set[str]) -> set[str]:
    return expected - actual


def combined_record_for_target(
    messages: list[Any],
    target_spec: dict[str, Any],
    missing_sorted: list[str],
) -> dict[str, Any]:
    """The per-target record indexing writes into index.json. A deliberate
    build-attempt skip (see scripts/target_analysis_build_attempt.py) is
    checked first and short-circuits to its own distinct shape -- it must
    never be indistinguishable from a real, attempted, zero-error build,
    which is exactly what unexpected_clean_std_build's contradiction check
    (Standing Principle 5) watches for."""
    if is_skip_record(messages):
        return {
            "skipped_no_std_guaranteed_fail": True,
            "missing_artifacts": missing_sorted,
        }
    std_mode_errors = [message for _entry, message in error_level_messages(messages)]
    return {
        "error_counts_by_target": count_errors_by_target(messages),
        "contradictions": {
            "unexpected_clean_std_build": unexpected_clean_std_build(target_spec, std_mode_errors),
        },
        "missing_artifacts": missing_sorted,
    }
