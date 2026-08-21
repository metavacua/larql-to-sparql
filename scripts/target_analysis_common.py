#!/usr/bin/env python3
"""Shared, dependency-free JSON parsing for the target-analysis pipeline's
indexing and promotion scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[Any]:
    """Read cargo's `--message-format=json` wire format: one JSON object per
    line, not a single JSON array (confirmed against real cargo output; see
    this pipeline's own workflow comments on the same distinction)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unit_graph_units_named(unit_graph: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        unit
        for unit in unit_graph.get("units", [])
        if unit.get("target", {}).get("name") == name
    ]


def error_level_messages(
    compiler_messages: list[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (entry, message) pairs for error-level compiler-message entries.

    The single definition of "counts as an error" that error_sites() and
    count_errors_by_target() both need — kept in one place so the two never
    drift out of sync on what qualifies.
    """
    for entry in compiler_messages:
        if entry.get("reason") != "compiler-message":
            continue
        message = entry.get("message", {})
        if message.get("level") != "error":
            continue
        yield entry, message


def error_sites(compiler_messages: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    sites: set[tuple[str, int, str]] = set()
    for _entry, message in error_level_messages(compiler_messages):
        code = (message.get("code") or {}).get("code") or message.get("message", "")[:60]
        for span in message.get("spans", []):
            if span.get("is_primary"):
                sites.add((span.get("file_name", ""), span.get("line_start", -1), code))
    return sites
