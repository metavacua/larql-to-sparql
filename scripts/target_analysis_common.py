#!/usr/bin/env python3
"""Shared, dependency-free JSON parsing for the target-analysis pipeline's
indexing and promotion scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

SKIP_MARKER_KEY = "skipped_no_std_guaranteed_fail"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_jsonl(text: str) -> list[Any]:
    """Parse already-in-memory JSONL text (one JSON object per line, blank
    lines skipped) -- the core `load_jsonl()` delegates to, for callers that
    have JSONL from somewhere other than a file (e.g. a subprocess's stdout)."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_jsonl(path: Path) -> list[Any]:
    """Read cargo's `--message-format=json` wire format: one JSON object per
    line, not a single JSON array (confirmed against real cargo output; see
    this pipeline's own workflow comments on the same distinction)."""
    return parse_jsonl(path.read_text(encoding="utf-8"))


def unit_graph_units_named(unit_graph: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        unit
        for unit in unit_graph.get("units", [])
        if unit.get("target", {}).get("name") == name
    ]


def _normalize_site_path(file_name: str) -> str:
    match = re.search(r"registry/src/[^/]+/([^/]+)-\d[^/]*/(.*)", file_name)
    if match:
        crate_name_no_version = match.group(1)
        rest = match.group(2)
        return f"{crate_name_no_version}/{rest}"
    return file_name


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
        primary_spans = [s for s in message.get("spans", []) if s.get("is_primary")]
        if primary_spans:
            for span in primary_spans:
                sites.add((_normalize_site_path(span.get("file_name", "")), span.get("line_start", -1), code))
        else:
            sites.add(("<spanless>", -1, code))
    return sites


def is_skip_record(compiler_messages: list[Any]) -> bool:
    """True iff this is a deliberate build-attempt skip record (see
    scripts/target_analysis_build_attempt.py's skip_record()), not a real
    cargo --message-format=json stream. Checked first by anything that
    would otherwise treat an empty error list as evidence of a clean build
    -- a skip is neither a pass nor a contradiction, and must never be
    indistinguishable from either."""
    return (
        bool(compiler_messages)
        and isinstance(compiler_messages[0], dict)
        and compiler_messages[0].get(SKIP_MARKER_KEY) is True
    )
