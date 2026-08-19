#!/usr/bin/env python3
"""Resolves the target matrix the discovery job hands downstream jobs via
fromJSON(). A requested target that isn't real rustc target-list output is
a loud failure here, never a silent narrowing (Standing Principle 8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_target_list(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def resolve_target_matrix(all_targets: list[str], requested: str | None) -> list[str]:
    if requested is None:
        return all_targets
    if requested not in all_targets:
        raise ValueError(
            f"requested target '{requested}' is not in rustc --print target-list output"
        )
    return [requested]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    matrix = resolve_target_matrix(all_targets, args.requested_target)
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
