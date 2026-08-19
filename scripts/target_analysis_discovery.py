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


def chunk_targets(targets: list[str], max_size: int = 256) -> list[list[str]]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    return [targets[i : i + max_size] for i in range(0, len(targets), max_size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)
    batches = chunk_targets(matrix, max_size=args.max_batch_size)
    batch_indices = list(range(len(batches)))

    print(json.dumps(matrix))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"batches={json.dumps(batches)}\n")
            handle.write(f"batch-indices={json.dumps(batch_indices)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
