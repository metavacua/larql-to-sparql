#!/usr/bin/env python3
"""Resolves the target matrix the discovery job hands downstream jobs via
fromJSON(). A requested target that isn't real rustc target-list output is
a loud failure here, never a silent narrowing (Standing Principle 8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.target_analysis_common import load_json


def parse_target_list(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def expand_batch_json(raw: str) -> str:
    """Turn a `toJSON(...)`-produced JSON array string (a batch's targets)
    into the newline-delimited list every batched job's per-target
    `while IFS= read` loop consumes."""
    return "\n".join(json.loads(raw))


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


def filter_by_tier(targets_with_tiers: dict[str, int | None], max_tier: int) -> list[str]:
    return [
        target
        for target, tier in targets_with_tiers.items()
        if tier is not None and tier <= max_tier
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--github-output", type=Path, default=None)
    parser.add_argument("--target-tiers-file", type=Path, default=None)
    parser.add_argument("--max-tier", type=int, default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None

    if requested is not None:
        matrix = resolve_target_matrix(all_targets, requested)
    elif args.target_tiers_file is not None and args.max_tier is not None:
        target_tiers = load_json(args.target_tiers_file)
        matrix = filter_by_tier(target_tiers, max_tier=args.max_tier)
    else:
        matrix = all_targets

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
