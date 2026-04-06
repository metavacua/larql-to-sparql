#!/usr/bin/env python3
"""Filter triple pairs to single/few-token entities.

Removes pairs where either the source or target entity has too many
tokens (words).  This is important for probe-based evaluation because
multi-token entities are harder to isolate in model activations.

Usage:
    python3 scripts/filter_entities.py [--max-tokens 2]
"""

import argparse
import json
from pathlib import Path


def count_tokens(text: str) -> int:
    """Count the number of whitespace-separated tokens in a string."""
    return len(text.split())


def filter_file(path: Path, max_tokens: int) -> tuple[int, int]:
    """Filter a single triple file in-place.

    Returns (original_count, final_count).
    """
    with open(path) as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    original = len(pairs)

    filtered = [
        pair for pair in pairs
        if count_tokens(pair[0]) <= max_tokens
        and count_tokens(pair[1]) <= max_tokens
    ]

    data["pairs"] = filtered

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return original, len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter triples to single/few-token entities"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2,
        help="Maximum number of tokens per entity (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without modifying files"
    )
    args = parser.parse_args()

    triples_dir = Path(__file__).parent.parent / "data" / "triples"
    if not triples_dir.exists():
        print("No triples directory found.")
        return

    total_removed = 0
    for f in sorted(triples_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        pairs = data.get("pairs", [])
        original = len(pairs)
        kept = [
            p for p in pairs
            if count_tokens(p[0]) <= args.max_tokens
            and count_tokens(p[1]) <= args.max_tokens
        ]
        removed = original - len(kept)
        total_removed += removed

        if args.dry_run:
            if removed > 0:
                print(f"  {f.name:<30s} {original:>4d} -> {len(kept):>4d}  (would remove {removed})")
                dropped = [p for p in pairs if p not in kept]
                for p in dropped[:5]:
                    print(f"    drop: {p[0]!r} -> {p[1]!r}")
                if len(dropped) > 5:
                    print(f"    ... and {len(dropped) - 5} more")
        else:
            _, final = filter_file(f, args.max_tokens)
            status = f"  removed {removed}" if removed > 0 else ""
            print(f"  {f.name:<30s} {original:>4d} -> {final:>4d}{status}")

    action = "would remove" if args.dry_run else "removed"
    print(f"\nTotal {action}: {total_removed} pairs (max_tokens={args.max_tokens})")


if __name__ == "__main__":
    main()
