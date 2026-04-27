#!/usr/bin/env python3
"""Deduplicate and sort pairs in all triple files.

For each data/triples/*.json file:
  - Remove duplicate pairs
  - Sort pairs alphabetically by first element, then second
  - Write back in-place

Usage:
    python3 scripts/normalize_triples.py
"""

import json
from pathlib import Path


def normalize_file(path: Path) -> tuple[int, int]:
    """Normalize a single triple file.

    Returns (original_count, final_count).
    """
    with open(path) as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    original = len(pairs)

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[list[str]] = []
    for pair in pairs:
        key = (pair[0], pair[1])
        if key not in seen:
            seen.add(key)
            deduped.append(pair)

    # Sort by (source, target)
    deduped.sort(key=lambda p: (p[0].lower(), p[1].lower()))

    data["pairs"] = deduped

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return original, len(deduped)


def main() -> None:
    triples_dir = Path(__file__).parent.parent / "data" / "triples"
    if not triples_dir.exists():
        print("No triples directory found.")
        return

    total_removed = 0
    for f in sorted(triples_dir.glob("*.json")):
        original, final = normalize_file(f)
        removed = original - final
        total_removed += removed
        status = f"  removed {removed}" if removed > 0 else ""
        print(f"  {f.name:<30s} {original:>4d} -> {final:>4d}{status}")

    print(f"\nTotal duplicates removed: {total_removed}")


if __name__ == "__main__":
    main()
