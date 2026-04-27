#!/usr/bin/env python3
"""Assemble individual triple files into a single wikidata_triples.json.

Reads all JSON files from data/triples/ and combines them into the format
expected by the clustering pipeline.

Usage:
    python3 scripts/assemble_triples.py
"""

import json
from pathlib import Path


def main() -> None:
    """Assemble individual triple files into a single combined JSON."""
    triples_dir = Path(__file__).parent.parent / "data" / "triples"
    output_path = Path(__file__).parent.parent / "data" / "wikidata_triples.json"

    if not triples_dir.exists():
        print(f"No triples directory: {triples_dir}")
        return

    combined = {}
    total_pairs = 0

    for f in sorted(triples_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        relation = data.get("relation", f.stem)
        pid = data.get("pid", "")
        pairs = data.get("pairs", [])

        combined[relation] = {
            "pid": pid,
            "pairs": pairs,
        }
        total_pairs += len(pairs)
        print(f"  {relation:<25s} {len(pairs):4d} pairs  ({f.name})")

    with open(output_path, "w") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)

    print(f"\nAssembled {len(combined)} relations, {total_pairs} pairs → {output_path}")


if __name__ == "__main__":
    main()
