#!/usr/bin/env python3
"""Generate English grammar bigram pairs and save to data/english_grammar.json.

Usage:
    python3 scripts/generate_grammar.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from larql_knowledge.ingest.grammar import save_grammar_pairs


def main() -> None:
    output = Path(__file__).parent.parent / "data" / "english_grammar.json"
    counts = save_grammar_pairs(output)
    total = sum(counts.values())
    print(f"Generated {len(counts)} categories, {total} pairs -> {output}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<25s} {n:>5d} pairs")


if __name__ == "__main__":
    main()
