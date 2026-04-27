#!/usr/bin/env python3
"""Extract keyword->token pairs from Python stdlib source files.

Parses Python source with the built-in ast module and writes the results
to data/ast/python_ast.json.

Usage:
    python3 scripts/extract_ast_pairs.py
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from larql_knowledge.ingest.ast_extract import (
    extract_pairs_from_directory,
    save_ast_pairs,
)


def main() -> None:
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "ast" / "python_ast.json"

    # Use Python stdlib as the source corpus
    stdlib_dir = None

    # Try multiple ways to find the stdlib
    for candidate in [
        # Standard CPython
        Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        # macOS framework
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        # Fallback: sysconfig
        Path(__import__("sysconfig").get_path("stdlib")),
    ]:
        try:
            if candidate.exists() and any(candidate.glob("*.py")):
                stdlib_dir = candidate
                break
        except (TypeError, AttributeError):
            continue

    if stdlib_dir is None:
        # Last resort: use our own source code
        print("Could not find Python stdlib, falling back to project source")
        stdlib_dir = Path(__file__).parent.parent / "src"

    print(f"Scanning: {stdlib_dir}")
    data = extract_pairs_from_directory(stdlib_dir, max_files=200)

    print(f"Parsed {data['num_files']} files")
    total_pairs = sum(len(r["pairs"]) for r in data["relations"].values())
    print(f"Extracted {len(data['relations'])} relation types, {total_pairs} pairs")

    for keyword, rel_data in sorted(
        data["relations"].items(), key=lambda x: -len(x[1]["pairs"])
    ):
        print(f"  {keyword:<20s} {len(rel_data['pairs']):>5d} pairs")

    save_ast_pairs(data, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
