#!/usr/bin/env python3
"""Validate all JSON files have correct format.

Checks:
  - All data/triples/*.json have required fields (relation, pairs)
  - All pairs are [str, str] lists
  - All JSON files parse without errors
  - No empty pair lists
  - data/ast/*.json have the expected structure
  - data/english_grammar.json has expected structure

Usage:
    python3 scripts/quality_check.py
"""

import json
import sys
from pathlib import Path


def check_triple_file(path: Path) -> list[str]:
    """Validate a single triple file. Returns list of error messages."""
    errors: list[str] = []

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"{path.name}: invalid JSON: {e}"]

    if not isinstance(data, dict):
        return [f"{path.name}: top-level must be a dict"]

    if "relation" not in data:
        errors.append(f"{path.name}: missing 'relation' field")

    if "pairs" not in data:
        errors.append(f"{path.name}: missing 'pairs' field")
        return errors

    pairs = data["pairs"]
    if not isinstance(pairs, list):
        errors.append(f"{path.name}: 'pairs' must be a list")
        return errors

    if len(pairs) == 0:
        errors.append(f"{path.name}: 'pairs' is empty")

    for i, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            errors.append(f"{path.name}: pair[{i}] must be [str, str], got {pair!r}")
            continue
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            errors.append(f"{path.name}: pair[{i}] values must be strings")

    return errors


def check_ast_file(path: Path) -> list[str]:
    """Validate an AST pairs file."""
    errors: list[str] = []

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"{path.name}: invalid JSON: {e}"]

    if "relations" not in data:
        errors.append(f"{path.name}: missing 'relations' field")
        return errors

    for rel_name, rel_data in data["relations"].items():
        if "pairs" not in rel_data:
            errors.append(f"{path.name}: relation '{rel_name}' missing 'pairs'")
            continue
        for i, pair in enumerate(rel_data["pairs"]):
            if not isinstance(pair, list) or len(pair) != 2:
                errors.append(
                    f"{path.name}: {rel_name}[{i}] must be [str, str]"
                )

    return errors


def check_grammar_file(path: Path) -> list[str]:
    """Validate the English grammar pairs file."""
    errors: list[str] = []

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"{path.name}: invalid JSON: {e}"]

    if "relations" not in data:
        errors.append(f"{path.name}: missing 'relations' field")
        return errors

    expected = {"determiner_noun", "preposition_noun",
                "copula_adjective", "auxiliary_verb"}
    found = set(data["relations"].keys())
    missing = expected - found
    if missing:
        errors.append(f"{path.name}: missing categories: {missing}")

    for cat, rel_data in data["relations"].items():
        if "pairs" not in rel_data:
            errors.append(f"{path.name}: category '{cat}' missing 'pairs'")

    return errors


def main() -> None:
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    all_errors: list[str] = []
    checks = 0

    # Check triple files
    triples_dir = data_dir / "triples"
    if triples_dir.exists():
        for f in sorted(triples_dir.glob("*.json")):
            errors = check_triple_file(f)
            all_errors.extend(errors)
            checks += 1
            status = "FAIL" if errors else "OK"
            print(f"  [{status}] triples/{f.name}")

    # Check AST files
    ast_dir = data_dir / "ast"
    if ast_dir.exists():
        for f in sorted(ast_dir.glob("*.json")):
            errors = check_ast_file(f)
            all_errors.extend(errors)
            checks += 1
            status = "FAIL" if errors else "OK"
            print(f"  [{status}] ast/{f.name}")

    # Check grammar file
    grammar_path = data_dir / "english_grammar.json"
    if grammar_path.exists():
        errors = check_grammar_file(grammar_path)
        all_errors.extend(errors)
        checks += 1
        status = "FAIL" if errors else "OK"
        print(f"  [{status}] english_grammar.json")

    print(f"\n{checks} files checked, {len(all_errors)} errors")
    if all_errors:
        print("\nErrors:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
