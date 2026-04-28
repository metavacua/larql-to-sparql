#!/usr/bin/env python3
"""
Deterministic changelog transformer: maps validated Conventional Commits to Keep a Changelog format.

This is a pure function transformer with no generative or heuristic logic.
Input: git commit range (already validated by commitlint)
Output: Markdown fragment for CHANGELOG.md [Unreleased] section

Mapping specification:
- feat:             -> ### Added
- fix:              -> ### Fixed
- BREAKING CHANGE:  -> ### Changed
- perf:             -> ### Changed
- refactor:         -> ### Changed
- docs:, chore:, test:, style: -> Omitted (internal quality)
"""

import subprocess
import re
import sys
from typing import Dict, List, Tuple


def get_commits(base: str, head: str) -> List[str]:
    """Fetch commit messages between base and head."""
    try:
        result = subprocess.run(
            ["git", "log", f"{base}...{head}", "--format=%B%n---COMMIT---"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.split("---COMMIT---")[:-1]
    except subprocess.CalledProcessError as e:
        print(f"Error fetching commits: {e}", file=sys.stderr)
        sys.exit(1)


def parse_conventional_commit(message: str) -> Tuple[str, str, str]:
    """
    Parse Conventional Commit format: type(scope): description
    Returns: (type, scope, description)
    """
    lines = message.strip().split("\n")
    first_line = lines[0]

    # Pattern: type(scope): description or type: description
    pattern = r"^(\w+)(?:\(([^)]+)\))?:\s+(.+)$"
    match = re.match(pattern, first_line)

    if not match:
        return "", "", first_line

    commit_type, scope, description = match.groups()
    return commit_type, scope or "", description


def extract_breaking_changes(message: str) -> bool:
    """Check if commit contains BREAKING CHANGE footer."""
    return "BREAKING CHANGE:" in message


def categorize_commit(
    commit_type: str, description: str, has_breaking: bool
) -> Tuple[str, str]:
    """
    Categorize commit into changelog category.
    Returns: (category_name, entry)
    """
    if has_breaking:
        return "Changed", description

    category_map = {
        "feat": "Added",
        "fix": "Fixed",
        "perf": "Changed",
        "refactor": "Changed",
    }

    category = category_map.get(commit_type)
    if category:
        return category, description

    return None, None


def generate_changelog(commits: List[str]) -> str:
    """
    Transform commits to changelog markdown.
    Pure function: identical input yields identical output.
    """
    categories: Dict[str, List[str]] = {
        "Added": [],
        "Changed": [],
        "Fixed": [],
    }

    for commit in commits:
        if not commit.strip():
            continue

        commit_type, _, description = parse_conventional_commit(commit)
        has_breaking = extract_breaking_changes(commit)
        category, entry = categorize_commit(commit_type, description, has_breaking)

        if category and entry:
            categories[category].append(entry)

    # Generate markdown in standard order
    output = []
    for section in ["Added", "Changed", "Fixed"]:
        if categories[section]:
            output.append(f"\n### {section}\n")
            for entry in sorted(categories[section]):  # Sort for determinism
                output.append(f"- {entry}")

    return "".join(output).lstrip()


def main():
    if len(sys.argv) < 3:
        print("Usage: changelog-transformer.py <base> <head>", file=sys.stderr)
        sys.exit(1)

    base = sys.argv[1]
    head = sys.argv[2]

    commits = get_commits(base, head)
    changelog_fragment = generate_changelog(commits)

    if changelog_fragment:
        print(changelog_fragment)
    else:
        print("No user-facing changes detected.", file=sys.stderr)


if __name__ == "__main__":
    main()
