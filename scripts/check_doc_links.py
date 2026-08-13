#!/usr/bin/env python3
"""Fail if any tracked Markdown file links to a path that does not exist.

Docs in this repo move between `docs/` and per-crate `crates/*/docs/`, and the
links pointing at them get left behind. That is how README.md ended up with
five dead spec links (PR #211) and the tree as a whole with 45 (2026-08-07).
A dead link is a silent defect: nothing fails, the reader just lands nowhere.

Scope: relative links only. External URLs are not checked — that needs the
network and is a different failure mode (rot vs. never-existed).

Usage:
    python3 scripts/check_doc_links.py            # check, exit 1 on any break
    python3 scripts/check_doc_links.py --list     # print every link checked
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Markdown inline links: [text](target). Reference-style and bare autolinks are
# not matched, deliberately — this is the form that carries a path.
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Links that are known-not-a-path and must not be reported.
#
# Each entry is (file, target, reason). Keep the reason: an allowlist without
# one becomes a place to hide real breakage.
ALLOW: list[tuple[str, str, str]] = [
    (
        "crates/larql-inference/docs/specs/async-compute-backend.md",
        "Self::flush",
        "rustdoc intra-doc link, not a file path",
    ),
    (
        "crates/larql-inference/docs/specs/async-compute-backend.md",
        "Self::read_hidden",
        "rustdoc intra-doc link, not a file path",
    ),
    (
        "crates/larql-inference/docs/specs/zone-engine.md",
        "./compiled-ffn.md",
        "the doc labels this '*to be written*' at the link site; it is a "
        "deliberate forward reference, not rot",
    ),
]


def tracked_markdown() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def targets_in(md: str) -> list[str]:
    text = (REPO / md).read_text(encoding="utf-8", errors="replace")
    found = []
    for raw in LINK.findall(text):
        target = raw.split()[0].strip()  # drop an optional "title"
        target = target.split("#", 1)[0]  # drop the anchor
        if not target:
            continue  # pure in-page anchor
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue  # http:, https:, mailto:, ...
        found.append(target)
    return found


def main() -> int:
    show_all = "--list" in sys.argv
    allow = {(f, t) for f, t, _ in ALLOW}

    broken: list[tuple[str, str]] = []
    checked = 0

    for md in tracked_markdown():
        base = (REPO / md).parent
        for target in targets_in(md):
            if (md, target) in allow:
                continue
            checked += 1
            resolved = (
                REPO / target.lstrip("/") if target.startswith("/") else base / target
            )
            ok = resolved.exists()
            if show_all:
                print(f"{'ok  ' if ok else 'DEAD'} {md} -> {target}")
            if not ok:
                broken.append((md, target))

    if broken:
        print(f"\n{len(broken)} broken documentation link(s):\n", file=sys.stderr)
        for md, target in broken:
            print(f"  {md} -> {target}", file=sys.stderr)
        print(
            "\nFix the path, or — if the target is deliberately not in the tree "
            "yet — add it to ALLOW in scripts/check_doc_links.py with a reason.",
            file=sys.stderr,
        )
        return 1

    print(f"all {checked} relative documentation links resolve "
          f"({len(ALLOW)} allowlisted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
