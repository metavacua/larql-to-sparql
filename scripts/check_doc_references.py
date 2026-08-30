#!/usr/bin/env python3
"""Check the documentation references that `check_doc_links.py` cannot see.

That script validates markdown links, and reports all of them resolving —
truthfully. But a 2026-08-22 audit of 311 documents found ~90 broken
references, and **not one was a markdown link**. They were inline-code
paths, `file.rs:LINE` coordinates, `--example` names, `make` targets and
env-var names. The link gate is structurally blind to every one, which is
why that drift accumulated silently for months.

Rules, each chosen because the audit found real breakage of that shape:

  coord     `path.rs:120`      file must exist, and have >= 120 lines
  path      `crates/foo/bar.rs` inline paths under known roots must exist
  example   `--example NAME`   must be a real example, and any `-p CRATE`
                               beside it must be the crate that owns it
  bench     `--bench NAME`     must be a real bench target
  make      `make target`      must be a rule in the Makefile
  env       `LARQL_FOO`        must appear as a literal in crates/ or scripts/

Two design decisions keep the output worth reading rather than ignored:

  * **Resolve basenames before judging.** Docs legitimately write
    `hidden.rs:38` rather than the full path. Matching naively flags ~60%
    of coordinates as broken — pure noise. This resolves by unique
    basename or path-suffix and *skips* ambiguous ones, which is the
    difference between a gate people run and one they disable.
  * **Only match `make x` in backticks or at line start.** Unanchored, it
    fires on English ("make the", "make it", "make sure").

Limits, stated so nobody mistakes a green run for a correct corpus: this
catches the BROKEN class only. It cannot see the STALE class — "not
started" on shipped work, superseded numbers, wrong counts — which the
audit found to be both larger (~45 vs ~15) and more damaging, because a
broken path fails loudly while a stale status marker silently reroutes a
week of work.

Usage:
    python3 scripts/check_doc_references.py            # whole repo, report
    python3 scripts/check_doc_references.py --strict   # exit 1 on findings
    python3 scripts/check_doc_references.py docs/ AGENTS.md
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = ("/target/", "/.git/", "/.claude/", "/.venv/", "/node_modules/", "/coverage/")
# Historical records describe what was true when written; a renamed symbol
# or moved example in a CHANGELOG is not a defect to fix.
HISTORICAL = re.compile(r"CHANGELOG|/adr/|/audits/|/replay/|/baselines/")
SOURCE_EXT = (".rs", ".py", ".md", ".toml", ".json", ".sh", ".yml", ".msl")
# References into upstream projects and sibling repos are not ours to resolve.
# Without this the checker reports ~10 findings nobody can act on, which is
# how a gate earns its way onto the ignore list.
EXTERNAL = re.compile(
    r"^(transformers/|modeling_|streaming_|inferencer\.py|torch/|mlx/|llama\.cpp|chris-experiments/|"
    r"larql_probes/|~/|/Users/)"
)
PATH_ROOTS = ("crates/", "docs/", "bench/", "scripts/", "knowledge/", ".github/")
# A reference the prose has already LABELLED is not a trap, and this gate
# exists to catch traps. Two shapes qualify: a roadmap naming the file it
# intends to create, and a doc that says outright the thing is missing.
# Both leave the reader correctly informed, which is the whole point —
# whereas an unannotated dangling path sends someone looking for code that
# is not there.
#
# Without this the gate flags legitimate content as breakage, and a gate
# that cries wolf is one people switch off. Deliberately narrow: it wants
# an explicit marker on the same line, not a vague future tense.
PLANNED = re.compile(
    r"\(new\)|new `|\bgenerate\b|\bplanned\b|\bproposed\b|\bTODO\b|"
    r"\bnot started\b|will land|lands the|to be (?:written|created|added)|"
    r"does not exist|not yet|never committed|was never|no longer exists",
    re.I,
)


def index_files() -> tuple[dict, dict]:
    by_rel: dict[str, str] = {}
    by_base: dict[str, list[str]] = collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if any(s in dirpath + "/" for s in SKIP_DIRS):
            dirnames[:] = []
            continue
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
            by_rel[rel] = os.path.join(dirpath, name)
            by_base[name].append(rel)
    return by_rel, by_base


def resolve(ref: str, by_rel: dict, by_base: dict) -> list[str]:
    """Candidates for a possibly-abbreviated path. Empty = not found."""
    if ref in by_rel:
        return [ref]
    hits = [r for r in by_rel if r.endswith("/" + ref)]
    if hits:
        return hits
    return by_base.get(os.path.basename(ref), []) if "/" not in ref else []


def line_count(rel: str) -> int:
    try:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def cargo_targets(kind: str) -> dict[str, set[str]]:
    """Map target name -> owning crates, from Cargo.toml plus conventions."""
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "crates")):
        if any(s in dirpath + "/" for s in SKIP_DIRS):
            dirnames[:] = []
            continue
        if "Cargo.toml" not in filenames:
            continue
        crate = os.path.basename(dirpath)
        text = open(os.path.join(dirpath, "Cargo.toml")).read()
        for m in re.finditer(rf'\[\[{kind}\]\]\s*\nname\s*=\s*"([^"]+)"', text):
            owners[m.group(1)].add(crate)
        conv = os.path.join(dirpath, kind + "s")
        if os.path.isdir(conv):
            for sub, _d, fs in os.walk(conv):
                for f in fs:
                    if f.endswith(".rs"):
                        owners[f[:-3]].add(crate)
    return owners


def main() -> int:
    strict = "--strict" in sys.argv
    scope = [a for a in sys.argv[1:] if not a.startswith("-")]

    by_rel, by_base = index_files()
    examples = cargo_targets("example")
    benches = cargo_targets("bench")
    try:
        make_rules = set(
            re.findall(r"^([A-Za-z0-9_.-]+):", open(os.path.join(ROOT, "Makefile")).read(), re.M)
        )
    except OSError:
        make_rules = set()
    try:
        env_blob = subprocess.run(
            ["grep", "-rho", "LARQL_[A-Z0-9_]*", "crates", "scripts"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout
        known_env = set(env_blob.split())
    except OSError:
        known_env = set()

    docs = []
    for rel in sorted(by_rel):
        if not rel.endswith(".md") or HISTORICAL.search(rel):
            continue
        if scope and not any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in scope):
            continue
        docs.append(rel)

    findings: list[tuple[str, int, str, str]] = []
    counts: collections.Counter = collections.Counter()

    for rel in docs:
        try:
            lines = open(os.path.join(ROOT, rel), encoding="utf-8").read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for num, line in enumerate(lines, 1):
            # 1. `path.rs:LINE`
            for m in re.finditer(
                r"`([A-Za-z0-9_./-]+\.(?:rs|py|md|toml|json|sh|yml)):(\d+)(?:[-–]\d+)?`", line
            ):
                counts["coord"] += 1
                ref, want = m.group(1), int(m.group(2))
                if EXTERNAL.search(ref):
                    counts["coord"] -= 1
                    continue
                cand = resolve(ref, by_rel, by_base)
                if not cand:
                    findings.append((rel, num, "coord: no such file", m.group(0)))
                elif len(cand) == 1 and line_count(cand[0]) < want:
                    findings.append(
                        (rel, num, f"coord: {cand[0]} has {line_count(cand[0])} lines", m.group(0))
                    )
            # 2. inline paths under known roots
            for m in re.finditer(r"`((?:" + "|".join(PATH_ROOTS) + r")[A-Za-z0-9_./-]+)`", line):
                ref = m.group(1).rstrip("/")
                counts["path"] += 1
                if ref in by_rel or os.path.isdir(os.path.join(ROOT, ref)):
                    continue
                if not ref.endswith(SOURCE_EXT):
                    continue  # prose-ish path, or a directory that moved
                if not resolve(ref, by_rel, by_base) and not PLANNED.search(line):
                    findings.append((rel, num, "path: does not exist", m.group(0)))
            # 3. --example NAME, with its -p CRATE if present
            m = re.search(r"--example\s+([\w-]+)", line)
            if m:
                counts["example"] += 1
                name = m.group(1)
                owners = examples.get(name)
                pm = re.search(r"-p\s+([\w-]+)", line)
                if not owners:
                    if pm:  # bare `--example` may be an external repo's
                        findings.append((rel, num, "example: does not exist", name))
                elif pm and pm.group(1) not in owners:
                    findings.append(
                        (rel, num, f"example: lives in {sorted(owners)}", f"-p {pm.group(1)}")
                    )
            # 4. --bench NAME
            m = re.search(r"--bench\s+([\w-]+)", line)
            if m:
                counts["bench"] += 1
                name = m.group(1)
                owners = benches.get(name)
                pm = re.search(r"-p\s+([\w-]+)", line)
                if not owners:
                    findings.append((rel, num, "bench: does not exist", name))
                elif pm and pm.group(1) not in owners:
                    findings.append(
                        (rel, num, f"bench: lives in {sorted(owners)}", f"-p {pm.group(1)}")
                    )
            # 5. make targets — anchored, or prose fires constantly
            for m in re.finditer(r"(?:^|`)make\s+([a-z][a-z0-9-]{2,})", line):
                counts["make"] += 1
                # `make larql-<crate>-ci` is a documented PATTERN, not a
                # target; the regex stops at the `<` and would report the
                # truncated stem. A placeholder is correct content.
                if line[m.end():m.end() + 1] == "<":
                    counts["make"] -= 1
                    continue
                if make_rules and m.group(1) not in make_rules:
                    findings.append((rel, num, "make: no such target", m.group(1)))
            # 6. env vars
            for m in re.finditer(r"`(LARQL_[A-Z0-9_]+)`", line):
                counts["env"] += 1
                if known_env and m.group(1) not in known_env:
                    findings.append((rel, num, "env: not read anywhere", m.group(1)))

    checked = sum(counts.values())
    print(f"checked {checked} references across {len(docs)} documents")
    for rule, n in sorted(counts.items()):
        print(f"  {rule:<8} {n}")
    if not findings:
        print("\nno broken references")
        return 0
    print(f"\n{len(findings)} finding(s):\n")
    for rel, num, why, what in findings:
        print(f"  {rel}:{num}  {why}  —  {what}")
    print(
        "\nNote: this checks the BROKEN class only. Stale status markers and "
        "superseded numbers are invisible to it and are the larger problem."
    )
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
