#!/usr/bin/env python3
"""Verify every workflow file that fires on this branch has a workflow-level
concurrency: block with cancel-in-progress: true.

Companion to verify_gating_skip.py -- same reader/evaluate/report shape,
same exact-expected-set discipline (a file silently losing its
concurrency block must fail, not just "whatever's there is fine").

Root cause this exists for: none of the 15 pre-existing workflows Task 6
touched had a concurrency: block, so every push to this branch piled up a
new run per workflow instead of cancelling the stale one -- confirmed via
`gh run list`, 5-6 accumulated runs per workflow. Jobs inside those runs
are skipped (Task 6's if: gate), but the runs themselves never got
cancelled, which is real waste and PR-checks clutter, not just cosmetic.

Usage: verify_concurrency.py [ref]   (defaults to HEAD; reads via `git show`)
"""
import subprocess
import sys

import yaml

# All 17 workflow files that fire on this branch's PR (16 Task-6-gated +
# larql-cli-gating.yml itself). release.yml is excluded: tag-only trigger,
# never fires here, so an absent concurrency block there is irrelevant.
EXPECTED_FILES = [
    "bench-regress.yml",
    "larql-boundary.yml",
    "larql-cli-gating.yml",
    "larql-cli.yml",
    "larql-compute-metal.yml",
    "larql-compute.yml",
    "larql-core.yml",
    "larql-demos.yml",
    "larql-factory.yml",
    "larql-inference.yml",
    "larql-kv.yml",
    "larql-lql.yml",
    "larql-models.yml",
    "larql-server.yml",
    "larql-vindex.yml",
    "quality.yml",
    "shannon-verify.yml",
]


def git_show_reader(ref):
    def read(relpath):
        path = f".github/workflows/{relpath}"
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"{path} not found at {ref}: {result.stderr.strip()}")
        return result.stdout

    return read


def evaluate(read):
    missing_block = []  # file has no top-level concurrency: key at all
    missing_cancel = []  # file has concurrency: but cancel-in-progress isn't true
    missing_group = []  # file has concurrency: but no group key

    for relpath in EXPECTED_FILES:
        try:
            doc = yaml.safe_load(read(relpath))
        except FileNotFoundError:
            missing_block.append(relpath)
            continue
        concurrency = doc.get("concurrency")
        if concurrency is None:
            missing_block.append(relpath)
            continue
        if not isinstance(concurrency, dict) or "group" not in concurrency:
            missing_group.append(relpath)
        if not isinstance(concurrency, dict) or concurrency.get("cancel-in-progress") is not True:
            missing_cancel.append(relpath)

    return {
        "missing_block": missing_block,
        "missing_group": missing_group,
        "missing_cancel": missing_cancel,
        "expected_count": len(EXPECTED_FILES),
    }


def is_clean(result):
    return not (result["missing_block"] or result["missing_group"] or result["missing_cancel"])


def report(result, ref):
    print(f"ref={ref}: checked {result['expected_count']} workflow file(s)")
    if result["missing_block"]:
        print(f"NO concurrency: block at all in {len(result['missing_block'])} file(s):")
        for f in result["missing_block"]:
            print(f"  {f}")
    if result["missing_group"]:
        print(f"concurrency: present but missing 'group' in {len(result['missing_group'])} file(s):")
        for f in result["missing_group"]:
            print(f"  {f}")
    if result["missing_cancel"]:
        print(f"concurrency: present but cancel-in-progress != true in {len(result['missing_cancel'])} file(s):")
        for f in result["missing_cancel"]:
            print(f"  {f}")
    if is_clean(result):
        print(f"PASS: all {result['expected_count']} workflow files have a proper concurrency: block")


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    result = evaluate(git_show_reader(ref))
    report(result, ref)
    return 0 if is_clean(result) else 1


if __name__ == "__main__":
    sys.exit(main())
