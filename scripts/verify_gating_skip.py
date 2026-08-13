#!/usr/bin/env python3
"""Verify every job in the 16 branch-gated workflows skips on
gating/larql-cli-wasm-and-safe, and that the two excluded workflows
(release.yml, larql-cli-gating.yml) are untouched by the condition.

Checks against an EXACT expected (file, job) set, not just "every job
found has the condition" -- a job deleted outright must fail the check,
not silently reduce the count and still report PASS. See
scripts/test_verify_gating_skip.py for the mutation test that proves
this (docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md,
Task 6).

Usage: verify_gating_skip.py [ref]   (defaults to HEAD; reads via `git show`)
"""
import subprocess
import sys

import yaml

BRANCH = "gating/larql-cli-wasm-and-safe"
CONDITION_SUBSTR = f"github.head_ref != '{BRANCH}'"

# Exact expected (file, [job names]) set, captured from the repo at the
# commit Task 6 landed. A job appearing here that later disappears from
# the file (renamed, deleted, whole file gutted) must fail the check --
# that's the property scripts/test_verify_gating_skip.py mutates for.
#
# This dict is not regenerated from the live files by anything -- it's a
# one-time snapshot for this branch's discovery window (see Task 6 in
# docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md),
# not an ongoing CI gate (grep .github/workflows/ for this script's name:
# nothing invokes it). If a job is renamed/added on this branch after
# this snapshot, this dict will silently drift from reality -- acceptable
# for a short-lived verification script, but worth re-deriving before
# reusing this pattern anywhere longer-lived.
EXPECTED_JOBS = {
    "bench-regress.yml": ["bench"],
    "larql-boundary.yml": ["test", "coverage"],
    "larql-cli.yml": ["test", "coverage"],
    "larql-compute-metal.yml": ["test"],
    "larql-compute.yml": ["test", "coverage"],
    "larql-core.yml": ["test", "coverage"],
    "larql-demos.yml": ["test"],
    "larql-factory.yml": ["test", "coverage"],
    "larql-inference.yml": ["test", "coverage"],
    "larql-kv.yml": ["test", "coverage"],
    "larql-lql.yml": ["test", "coverage"],
    "larql-models.yml": ["test", "coverage"],
    "larql-server.yml": ["test", "coverage"],
    "larql-vindex.yml": ["test", "coverage"],
    "quality.yml": ["audit", "deny", "msrv", "doc-links", "proto-lint", "mutants"],
    "shannon-verify.yml": ["verify"],
}
EXPECTED_JOB_COUNT = sum(len(v) for v in EXPECTED_JOBS.values())
EXCLUDED_FILES = ["release.yml", "larql-cli-gating.yml"]


def git_show_reader(ref):
    """Return a reader(relpath) -> content, backed by `git show <ref>:<path>`."""

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
    """Core check, parameterized by a reader(relpath) -> yaml text.

    Returns a dict: missing_condition, missing_jobs, extra_unguarded, leaked,
    checked_jobs, expected_job_count. Pure function of `read` -- no git,
    no filesystem, no globals mutated -- so it's directly unit-testable
    against fabricated content (see scripts/test_verify_gating_skip.py).
    """
    missing_condition = []  # (file, job, if-string) -- job exists, condition absent
    missing_jobs = []  # (file, job) -- expected job doesn't exist at all anymore
    extra_unguarded = []  # (file, job) -- job exists, wasn't in EXPECTED, lacks condition
    leaked = []  # (file, job) -- excluded file picked up the condition
    checked_jobs = 0

    for relpath, expected_names in EXPECTED_JOBS.items():
        content = read(relpath)
        doc = yaml.safe_load(content)
        jobs = doc.get("jobs", {})

        for job_name in expected_names:
            if job_name not in jobs:
                missing_jobs.append((relpath, job_name))

        for job_name, job in jobs.items():
            checked_jobs += 1
            job_if = job.get("if", "") or ""
            has_condition = CONDITION_SUBSTR in job_if
            if not has_condition:
                if job_name in expected_names:
                    missing_condition.append((relpath, job_name, job_if))
                else:
                    extra_unguarded.append((relpath, job_name))

    for relpath in EXCLUDED_FILES:
        content = read(relpath)
        doc = yaml.safe_load(content)
        jobs = doc.get("jobs", {})
        for job_name, job in jobs.items():
            job_if = job.get("if", "") or ""
            if CONDITION_SUBSTR in job_if:
                leaked.append((relpath, job_name))

    return {
        "missing_condition": missing_condition,
        "missing_jobs": missing_jobs,
        "extra_unguarded": extra_unguarded,
        "leaked": leaked,
        "checked_jobs": checked_jobs,
        "expected_job_count": EXPECTED_JOB_COUNT,
    }


def is_clean(result):
    return not (
        result["missing_condition"]
        or result["missing_jobs"]
        or result["extra_unguarded"]
        or result["leaked"]
    )


def report(result, ref):
    print(
        f"ref={ref}: checked {result['checked_jobs']} job(s) present "
        f"across {len(EXPECTED_JOBS)} files "
        f"(expected exactly {result['expected_job_count']})"
    )
    if result["missing_jobs"]:
        print(f"MISSING JOB ENTIRELY (expected but not found) in {len(result['missing_jobs'])} case(s):")
        for relpath, job_name in result["missing_jobs"]:
            print(f"  {relpath}: job '{job_name}' no longer exists")
    if result["missing_condition"]:
        print(f"MISSING condition in {len(result['missing_condition'])} job(s):")
        for relpath, job_name, job_if in result["missing_condition"]:
            print(f"  {relpath}: job '{job_name}' if={job_if!r}")
    if result["extra_unguarded"]:
        print(f"UNGUARDED new job(s) not in the original expected set, {len(result['extra_unguarded'])} case(s):")
        for relpath, job_name in result["extra_unguarded"]:
            print(f"  {relpath}: job '{job_name}' has no gating condition")
    if result["leaked"]:
        print(f"UNEXPECTED condition leaked into {len(result['leaked'])} excluded job(s):")
        for relpath, job_name in result["leaked"]:
            print(f"  {relpath}: job '{job_name}'")

    if is_clean(result):
        print(
            f"PASS: all {result['expected_job_count']} expected jobs present and gated, "
            "excluded files clean"
        )


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    result = evaluate(git_show_reader(ref))
    report(result, ref)
    return 0 if is_clean(result) else 1


if __name__ == "__main__":
    sys.exit(main())
