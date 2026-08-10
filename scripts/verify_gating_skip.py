#!/usr/bin/env python3
"""Verify every job in the 16 branch-gated workflows skips on
gating/larql-cli-wasm-and-safe, and that the two excluded workflows
(release.yml, larql-cli-gating.yml) are untouched by the condition.

Reads file content via `git show <ref>:<path>` rather than the working
tree, so it can be run against any historical commit to prove a RED/GREEN
transition (see docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md,
Task 6).

Usage: verify_gating_skip.py [ref]   (defaults to HEAD)
"""
import subprocess
import sys

import yaml

BRANCH = "gating/larql-cli-wasm-and-safe"
CONDITION_SUBSTR = f"github.head_ref != '{BRANCH}'"

GATED_FILES = [
    "bench-regress.yml",
    "larql-boundary.yml",
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
EXCLUDED_FILES = ["release.yml", "larql-cli-gating.yml"]


def read_file_at_ref(ref, relpath):
    path = f".github/workflows/{relpath}"
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"{path} not found at {ref}: {result.stderr.strip()}")
    return result.stdout


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    missing = []
    checked_jobs = 0

    for relpath in GATED_FILES:
        content = read_file_at_ref(ref, relpath)
        doc = yaml.safe_load(content)
        jobs = doc.get("jobs", {})
        for job_name, job in jobs.items():
            checked_jobs += 1
            job_if = job.get("if", "") or ""
            if CONDITION_SUBSTR not in job_if:
                missing.append((relpath, job_name, job_if))

    leaked = []
    for relpath in EXCLUDED_FILES:
        content = read_file_at_ref(ref, relpath)
        doc = yaml.safe_load(content)
        jobs = doc.get("jobs", {})
        for job_name, job in jobs.items():
            job_if = job.get("if", "") or ""
            if CONDITION_SUBSTR in job_if:
                leaked.append((relpath, job_name))

    print(f"ref={ref}: checked {checked_jobs} jobs across {len(GATED_FILES)} files")

    ok = True
    if missing:
        ok = False
        print(f"MISSING condition in {len(missing)} job(s):")
        for relpath, job_name, job_if in missing:
            print(f"  {relpath}: job '{job_name}' if={job_if!r}")
    if leaked:
        ok = False
        print(f"UNEXPECTED condition leaked into {len(leaked)} excluded job(s):")
        for relpath, job_name in leaked:
            print(f"  {relpath}: job '{job_name}'")

    if ok:
        print(f"PASS: all {checked_jobs} jobs gated, excluded files clean")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
