#!/usr/bin/env python3
"""Unit tests for verify_gating_skip.py's core check logic.

No git, no filesystem, no subprocess: `evaluate()` is exercised against
fabricated in-memory workflow content via a plain dict reader, so these
tests run in milliseconds and mutate exactly the property under test.

The `old_style_check` function is a deliberate, literal transcription of
verify_gating_skip.py's pre-hardening logic (as committed in c2c508e6):
it only iterates jobs *present* in the parsed YAML and checks each one's
`if:` condition -- it never cross-references an expected job set. It
exists here to prove, empirically, the exact defect the hardening in
Task 6's follow-up fixed: deleting a job outright is invisible to it.

Run: python3 scripts/test_verify_gating_skip.py
"""
import sys

import yaml

from verify_gating_skip import (
    EXPECTED_JOBS,
    EXPECTED_JOB_COUNT,
    EXCLUDED_FILES,
    CONDITION_SUBSTR,
    evaluate,
    is_clean,
)

def _job_block(job_name, with_condition, merged_mutants=False):
    lines = [f"  {job_name}:", "    runs-on: ubuntu-latest"]
    if with_condition:
        if merged_mutants:
            lines.append(f"    if: github.event_name == 'pull_request' && {CONDITION_SUBSTR}")
        else:
            lines.append(f"    if: {CONDITION_SUBSTR}")
    lines.append("    steps: []")
    return "\n".join(lines)


def make_clean_fixture():
    """A dict {relpath: yaml text} matching EXPECTED_JOBS exactly, every
    job correctly gated (mutants merged, everything else pure), plus
    the two excluded files with no condition at all."""
    files = {}
    for relpath, job_names in EXPECTED_JOBS.items():
        blocks = [
            _job_block(name, with_condition=True, merged_mutants=(name == "mutants"))
            for name in job_names
        ]
        files[relpath] = "jobs:\n" + "\n".join(blocks) + "\n"
    for relpath in EXCLUDED_FILES:
        files[relpath] = "jobs:\n" + _job_block("build", with_condition=False) + "\n"
    return files


def reader_from_dict(files):
    def read(relpath):
        return files[relpath]

    return read


def old_style_check(read):
    """Literal transcription of the pre-hardening (c2c508e6) logic:
    checks only jobs present, never asks whether an expected job vanished.
    """
    missing = []
    for relpath in EXPECTED_JOBS:
        doc = yaml.safe_load(read(relpath))
        jobs = doc.get("jobs", {})
        for job_name, job in jobs.items():
            job_if = job.get("if", "") or ""
            if CONDITION_SUBSTR not in job_if:
                missing.append((relpath, job_name))
    leaked = []
    for relpath in EXCLUDED_FILES:
        doc = yaml.safe_load(read(relpath))
        jobs = doc.get("jobs", {})
        for job_name, job in jobs.items():
            job_if = job.get("if", "") or ""
            if CONDITION_SUBSTR in job_if:
                leaked.append((relpath, job_name))
    return not missing and not leaked  # True == "reports PASS"


def test_clean_fixture_passes():
    result = evaluate(reader_from_dict(make_clean_fixture()))
    assert is_clean(result), f"expected clean fixture to pass, got {result}"
    assert result["checked_jobs"] == EXPECTED_JOB_COUNT == 32


def test_condition_missing_on_existing_job_is_caught():
    files = make_clean_fixture()
    files["larql-core.yml"] = "jobs:\n" + _job_block("test", with_condition=True) + "\n" + _job_block(
        "coverage", with_condition=False
    )
    result = evaluate(reader_from_dict(files))
    assert not is_clean(result)
    assert ("larql-core.yml", "coverage", "") in result["missing_condition"]
    # Old logic already caught this class of defect -- not the regression under test.
    assert old_style_check(reader_from_dict(files)) is False


def test_new_unexpected_job_without_condition_is_caught():
    """A job added later that isn't in EXPECTED_JOBS at all (e.g. a fresh
    job someone appends to larql-core.yml after this script was written)
    must still be required to carry the gating condition -- it shouldn't
    be able to slip in ungated just because it predates no expectation."""
    files = make_clean_fixture()
    files["larql-core.yml"] = (
        "jobs:\n"
        + _job_block("test", with_condition=True)
        + "\n"
        + _job_block("coverage", with_condition=True)
        + "\n"
        + _job_block("lint", with_condition=False)
    )
    result = evaluate(reader_from_dict(files))
    assert not is_clean(result)
    assert ("larql-core.yml", "lint") in result["extra_unguarded"]
    assert result["checked_jobs"] == EXPECTED_JOB_COUNT + 1


def test_leaked_condition_into_excluded_file_is_caught():
    files = make_clean_fixture()
    files["release.yml"] = "jobs:\n" + _job_block("build", with_condition=True) + "\n"
    result = evaluate(reader_from_dict(files))
    assert not is_clean(result)
    assert ("release.yml", "build") in result["leaked"]


def test_job_deleted_entirely_is_caught_by_new_logic_but_not_old():
    """The mutation this hardening exists for: delete larql-core.yml's
    'coverage' job outright (not just strip its condition -- remove the
    job block completely, as a careless future edit might)."""
    files = make_clean_fixture()
    files["larql-core.yml"] = "jobs:\n" + _job_block("test", with_condition=True) + "\n"

    # RED, reproduced: the pre-hardening logic only iterates jobs that
    # exist. With 'coverage' gone, there's nothing to flag -- it reports
    # PASS on a file that silently lost its gate for an entire job.
    old_says_pass = old_style_check(reader_from_dict(files))
    assert old_says_pass is True, (
        "expected the old (pre-hardening) logic to be fooled by a deleted "
        "job -- if this fails, the 'old' transcription no longer matches "
        "history and this test's premise is stale"
    )

    # GREEN: the hardened evaluate() cross-references EXPECTED_JOBS and
    # must catch the missing job explicitly.
    result = evaluate(reader_from_dict(files))
    assert not is_clean(result), "hardened evaluate() failed to catch a deleted job"
    assert ("larql-core.yml", "coverage") in result["missing_jobs"]
    assert result["checked_jobs"] == EXPECTED_JOB_COUNT - 1


TESTS = [
    test_clean_fixture_passes,
    test_condition_missing_on_existing_job_is_caught,
    test_new_unexpected_job_without_condition_is_caught,
    test_leaked_condition_into_excluded_file_is_caught,
    test_job_deleted_entirely_is_caught_by_new_logic_but_not_old,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
        else:
            print(f"PASS: {test.__name__}")
    if failures:
        print(f"{failures}/{len(TESTS)} test(s) failed")
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
