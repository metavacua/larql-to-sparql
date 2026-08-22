#!/usr/bin/env python3
"""Mechanical T-schema enforcement for `needs:` in
target-analysis-pipeline.yml (see the design spec's "`needs:` T-schema"
section, added 2026-08-21).

needs(j, k) is *correct* iff EXPR_EDGE(j, k) or ARTIFACT_EDGE(j, k):

  EXPR_EDGE(j, k)     -- the literal substring "needs.<k>." appears anywhere
                         in job j's own step bodies or its strategy.matrix
                         (never in the needs: list itself, which is exactly
                         the thing being checked) -- i.e. j dereferences one
                         of k's job-level `outputs:`. Always Content --
                         outputs are values, never mere existence signals.

  ARTIFACT_EDGE(j, k) -- j retrieves an artifact whose static name/pattern
                         matches a prefix k is known to upload, through one
                         of this repo's three real retrieval mechanisms:
                           (a) actions/download-artifact@v4 (with.name / with.pattern)
                           (b) `gh run download ... -n "<name>" ... -D <dir>`
                           (c) `gh api .../actions/artifacts ... name=<name>`
                         Classified Content if a known content-read call
                         (load_json(, load_jsonl(, .read_text(, open(, or a
                         bash `<` redirect / `cat`) appears anywhere in j's
                         steps at or after the retrieval step; Presence
                         otherwise (existence/filename-set use only:
                         .iterdir(, .glob(, .is_file(, f.name).

A cross-run fetch (actions/download-artifact with `run-id:` set) is outside
this schema's domain: it names a different run's job instance, which
`needs:` cannot express, so it is excluded from ARTIFACT_EDGE entirely
rather than misattributed to a same-run job of the same name.

Two properties, both checkable from the two edge relations above:

  SOUNDNESS    -- every real EXPR/ARTIFACT edge must be in the declared
                  needs: list. A violation is a real race.
  COMPLETENESS -- every declared needs: entry must be grounded by EXPR or
                  ARTIFACT. A violation costs only wall-clock time, never
                  correctness -- which is exactly why it can drift silently.

Closed-vocabulary limitation, stated rather than silently assumed away: the
retrieval-mechanism and content-read vocabularies above are enumerable
because they were built by direct inspection of every job in this one file.
This is not a general GitHub Actions data-flow analyzer. A future job that
retrieves an artifact or reads its content through a mechanism outside this
vocabulary must make `classify()` return "unrecognized" rather than silently
concluding "no dependency" -- `audit()` raises on any such case rather than
treating it as a clean result, so a genuinely new pattern fails loudly
instead of silently passing.
"""
from __future__ import annotations

import re
from typing import Any

CONTENT_READ_RE = re.compile(
    r"\bload_json\(|\bload_jsonl\(|\.read_text\(|\bopen\("
    r"|<\s*\"|\bcat\s"
)
EXISTENCE_ONLY_RE = re.compile(r"\.iterdir\(|\.glob\(|\.is_file\(|\bf\.name\b")


class UnrecognizedRetrievalError(Exception):
    """Raised when a job retrieves an artifact through a recognized
    mechanism but neither a known content-read nor existence-only call is
    found near it -- see the closed-vocabulary limitation above."""


def _step_text(step: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("run", "if"):
        v = step.get(key)
        if isinstance(v, str):
            parts.append(v)
    for block in (step.get("with") or {}, step.get("env") or {}):
        for v in block.values():
            if isinstance(v, str):
                parts.append(v)
    return "\n".join(parts)


def job_full_text(job: dict[str, Any]) -> str:
    """Includes strategy.matrix -- job-level expressions like
    `strategy.matrix.crate: ${{ fromJSON(needs.discovery.outputs.fmt-crates) }}`
    live outside steps: entirely and are real EXPR edges."""
    parts = [_step_text(s) for s in job.get("steps", [])]
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    parts.extend(v for v in matrix.values() if isinstance(v, str))
    return "\n".join(parts)


def upload_prefixes(job: dict[str, Any]) -> set[str]:
    prefixes = set()
    for step in job.get("steps", []):
        if (step.get("uses") or "").startswith("actions/upload-artifact"):
            name = (step.get("with") or {}).get("name", "")
            prefixes.add(re.split(r"\$\{\{", name)[0])
    return prefixes


def find_retrievals(job: dict[str, Any]) -> list[tuple[str, "str | None", int]]:
    """Returns (artifact_name_prefix, download_dir_or_None, step_index) for
    every same-run artifact retrieval in this job -- cross-run fetches
    (run-id: present) are excluded here, at the source, per the schema's
    domain restriction."""
    out: list[tuple[str, "str | None", int]] = []
    steps = job.get("steps", [])
    for i, step in enumerate(steps):
        uses = step.get("uses") or ""
        withb = step.get("with") or {}
        if uses.startswith("actions/download-artifact"):
            if "run-id" in withb:
                continue
            want = withb.get("name") or withb.get("pattern") or ""
            want_prefix = re.split(r"\$\{\{", want)[0].rstrip("*")
            out.append((want_prefix, withb.get("path", ""), i))
            continue
        text = _step_text(step)
        for m in re.finditer(r'gh run download\s[\s\S]*?-n\s+"([^"]+)"[\s\S]*?-D\s+(\S+)', text):
            out.append((re.split(r"\$", m.group(1))[0], m.group(2), i))
        for m in re.finditer(r'gh api\s+"?[^\n"]*actions/artifacts[^\n"]*"?[^\n]*name=([A-Za-z0-9_-]+)', text):
            out.append((m.group(1), None, i))
    return out


def classify(job: dict[str, Any], download_dir: "str | None", from_step_index: int) -> str:
    if download_dir is None:
        return "Content"  # gh api artifacts: the enumerated metadata is itself the consumed data
    later_text = "\n".join(_step_text(s) for s in job.get("steps", [])[from_step_index:])
    if CONTENT_READ_RE.search(later_text):
        return "Content"
    if EXISTENCE_ONLY_RE.search(later_text):
        return "Presence"
    return "unrecognized"


def grounded_edges(job_name: str, job: dict[str, Any], uploads: dict[str, set[str]]) -> dict[str, str]:
    """Maps k -> "Content"/"Presence" for every k this job (job_name, job)
    has a real EXPR or ARTIFACT edge to, given every job's upload prefixes."""
    text = job_full_text(job)
    grounded: dict[str, str] = {}
    for k in uploads:
        if k != job_name and re.search(rf"needs\.{re.escape(k)}\.", text):
            grounded[k] = "Content"
    for want_prefix, dl_dir, step_idx in find_retrievals(job):
        for k, prefixes in uploads.items():
            if k == job_name:
                continue
            if any(p and (p == want_prefix or p.rstrip("*") == want_prefix.rstrip("*")) for p in prefixes):
                kind = classify(job, dl_dir, step_idx)
                if kind == "unrecognized":
                    raise UnrecognizedRetrievalError(
                        f"{job_name} retrieves an artifact matching {k}'s upload "
                        f"(prefix~={want_prefix!r}) but neither a known content-read "
                        f"nor existence-only call was found near it"
                    )
                grounded[k] = kind
    return grounded


def audit(jobs: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Returns (soundness_violations, completeness_violations).

    soundness_violations: (job, k, kind) for every real edge missing from
      job's declared needs: list.
    completeness_violations: (job, k) for every declared needs: entry with
      no EXPR/ARTIFACT grounding found.
    """
    uploads = {name: upload_prefixes(body) for name, body in jobs.items()}
    soundness: list[tuple[str, str, str]] = []
    completeness: list[tuple[str, str]] = []
    for name, body in jobs.items():
        declared = set(body.get("needs") or [])
        grounded = grounded_edges(name, body, uploads)
        for k, kind in grounded.items():
            if k not in declared:
                soundness.append((name, k, kind))
        for k in declared:
            if k not in grounded:
                completeness.append((name, k))
    return soundness, completeness
