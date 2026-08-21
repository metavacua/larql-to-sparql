# Workflow Iteration-Time Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `target-analysis-pipeline.yml`'s real wall-clock cost without touching what it measures, how, or how often — by removing scheduling waste the workflow's own `needs:` graph carries for no real reason, and by eliminating one confirmed, large, avoidable cost inside `build-attempt`'s per-target loop.

**Architecture:** Two independent fixes, each mechanically verified before being written here (not asserted from memory): (1) a formal, code-enforced T-schema for the `needs:` keyword — every declared edge must be grounded by a real content-read or existence-check found in the job's own steps, checked by a real test against the live workflow file — removes 9 confirmed scheduling-only edges, unblocking Stage C to run concurrently with `build-attempt` instead of strictly after the whole Primary layer plus `indexing`. (2) `build-attempt` short-circuits its 6 cmd/feature closure-check attempts for any target `target-capability`'s own already-collected metadata proves lacks `std`, since `larql-cli`'s dependency closure (`reqwest`, unconditional) cannot build there regardless — recording an explicit, structured skip rather than paying `ring`'s native build script to re-derive an already-known outcome.

**Tech Stack:** GitHub Actions workflow YAML, Python 3.11 (stdlib only, `PyYAML` for the schema audit), pytest.

**Spec:** `docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md` — see its "Explicitly not doing" caching amendment, "`needs:` T-schema" section, and "`build-attempt` no-std short-circuit" section (all added 2026-08-21), which this plan implements.

## Global Constraints

- No `actions/cache` or `Swatinem/rust-cache` anywhere in the pipeline (spec: Explicitly
  not doing — real per-job timing data confirms toolchain install is not the cost, and
  the one cache that would meaningfully cut real cost, build artifacts across runs, is
  the single highest measurement-validity risk available given Stage B's per-run
  mutation).
- No commits from CI, ever, anywhere in either layer.
- Validated by an actual run on GitHub-hosted runners, never by local simulation alone —
  every task below ends with a real push and real `gh api`/`gh run` evidence, not just
  a passing local test.
- Every relevant probe still runs unconditionally, every run (Standing Principle 3) —
  neither task changes which targets get measured or how; both only change scheduling
  and one job's internal control flow for a target whose real outcome is already
  provably fixed by data the pipeline already collects.
- No agent-authored curation presented as mechanically-grounded (L1) — the "9 violations"
  and "`ring` dominates this cost" claims below are both real, run-derived measurements
  quoted with their evidence, not asserted from memory.

---

### Task 1: Add the `needs:` T-schema audit as a real test, and fix the 9 completeness violations it finds

**Real, verified finding (2026-08-21).** `needs:` in GitHub Actions is a pure scheduling
primitive — nothing in the platform checks it against what a job's own steps actually
read. A direct, mechanically verified audit of this workflow's real `needs:` graph
against what each job's steps actually reference (built and tested against the live
file, not asserted) found the graph **sound everywhere** (zero real content/existence
dependencies missing from any job's declared `needs:` — no races) but **incomplete in
9 of its ~23 real edges**, across 6 of its 12 jobs: `target-capability`,
`dependency-graph`, `build-attempt`, and `runtime-test` each declare `needs:
target-independent-checks` despite never reading anything from it;
`secondary-mutate` declares `needs: [discovery, indexing]` despite reading from
neither (a pure, self-contained source rewrite of its own checkout);
`secondary-stage-c-and-promotion` declares `needs: indexing` despite never reading
anything from it (redundant besides, since it's already implied transitively via
`secondary-mutate`); `fetch-prior-round-baseline` and `secondary-layer-self-test` each
declare `needs: discovery` despite reading nothing from it either. Real timing evidence
from run `32441374624`: `secondary-mutate` doesn't start until t=905s (it needs only a
checkout — its own 50s of work could start at t≈0), and Stage C doesn't start until
t=956s, purely because it's forced to wait for `indexing`, which itself waits for the
*entire* Primary layer (`build-attempt`'s longest batch alone runs until t≈874s).
Dropping the 2 edges on `secondary-mutate` and the 1 on Stage C removes that forced
wait entirely — Stage C's only *real* prerequisites are `secondary-mutate` (~50s) and
`dependency-graph` (~250s), so it could start around t≈250–300s instead of t≈956s.

**Files:**
- Create: `scripts/target_analysis_needs_schema.py`
- Create: `tests/test_target_analysis_needs_schema.py`
- Modify: `.github/workflows/target-analysis-pipeline.yml` (6 jobs' `needs:` lines)

**Interfaces:**
- Consumes: nothing new.
- Produces: `audit(jobs: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]` (soundness violations, completeness violations) — used only by this task's own test, nothing downstream relies on it.

- [ ] **Step 1: Write `scripts/target_analysis_needs_schema.py`**

```python
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
```

- [ ] **Step 2: Write `tests/test_target_analysis_needs_schema.py`**

```python
from pathlib import Path

import yaml

from scripts.target_analysis_needs_schema import audit

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "target-analysis-pipeline.yml"


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def test_needs_graph_is_sound():
    # No job may read (Content) or require-the-existence-of (Presence) an
    # artifact from a job it does not declare needs: on -- a violation here
    # is a real race, not just wasted time.
    soundness, _ = audit(_jobs())
    assert soundness == [], (
        f"jobs read from other jobs' artifacts without declaring needs: on them "
        f"(a real race): {soundness}"
    )


def test_needs_graph_is_complete():
    # Every declared needs: entry must be grounded by a real Content or
    # Presence edge -- an ungrounded entry costs only wall-clock time, but
    # costs it forever, silently, since nothing else checks this.
    _, completeness = audit(_jobs())
    assert completeness == [], (
        f"jobs declare needs: on other jobs they never read from or check "
        f"the existence of -- pure scheduling waste: {completeness}"
    )
```

- [ ] **Step 3: Run the tests and confirm the exact expected RED state**

Run: `pytest tests/test_target_analysis_needs_schema.py -v`

Expected: `test_needs_graph_is_sound` PASSES (0 real races in the current file — confirmed
by direct construction of this test against the live file before this plan was written).
`test_needs_graph_is_complete` FAILS, with exactly these 9 violations in its assertion
message: `('target-capability', 'target-independent-checks')`,
`('dependency-graph', 'target-independent-checks')`,
`('build-attempt', 'target-independent-checks')`,
`('runtime-test', 'target-independent-checks')`, `('secondary-mutate', 'indexing')`,
`('secondary-mutate', 'discovery')`, `('secondary-stage-c-and-promotion', 'indexing')`,
`('fetch-prior-round-baseline', 'discovery')`, `('secondary-layer-self-test', 'discovery')`.
If the actual failure differs from this list, stop and diagnose before proceeding — the
fix in Step 4 is written against this exact list.

- [ ] **Step 4: Fix the workflow — remove the 9 ungrounded `needs:` tokens**

In `.github/workflows/target-analysis-pipeline.yml`, make exactly these 6 edits (line
numbers as of this plan's writing; find each job by name if line numbers have since
shifted):

`target-capability` (line 117): change
```yaml
    needs: [discovery, target-independent-checks]
```
to
```yaml
    needs: [discovery]
```

`dependency-graph` (line 160): the same change (`needs: [discovery,
target-independent-checks]` → `needs: [discovery]`).

`build-attempt` (line 205): change
```yaml
    needs: [discovery, target-capability, target-independent-checks]
```
to
```yaml
    needs: [discovery, target-capability]
```

`runtime-test` (line 281): the same change as `target-capability`/`dependency-graph`
(`needs: [discovery, target-independent-checks]` → `needs: [discovery]`).

`secondary-mutate` (line 526): delete the `needs:` line entirely (it needs neither
`discovery` nor `indexing` — its steps are a fully self-contained rewrite of its own
checkout). The job definition becomes:
```yaml
  secondary-mutate:
    name: "Secondary layer: mutate (Stages A, B, B2, B3)"
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
```

`secondary-stage-c-and-promotion` (line 610): change
```yaml
    needs: [discovery, indexing, secondary-mutate, dependency-graph, fetch-prior-round-baseline]
```
to
```yaml
    needs: [discovery, secondary-mutate, dependency-graph, fetch-prior-round-baseline]
```

`fetch-prior-round-baseline` (line 784): delete the `needs:` line entirely. The job
definition becomes:
```yaml
  fetch-prior-round-baseline:
    name: Fetch prior round's baseline (cross-run artifact lookup)
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
```

`secondary-layer-self-test` (line 927): delete the `needs:` line entirely. The job
definition becomes:
```yaml
  secondary-layer-self-test:
    name: Secondary-layer self-test (noise floor, blast radius, ephemerality)
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
```

- [ ] **Step 5: Validate YAML and re-run the tests, confirm both now pass**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
pytest tests/test_target_analysis_needs_schema.py -v
```
Expected: `YAML OK`, no `actionlint` errors, both tests PASS (0 soundness violations, 0
completeness violations).

- [ ] **Step 6: Run the full existing test suite to confirm no other regression**

```bash
pytest tests/ -v
```
Expected: every existing test still passes — this task touches no script logic, only
the workflow YAML's `needs:` lines, so no other test's behavior should change.

- [ ] **Step 7: Commit**

```bash
git add scripts/target_analysis_needs_schema.py tests/test_target_analysis_needs_schema.py .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: remove 9 ungrounded needs: edges, mechanically found and enforced by a real T-schema audit test -- unblocks Stage C from waiting on the whole Primary layer plus indexing for no real reason"
```

- [ ] **Step 8: Push and verify on a real runner**

Push and, from real job-start timestamps (`gh api repos/<owner>/<repo>/actions/runs/<run_id>/jobs?per_page=100`, comparing each job's `started_at`), confirm: (a) `secondary-mutate` now starts within a few seconds of the run beginning, not after the whole Primary layer; (b) `secondary-stage-c-and-promotion`'s first batch now starts once `secondary-mutate` and `dependency-graph` are both done, not after `indexing`; (c) every job in the run still completes successfully, same as before; (d) report the real before/after total run wall-clock (compare against a prior run's `completed_at - started_at` across all jobs) — do not assume the improvement, measure it from the real run.

---

### Task 2: Short-circuit `build-attempt`'s guaranteed-to-fail no-std closure checks

**Real, verified finding (2026-08-21).** `build-attempt` checks/clippys/builds
`larql-cli` against every real target, 6 cmd×feature combinations each, with
`--keep-going`. `crates/larql-cli/Cargo.toml`'s `reqwest` dependency carries no
`optional = true` and is not gated behind any `[features]` entry (confirmed by direct
inspection) — so for any target `target-capability`'s own `target-spec-<target>.json`
already proves lacks `std` (`metadata.std == false`, computed unconditionally for every
target, every run, already), all 6 combinations are a guaranteed failure before they
even start. Real job-log evidence (run `32441374624`, job "Build attempt: batch 3",
target `armv7a-none-eabihf`): the first of the 6 invocations took 3m40s total, of
which the *last* 3m19s (91%) is silent — no compiler output at all — bracketed by
`ring v0.17.14`'s build script starting just before the gap and a first-party
workspace crate (`larql-router-protocol`) appearing right after it ends. `--keep-going`
is exactly why this cost is paid at all: without it, cargo would likely stop at the
first no-std error; with it, cargo works through the entire dependency graph —
including `ring`'s expensive native build script — before giving up, on a target whose
outcome was already knowable for free.

**This must not be confused with `indexing`'s existing `unexpected_clean_std_build`
contradiction check** (Standing Principle 5, the nvptx canary), which watches for a
target reporting zero errors *despite an actual attempt*. A deliberate skip is a third,
distinct outcome — neither a pass nor that contradiction — and must be recorded so it's
never indistinguishable from either.

**Files:**
- Create: `scripts/target_analysis_build_attempt.py`
- Modify: `scripts/target_analysis_common.py` (add `SKIP_MARKER_KEY`, `is_skip_record`)
- Modify: `scripts/target_analysis_indexing.py` (add `combined_record_for_target`)
- Modify: `tests/test_target_analysis_indexing.py` (add tests; existing tests unchanged)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`build-attempt`'s per-target
  loop; `indexing`'s "Run indexing script per target" step)

**Interfaces:**
- Consumes: `build_attempt_filenames(target: str) -> list[str]` (existing, from
  `target_analysis_indexing.py`).
- Produces: `guaranteed_std_failure(target_spec: dict[str, Any]) -> bool`,
  `skip_record(target: str) -> dict[str, Any]` (both in
  `target_analysis_build_attempt.py`); `is_skip_record(compiler_messages: list[Any]) ->
  bool` (in `target_analysis_common.py`); `combined_record_for_target(messages:
  list[Any], target_spec: dict[str, Any], missing_sorted: list[str]) -> dict[str, Any]`
  (in `target_analysis_indexing.py`) — the workflow's own per-target step calls this
  directly instead of constructing `combined[t]` inline.

- [ ] **Step 1: Add `SKIP_MARKER_KEY` and `is_skip_record` to `scripts/target_analysis_common.py`**

Add this constant near the top of the file (after the imports) and this function at the
end of the file:

```python
SKIP_MARKER_KEY = "skipped_no_std_guaranteed_fail"
```

```python
def is_skip_record(compiler_messages: list[Any]) -> bool:
    """True iff this is a deliberate build-attempt skip record (see
    scripts/target_analysis_build_attempt.py's skip_record()), not a real
    cargo --message-format=json stream. Checked first by anything that
    would otherwise treat an empty error list as evidence of a clean build
    -- a skip is neither a pass nor a contradiction, and must never be
    indistinguishable from either."""
    return (
        bool(compiler_messages)
        and isinstance(compiler_messages[0], dict)
        and compiler_messages[0].get(SKIP_MARKER_KEY) is True
    )
```

- [ ] **Step 2: Write `scripts/target_analysis_build_attempt.py`**

```python
#!/usr/bin/env python3
"""Short-circuit logic for build-attempt's per-target loop (see the design
spec's "build-attempt no-std short-circuit" section, added 2026-08-21).

larql-cli's real dependency closure (reqwest -> rustls -> ring, tokenizers,
wasmtime, ...) carries no `[features]` gate in crates/larql-cli/Cargo.toml
that removes it -- confirmed by direct inspection 2026-08-21. Any target
target-capability's own metadata already proves lacks `std` is therefore a
guaranteed failure before the attempt starts: larql-cli cannot build
without std under any of the 6 cmd/feature combinations that exist today.
Real CI log evidence (run 32441374624, job "Build attempt: batch 3"):
`ring`'s native build script ran silently for ~3m19s of one target's
~3m40s total invocation on armv7a-none-eabihf -- ~91% of that target's own
cost, paid to re-derive a fact target-capability already recorded for free
in the same run.
"""
from __future__ import annotations

from typing import Any

from scripts.target_analysis_common import SKIP_MARKER_KEY

SKIP_REASON = (
    "target-capability's own metadata.std is false for this target, and "
    "larql-cli's reqwest dependency (crates/larql-cli/Cargo.toml) carries "
    "no feature gate that removes it -- this attempt is a guaranteed "
    "failure whose real cost is dominated by ring's native build script, "
    "not new information."
)


def guaranteed_std_failure(target_spec: dict[str, Any]) -> bool:
    return target_spec.get("metadata", {}).get("std") is False


def skip_record(target: str) -> dict[str, Any]:
    return {
        SKIP_MARKER_KEY: True,
        "target": target,
        "skip_reason": SKIP_REASON,
    }
```

- [ ] **Step 3: Add `combined_record_for_target` to `scripts/target_analysis_indexing.py`**

Change its import line from:
```python
from scripts.target_analysis_common import error_level_messages
```
to:
```python
from scripts.target_analysis_common import error_level_messages, is_skip_record
```

Add this function at the end of the file:

```python
def combined_record_for_target(
    messages: list[Any],
    target_spec: dict[str, Any],
    missing_sorted: list[str],
) -> dict[str, Any]:
    """The per-target record indexing writes into index.json. A deliberate
    build-attempt skip (see scripts/target_analysis_build_attempt.py) is
    checked first and short-circuits to its own distinct shape -- it must
    never be indistinguishable from a real, attempted, zero-error build,
    which is exactly what unexpected_clean_std_build's contradiction check
    (Standing Principle 5) watches for."""
    if is_skip_record(messages):
        return {
            "skipped_no_std_guaranteed_fail": True,
            "missing_artifacts": missing_sorted,
        }
    std_mode_errors = [message for _entry, message in error_level_messages(messages)]
    return {
        "error_counts_by_target": count_errors_by_target(messages),
        "contradictions": {
            "unexpected_clean_std_build": unexpected_clean_std_build(target_spec, std_mode_errors),
        },
        "missing_artifacts": missing_sorted,
    }
```

- [ ] **Step 4: Add tests to `tests/test_target_analysis_indexing.py`**

Add these imports (alongside the existing ones at the top of the file):
```python
from scripts.target_analysis_build_attempt import guaranteed_std_failure, skip_record
from scripts.target_analysis_indexing import combined_record_for_target
```
(add `combined_record_for_target` to the existing `from scripts.target_analysis_indexing import (...)` block rather than a separate line, if that's how the file already imports from that module).

Add these test functions:

```python
def test_guaranteed_std_failure_true_when_target_capability_says_no_std():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert guaranteed_std_failure(target_spec) is True


def test_guaranteed_std_failure_false_when_target_has_std():
    assert guaranteed_std_failure({"metadata": {"std": True}}) is False


def test_skip_record_carries_the_marker_key_and_target():
    record = skip_record("armv7a-none-eabihf")
    assert record["skipped_no_std_guaranteed_fail"] is True
    assert record["target"] == "armv7a-none-eabihf"
    assert "ring" in record["skip_reason"] or "std" in record["skip_reason"]


def test_combined_record_for_target_reports_skip_not_a_contradiction():
    # Load-bearing case: a skip must NEVER be reported as
    # unexpected_clean_std_build's contradiction (Standing Principle 5),
    # which specifically watches for zero errors despite an ACTUAL attempt.
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")  # metadata.std is False
    messages = [skip_record("armv7a-none-eabihf")]
    record = combined_record_for_target(messages, target_spec, missing_sorted=[])
    assert record["skipped_no_std_guaranteed_fail"] is True
    assert "contradictions" not in record
    assert "error_counts_by_target" not in record


def test_combined_record_for_target_preserves_existing_behavior_when_not_skipped():
    # Exact regression check: a real (non-skip) message list must produce
    # the identical shape the pre-existing inline workflow logic always did.
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    target_spec = {"metadata": {"std": True}}
    record = combined_record_for_target(messages, target_spec, missing_sorted=["x"])
    assert record == {
        "error_counts_by_target": {"larql-boundary": 2},
        "contradictions": {"unexpected_clean_std_build": False},
        "missing_artifacts": ["x"],
    }
```

- [ ] **Step 5: Run the tests, confirm all pass (9 existing + 5 new = 14 total)**

```bash
pytest tests/test_target_analysis_indexing.py -v
```
Expected: 14 passed, 0 failed. If any of the 9 pre-existing tests fail, stop — that
means Step 3's change altered existing behavior, which it must not.

- [ ] **Step 6: Wire the short-circuit into `build-attempt`'s per-target loop**

In `.github/workflows/target-analysis-pipeline.yml`'s `build-attempt` job, in the "Build-attempt probes for every target/cmd/features combination in this batch" step, initialize a counter alongside the existing `FAILED=0`:

```bash
          FAILED=0
          SKIPPED=0
```

Immediately after the existing crate-type canary block (the `if [ -n "$FIRST_CRATE_TYPE" ]; then ... fi` block) and before the existing `BUILD_STD=none` line, insert:

```bash
            GUARANTEED_FAIL=$(python3 - "$TARGET" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from scripts.target_analysis_common import load_json
from scripts.target_analysis_build_attempt import guaranteed_std_failure

target = sys.argv[1]
target_spec = load_json(Path(f"capability/target-spec-{target}.json"))
print("true" if guaranteed_std_failure(target_spec) else "false")
PYEOF
)
            if [ "$GUARANTEED_FAIL" = "true" ]; then
              echo "=== target=$TARGET: skipping the 6 cmd/feature closure checks -- target-capability's own metadata already proves std is unavailable and larql-cli's reqwest dependency is unconditional (crates/larql-cli/Cargo.toml, confirmed 2026-08-21) ==="
              SKIPPED=$((SKIPPED+1))
              python3 - "$TARGET" <<'PYEOF'
import json
import sys
sys.path.insert(0, ".")
from pathlib import Path
from scripts.target_analysis_build_attempt import skip_record
from scripts.target_analysis_indexing import build_attempt_filenames

target = sys.argv[1]
for filename in build_attempt_filenames(target):
    Path(f"out/{filename}").write_text(json.dumps(skip_record(target)))
PYEOF
              continue
            fi
```

At the end of the same step, change:
```bash
          echo "batch-had-failure=$FAILED" >> "$GITHUB_OUTPUT"
```
to:
```bash
          echo "batch-had-failure=$FAILED" >> "$GITHUB_OUTPUT"
          echo "batch-skipped-count=$SKIPPED" >> "$GITHUB_OUTPUT"
```

And add `batch-skipped-count` to the step's `id: attempt` output usage in the following "Assert honest result" step, appending this line to it:
```bash
          echo "batch ${{ matrix.batch_index }} skipped ${{ steps.attempt.outputs.batch-skipped-count }} target(s) via the guaranteed-std-failure short-circuit"
```

- [ ] **Step 7: Wire `combined_record_for_target` into `indexing`'s per-target step**

In the same workflow file's `indexing` job, "Run indexing script per target" step, change the import line from:
```python
          from scripts.target_analysis_common import error_level_messages, load_json, load_jsonl
          from scripts.target_analysis_indexing import (
              build_attempt_filenames,
              count_errors_by_target,
              missing_artifacts,
              unexpected_clean_std_build,
          )
```
to:
```python
          from scripts.target_analysis_common import load_json, load_jsonl
          from scripts.target_analysis_indexing import (
              build_attempt_filenames,
              combined_record_for_target,
              missing_artifacts,
          )
```

Change this block:
```python
              std_mode_errors = [message for _entry, message in error_level_messages(messages)]
              target_spec = load_json(spec_file)

              combined[t] = {
                  "error_counts_by_target": count_errors_by_target(messages),
                  "contradictions": {
                      "unexpected_clean_std_build": unexpected_clean_std_build(target_spec, std_mode_errors),
                  },
                  "missing_artifacts": missing_sorted,
              }
```
to:
```python
              target_spec = load_json(spec_file)
              combined[t] = combined_record_for_target(messages, target_spec, missing_sorted)
```

- [ ] **Step 8: Validate and run the full test suite**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
pytest tests/ -v
```
Expected: `YAML OK`, no `actionlint` errors, every test passes (including the 14 in
`test_target_analysis_indexing.py` and the 2 in `test_target_analysis_needs_schema.py`
from Task 1).

- [ ] **Step 9: Commit**

```bash
git add scripts/target_analysis_build_attempt.py scripts/target_analysis_common.py scripts/target_analysis_indexing.py tests/test_target_analysis_indexing.py .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: short-circuit build-attempt's guaranteed-to-fail no-std closure checks -- records the same real fact as an explicit skip instead of paying ring's native build script to re-derive it, never conflated with the nvptx-canary contradiction check"
```

- [ ] **Step 10: Push and verify on a real runner**

Push and confirm, from real job logs (not assumed): the `armv7a-none-eabihf`-class
targets (any target whose `target-spec-<target>.json` shows `metadata.std: false`) now
show the `"=== target=... skipping ..."` message and complete in seconds rather than
minutes; the batch's own real duration drops (compare `gh api .../jobs` timing against
a prior run's same batch); `index.json` (download the `navigation-index` artifact)
shows `"skipped_no_std_guaranteed_fail": true` for those targets and no
`"contradictions"` key for them, while a target with real `std` support is completely
unaffected (identical `error_counts_by_target`/`contradictions` shape as before); the
batch's own "Assert honest result" step reports a nonzero skipped-count. Report the
real skipped-count and the real before/after batch duration — do not assume the
speedup, measure it.
