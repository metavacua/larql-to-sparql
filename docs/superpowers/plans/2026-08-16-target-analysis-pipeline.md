# Generalized Target-Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad hoc `experiment-cuda-nvptx.yml` workflow with a target-parameterized, fully autonomous GitHub Actions pipeline that, given a target triple, produces a complete map of every place gating or fixing is necessary to build and runtime-test for that target — and implements the mutually-recursive Primary(observe)/Secondary(mutate) loop with a precisely defined, mechanical promotion rule.

**Architecture:** A Discovery job resolves the real target list and batches it (GitHub Actions caps `strategy.matrix` at 256 jobs; the real target count is 331) into `batch_index` matrices consumed via `fromJSON()` by four Primary-layer probe job families (target-capability, dependency-graph, build-attempt, runtime-test), each internally looping over its batch's targets in a shell loop rather than fanning `strategy.matrix` directly over every target. An Indexing job structurally aggregates every probe's raw per-target files (now nested inside per-batch artifacts) with loud-failure-on-missing-file completeness checking. A single, target-independent Secondary-layer mutation job runs the existing four-stage pipeline (Stages A/B/B2/B3 are all target-independent) exactly once, producing one patch; a batched Stage C job downloads and applies that patch before checking each target, captures before/after Primary-layer diffs per stage, and applies a mechanical promotion/depth-advancement rule to decide what folds into the next round's shared baseline.

**Tech Stack:** GitHub Actions YAML (`needs:`, `strategy.matrix` + `fromJSON()`, `background`/`wait` steps, `actions/upload-artifact`), Python 3 (stdlib only — `json`, `argparse`, `pathlib`), pytest for script unit tests, existing Rust/Cargo/rustc toolchain invocations already proven in `experiment-cuda-nvptx.yml`.

**Spec:** `docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md`

## Global Constraints

- No `actions/cache` or `Swatinem/rust-cache` anywhere in the pipeline (spec: Explicitly not doing).
- No `git commit`/`git push` from any CI job, in either layer — mechanically greppable absence (spec: Explicitly not doing, Testing).
- No agent-mediated filtering, summarizing, or relevance-judgment inside the pipeline; every probe's full raw output is preserved and retrievable via `actions/upload-artifact` (Standing Principle 2).
- No exclusion of a probe or tool based on dev-machine availability — every probe targets the GitHub-hosted runner only (Standing Principle 1).
- Every relevant probe runs unconditionally every time; nothing is skipped because an earlier result seems to already explain things (Standing Principle 3).
- No aggregation step collapses one probe's verdict into another's — disagreement between independently mechanically-grounded sources is preserved and surfaced, never merged away (Standing Principle 4).
- `nvptx64-nvidia-cuda` is a standing canary: any probe reporting unexpected clean/success against it is presumptively a bug in that probe, not progress (Standing Principle 5).
- `needs:`/`wait` express only genuine mechanical prerequisites, never judgment-based ordering (Standing Principle 8).
- The Discovery job and the Indexing job are plain `run:` steps executed by the runner autonomously on every run, with zero human or agent intervention beyond the initial trigger (Standing Principle 9).
- Every curated (L2) data source (`deny-nvptx.toml`, the target-family tooling registry) is explicitly labeled as curated and checkable against raw, uncurated scan output (Foundational framework).
- All claims about GitHub Actions mechanics or probe behavior are validated by an actual run on a GitHub-hosted runner, never by local simulation (Validation approach).
- Trigger is `push` alone, scoped to the pipeline's branch pattern, with `workflow_dispatch` as a secondary single-target convenience path; no `pull_request` trigger (Data flow: Triggers).
- Concurrency and least-privilege are standing requirements on every job and the workflow itself, not a one-time fix (user directive): the workflow declares `concurrency: {group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true}` so a new push cancels a stale in-flight run on the same branch instead of queue-contending with it (a real, observed cost — Task 12's real run lost ~40 minutes to exactly this). Every job declares an explicit, minimal `permissions:` block — `contents: read` as the floor (checkout), with `actions: read` added only for jobs that actually call `actions/download-artifact` (the empirically-established pattern from the `indexing` job, Task 10) — never left unset to inherit whatever the repository's own default happens to be.

---

## File Structure

- `.github/workflows/target-analysis-pipeline.yml` — the new, generalized pipeline: `discovery` (also computes batches, Task 6), `target-capability`, `dependency-graph`, `build-attempt`, `runtime-test` (all four batched over `batch_index`, Tasks 7-9), `indexing` (Task 10), `secondary-mutate` (Stages A/B/B2/B3, single job, target-independent, Task 16), `secondary-stage-c-and-promotion` (batched, applies the mutation patch, Task 17), `next-round-baseline` (Task 20), `secondary-layer-self-test` (Task 21).
- `scripts/target_analysis_common.py` — shared, dependency-free helpers: JSON loading, `--message-format=json` compiler-message parsing into `(file, line, code)` error-site tuples, `--unit-graph` unit lookup by crate name.
- `scripts/target_analysis_discovery.py` — turns raw `rustc --print target-list` output (plus an optional single requested target) into the target matrix consumed by downstream jobs' `fromJSON()`.
- `scripts/target_analysis_indexing.py` — structural extraction (error counts by target name), the `unexpected-clean-std-build` contradiction rule, and artifact-completeness checking.
- `scripts/target_analysis_promotion.py` — the measurable-difference definition: per-stage promotion (declared post-condition FALSE→TRUE transition) and per-round depth-advancement (error-site set difference).
- `tests/test_target_analysis_common.py`, `tests/test_target_analysis_discovery.py`, `tests/test_target_analysis_indexing.py`, `tests/test_target_analysis_promotion.py` — pytest unit tests.
- `tests/fixtures/target_analysis/` — synthetic JSON fixtures: `unit_graph_serde_default.json`, `unit_graph_serde_patched.json`, `cargo_metadata_full_workspace.json`, `cargo_metadata_trimmed_workspace.json`, `compiler_messages_baseline.json`, `compiler_messages_sibling_progress.json`, `target_spec_nvptx.json`, `target_list_sample.txt`.

---

### Task 1: Shared parsing helpers

**Files:**
- Create: `scripts/target_analysis_common.py`
- Test: `tests/test_target_analysis_common.py`
- Test fixtures: `tests/fixtures/target_analysis/unit_graph_serde_default.json`, `tests/fixtures/target_analysis/compiler_messages_baseline.json`

**Interfaces:**
- Produces: `load_json(path: Path) -> Any`; `unit_graph_units_named(unit_graph: dict, name: str) -> list[dict]`; `error_sites(compiler_messages: list[dict]) -> set[tuple[str, int, str]]`. Every later task (indexing, promotion) imports these three names from `scripts/target_analysis_common.py` — do not re-implement JSON parsing elsewhere.

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/target_analysis/unit_graph_serde_default.json`:
```json
{
  "version": 1,
  "units": [
    {
      "pkg_id": "serde 1.0.210",
      "target": {"kind": ["lib"], "name": "serde"},
      "features": ["default", "std", "derive"]
    },
    {
      "pkg_id": "larql-cli 0.1.0",
      "target": {"kind": ["bin"], "name": "larql-cli"},
      "features": []
    }
  ],
  "roots": [1]
}
```

`tests/fixtures/target_analysis/compiler_messages_baseline.json` (one JSON object per line, matching real `cargo build --message-format=json` streaming output):
```json
[
  {"reason": "compiler-message", "target": {"name": "larql-boundary"}, "message": {"level": "error", "code": {"code": "E0433"}, "message": "failed to resolve: use of undeclared crate or module `std`", "spans": [{"file_name": "crates/larql-boundary/src/lib.rs", "line_start": 12, "is_primary": true}]}},
  {"reason": "compiler-message", "target": {"name": "larql-boundary"}, "message": {"level": "error", "code": {"code": "E0433"}, "message": "failed to resolve: use of undeclared crate or module `std`", "spans": [{"file_name": "crates/larql-boundary/src/lib.rs", "line_start": 47, "is_primary": true}]}},
  {"reason": "compiler-message", "target": {"name": "larql-boundary"}, "message": {"level": "warning", "code": null, "message": "unused import", "spans": [{"file_name": "crates/larql-boundary/src/lib.rs", "line_start": 3, "is_primary": true}]}}
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_target_analysis_common.py
import json
from pathlib import Path

from scripts.target_analysis_common import error_sites, load_json, unit_graph_units_named

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_load_json_reads_a_real_file():
    data = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert data["version"] == 1


def test_unit_graph_units_named_finds_serde_only():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    units = unit_graph_units_named(unit_graph, "serde")
    assert len(units) == 1
    assert units[0]["features"] == ["default", "std", "derive"]


def test_unit_graph_units_named_returns_empty_for_absent_crate():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert unit_graph_units_named(unit_graph, "not-a-real-crate") == []


def test_error_sites_extracts_only_error_level_primary_spans():
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    sites = error_sites(messages)
    assert sites == {
        ("crates/larql-boundary/src/lib.rs", 12, "E0433"),
        ("crates/larql-boundary/src/lib.rs", 47, "E0433"),
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.target_analysis_common'` (or `ImportError`) — the module doesn't exist yet.

- [ ] **Step 4: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Shared, dependency-free JSON parsing for the target-analysis pipeline's
indexing and promotion scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unit_graph_units_named(unit_graph: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        unit
        for unit in unit_graph.get("units", [])
        if unit.get("target", {}).get("name") == name
    ]


def error_sites(compiler_messages: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    sites: set[tuple[str, int, str]] = set()
    for entry in compiler_messages:
        if entry.get("reason") != "compiler-message":
            continue
        message = entry.get("message", {})
        if message.get("level") != "error":
            continue
        code = (message.get("code") or {}).get("code") or message.get("message", "")[:60]
        for span in message.get("spans", []):
            if span.get("is_primary"):
                sites.add((span.get("file_name", ""), span.get("line_start", -1), code))
    return sites
```

Also create an empty `scripts/__init__.py` and `tests/__init__.py` if pytest's rootdir import doesn't already resolve `scripts.*` — check first by running Step 3's command from the repo root; only add `__init__.py` files if the `ModuleNotFoundError` names the package itself as missing, not just the module.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_common.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/target_analysis_common.py tests/test_target_analysis_common.py tests/fixtures/target_analysis/unit_graph_serde_default.json tests/fixtures/target_analysis/compiler_messages_baseline.json
git commit -m "feat: add shared JSON parsing helpers for target-analysis pipeline scripts"
```

---

### Task 2: Discovery script — target matrix resolution

**Files:**
- Create: `scripts/target_analysis_discovery.py`
- Test: `tests/test_target_analysis_discovery.py`
- Test fixture: `tests/fixtures/target_analysis/target_list_sample.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `parse_target_list(raw: str) -> list[str]`; `resolve_target_matrix(all_targets: list[str], requested: str | None) -> list[str]` (raises `ValueError` if `requested` is not in `all_targets` — this is the sanity check Standing Principle 8 requires before any downstream job trusts a `workflow_dispatch` input). Task 9 (Discovery job wiring) invokes this script's CLI to emit the matrix JSON that `fromJSON()` consumes.

- [ ] **Step 1: Write the fixture**

`tests/fixtures/target_analysis/target_list_sample.txt` (a small, realistic excerpt — real `rustc --print target-list` output, one triple per line):
```
aarch64-apple-darwin
nvptx64-nvidia-cuda
wasm32v1-none
x86_64-unknown-linux-gnu
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_target_analysis_discovery.py
import pytest

from scripts.target_analysis_discovery import parse_target_list, resolve_target_matrix

RAW = "aarch64-apple-darwin\nnvptx64-nvidia-cuda\nwasm32v1-none\nx86_64-unknown-linux-gnu\n"


def test_parse_target_list_splits_and_strips_blank_lines():
    raw_with_trailing_blank = RAW + "\n"
    assert parse_target_list(raw_with_trailing_blank) == [
        "aarch64-apple-darwin",
        "nvptx64-nvidia-cuda",
        "wasm32v1-none",
        "x86_64-unknown-linux-gnu",
    ]


def test_resolve_target_matrix_with_no_request_returns_everything():
    targets = parse_target_list(RAW)
    assert resolve_target_matrix(targets, None) == targets


def test_resolve_target_matrix_with_valid_request_returns_singleton():
    targets = parse_target_list(RAW)
    assert resolve_target_matrix(targets, "nvptx64-nvidia-cuda") == ["nvptx64-nvidia-cuda"]


def test_resolve_target_matrix_with_invalid_request_raises():
    targets = parse_target_list(RAW)
    with pytest.raises(ValueError, match="not-a-real-target"):
        resolve_target_matrix(targets, "not-a-real-target")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.target_analysis_discovery'`

- [ ] **Step 4: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Resolves the target matrix the discovery job hands downstream jobs via
fromJSON(). A requested target that isn't real rustc target-list output is
a loud failure here, never a silent narrowing (Standing Principle 8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_target_list(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def resolve_target_matrix(all_targets: list[str], requested: str | None) -> list[str]:
    if requested is None:
        return all_targets
    if requested not in all_targets:
        raise ValueError(
            f"requested target '{requested}' is not in rustc --print target-list output"
        )
    return [requested]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    matrix = resolve_target_matrix(all_targets, args.requested_target)
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/target_analysis_discovery.py tests/test_target_analysis_discovery.py tests/fixtures/target_analysis/target_list_sample.txt
git commit -m "feat: add target-matrix resolution script for the discovery job"
```

---

### Task 3: Indexing script — error counts, contradiction rule, completeness check

**Files:**
- Create: `scripts/target_analysis_indexing.py`
- Test: `tests/test_target_analysis_indexing.py`
- Test fixture: `tests/fixtures/target_analysis/target_spec_nvptx.json`

**Interfaces:**
- Consumes: `load_json`, `error_sites` from `scripts/target_analysis_common.py` (Task 1).
- Produces: `count_errors_by_target(compiler_messages: list[dict]) -> dict[str, int]`; `unexpected_clean_std_build(target_spec: dict, std_mode_errors: list) -> bool`; `missing_artifacts(expected: set[str], actual: set[str]) -> set[str]`. Task 10 (Indexing job wiring) calls this script's CLI to build the navigation index and fails the job if `missing_artifacts(...)` is non-empty.

- [ ] **Step 1: Write the fixture**

`tests/fixtures/target_analysis/target_spec_nvptx.json` (trimmed real fields from `rustc --print target-spec-json --target nvptx64-nvidia-cuda -Z unstable-options`):
```json
{
  "arch": "nvptx64",
  "os": "cuda",
  "std": false,
  "only-cdylib": true,
  "panic-strategy": "abort"
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_target_analysis_indexing.py
from pathlib import Path

from scripts.target_analysis_common import load_json
from scripts.target_analysis_indexing import (
    count_errors_by_target,
    missing_artifacts,
    unexpected_clean_std_build,
)

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_count_errors_by_target_counts_only_error_level():
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    assert count_errors_by_target(messages) == {"larql-boundary": 2}


def test_unexpected_clean_std_build_flags_the_nvptx_canary_contradiction():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert unexpected_clean_std_build(target_spec, std_mode_errors=[]) is True


def test_unexpected_clean_std_build_does_not_flag_a_real_failure():
    target_spec = load_json(FIXTURES / "target_spec_nvptx.json")
    assert unexpected_clean_std_build(target_spec, std_mode_errors=["some error"]) is False


def test_unexpected_clean_std_build_does_not_flag_targets_with_std():
    host_spec = {"std": True}
    assert unexpected_clean_std_build(host_spec, std_mode_errors=[]) is False


def test_missing_artifacts_returns_the_set_difference():
    expected = {"probe-a", "probe-b", "probe-c"}
    actual = {"probe-a", "probe-c"}
    assert missing_artifacts(expected, actual) == {"probe-b"}


def test_missing_artifacts_empty_when_nothing_missing():
    expected = {"probe-a"}
    assert missing_artifacts(expected, expected) == set()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_indexing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.target_analysis_indexing'`

- [ ] **Step 4: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Structural extraction for the indexing job: error counts by cargo target
name, the unexpected-clean-std-build contradiction rule (Standing Principle
5 — the nvptx canary), and artifact-completeness checking (Standing
Principle 9 — the indexing job fails loudly on any missing expected
artifact rather than silently indexing whatever showed up)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.target_analysis_common import error_sites


def count_errors_by_target(compiler_messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in compiler_messages:
        if entry.get("reason") != "compiler-message":
            continue
        if entry.get("message", {}).get("level") != "error":
            continue
        name = entry.get("target", {}).get("name", "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def unexpected_clean_std_build(target_spec: dict[str, Any], std_mode_errors: list[Any]) -> bool:
    return target_spec.get("std") is False and len(std_mode_errors) == 0


def missing_artifacts(expected: set[str], actual: set[str]) -> set[str]:
    return expected - actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-messages-file", required=True, type=Path)
    parser.add_argument("--target-spec-file", required=True, type=Path)
    parser.add_argument("--std-mode-errors-file", required=True, type=Path)
    parser.add_argument("--expected-artifacts-file", required=True, type=Path)
    parser.add_argument("--actual-artifacts-file", required=True, type=Path)
    args = parser.parse_args()

    compiler_messages = json.loads(args.compiler_messages_file.read_text(encoding="utf-8"))
    target_spec = json.loads(args.target_spec_file.read_text(encoding="utf-8"))
    std_mode_errors = json.loads(args.std_mode_errors_file.read_text(encoding="utf-8"))
    expected = set(json.loads(args.expected_artifacts_file.read_text(encoding="utf-8")))
    actual = set(json.loads(args.actual_artifacts_file.read_text(encoding="utf-8")))

    missing = missing_artifacts(expected, actual)
    result = {
        "error_counts_by_target": count_errors_by_target(compiler_messages),
        "contradictions": {
            "unexpected_clean_std_build": unexpected_clean_std_build(target_spec, std_mode_errors),
        },
        "missing_artifacts": sorted(missing),
    }
    print(json.dumps(result, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_indexing.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/target_analysis_indexing.py tests/test_target_analysis_indexing.py tests/fixtures/target_analysis/target_spec_nvptx.json
git commit -m "feat: add indexing script with contradiction rule and completeness check"
```

---

### Task 4: Promotion script — the measurable-difference definition

This is the mechanical, coded criterion decided in this session: **stage-level promotion** (did a sibling mutation's own declared, named post-condition become true against the correct L1 source, where it was false in the baseline) is a distinct check from **round-level depth advancement** (did we get past a concrete blocker, regardless of new blockers revealed). Neither is defined in terms of Stage C's aggregate error *count* — a naive count-based check would punish a mutation for successfully revealing a previously-hidden layer of problems, which is the Secondary layer's actual purpose (spec, Standing Principle 6).

**Files:**
- Create: `scripts/target_analysis_promotion.py`
- Test: `tests/test_target_analysis_promotion.py`
- Test fixtures: `tests/fixtures/target_analysis/unit_graph_serde_patched.json`, `tests/fixtures/target_analysis/cargo_metadata_full_workspace.json`, `tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json`, `tests/fixtures/target_analysis/compiler_messages_sibling_progress.json`

**Interfaces:**
- Consumes: `unit_graph_units_named`, `error_sites` from `scripts/target_analysis_common.py` (Task 1).
- Produces: `serde_features_ok(unit_graph: dict) -> bool`; `workspace_members_ok(metadata: dict, expected_members: list[str]) -> bool`; `no_std_scaffold_ok(lib_rs_content: str) -> bool`; `stage_promotes(stage_name: str, baseline_state: dict, sibling_state: dict) -> bool`; `depth_advanced(baseline_sites: set[tuple[str, int, str]], sibling_sites: set[tuple[str, int, str]]) -> bool`. Task 17 (Secondary-layer promotion wiring) calls `stage_promotes` once per stage (`stage-b`, `stage-b2`, `stage-b3`) and `depth_advanced` once per round to decide what folds into the next round's shared baseline.

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/target_analysis/unit_graph_serde_patched.json` (same shape as `unit_graph_serde_default.json` from Task 1, but with the intended Stage B2 result — `serde`'s features narrowed to exactly `alloc` + `derive`, no `default`, no `std`):
```json
{
  "version": 1,
  "units": [
    {
      "pkg_id": "serde 1.0.210",
      "target": {"kind": ["lib"], "name": "serde"},
      "features": ["alloc", "derive"]
    },
    {
      "pkg_id": "larql-cli 0.1.0",
      "target": {"kind": ["bin"], "name": "larql-cli"},
      "features": []
    }
  ],
  "roots": [1]
}
```

`tests/fixtures/target_analysis/cargo_metadata_full_workspace.json` (trimmed real shape of `cargo metadata`'s `workspace_members`, package-id-style entries — the baseline, unmutated 19-crate workspace):
```json
{
  "workspace_members": [
    "larql-cli 0.1.0 (path+file:///repo/crates/larql-cli)",
    "larql-boundary 0.1.0 (path+file:///repo/crates/larql-boundary)",
    "larql-python 0.1.0 (path+file:///repo/crates/larql-python)",
    "larql-vindex-spec 0.1.0 (path+file:///repo/crates/larql-vindex-spec)"
  ]
}
```

`tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json` (the intended Stage B3 result — `larql-python` removed, since it's out of `larql-cli`'s real reachable tree, confirmed earlier this session):
```json
{
  "workspace_members": [
    "larql-cli 0.1.0 (path+file:///repo/crates/larql-cli)",
    "larql-boundary 0.1.0 (path+file:///repo/crates/larql-boundary)",
    "larql-vindex-spec 0.1.0 (path+file:///repo/crates/larql-vindex-spec)"
  ]
}
```

`tests/fixtures/target_analysis/compiler_messages_sibling_progress.json` (a sibling round where the baseline's line-12 `E0433` is resolved — Stage B2 genuinely fixed something — but a new, previously-hidden error at line 60 is now visible; this is the case a naive error-count check gets wrong):
```json
[
  {"reason": "compiler-message", "target": {"name": "larql-boundary"}, "message": {"level": "error", "code": {"code": "E0433"}, "message": "failed to resolve: use of undeclared crate or module `std`", "spans": [{"file_name": "crates/larql-boundary/src/lib.rs", "line_start": 47, "is_primary": true}]}},
  {"reason": "compiler-message", "target": {"name": "larql-boundary"}, "message": {"level": "error", "code": {"code": "E0658"}, "message": "use of unstable library feature", "spans": [{"file_name": "crates/larql-boundary/src/lib.rs", "line_start": 60, "is_primary": true}]}}
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_target_analysis_promotion.py
from pathlib import Path

from scripts.target_analysis_common import error_sites, load_json
from scripts.target_analysis_promotion import (
    depth_advanced,
    no_std_scaffold_ok,
    serde_features_ok,
    stage_promotes,
    workspace_members_ok,
)

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_serde_features_ok_is_false_for_default_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert serde_features_ok(unit_graph) is False


def test_serde_features_ok_is_true_for_patched_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_patched.json")
    assert serde_features_ok(unit_graph) is True


def test_serde_features_ok_is_false_when_serde_is_absent():
    assert serde_features_ok({"units": []}) is False


def test_workspace_members_ok_false_for_full_workspace():
    metadata = load_json(FIXTURES / "cargo_metadata_full_workspace.json")
    expected = ["larql-cli", "larql-boundary", "larql-vindex-spec"]
    assert workspace_members_ok(metadata, expected) is False


def test_workspace_members_ok_true_for_trimmed_workspace():
    metadata = load_json(FIXTURES / "cargo_metadata_trimmed_workspace.json")
    expected = ["larql-cli", "larql-boundary", "larql-vindex-spec"]
    assert workspace_members_ok(metadata, expected) is True


def test_no_std_scaffold_ok_requires_both_markers():
    assert no_std_scaffold_ok("//! docs\n#![no_std]\nextern crate alloc;\n") is True
    assert no_std_scaffold_ok("//! docs\n#![no_std]\n") is False
    assert no_std_scaffold_ok("//! docs\nextern crate alloc;\n") is False


def test_stage_promotes_serde_patch_transitions_false_to_true():
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is True


def test_stage_promotes_serde_patch_false_when_sibling_also_unpatched():
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_default.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is False


def test_stage_promotes_false_when_baseline_already_satisfies_postcondition():
    # A sibling can never promote by matching an already-true baseline —
    # promotion requires a genuine false-to-true transition, not just "true".
    baseline = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    sibling = {"unit_graph": load_json(FIXTURES / "unit_graph_serde_patched.json")}
    assert stage_promotes("stage-b2", baseline, sibling) is False


def test_depth_advanced_true_when_a_baseline_site_is_resolved():
    baseline = error_sites(load_json(FIXTURES / "compiler_messages_baseline.json"))
    sibling = error_sites(load_json(FIXTURES / "compiler_messages_sibling_progress.json"))
    # baseline has line 12 and 47; sibling has 47 and a new line 60.
    # Line 12 disappearing is real progress, even though a new site (60) appeared.
    assert depth_advanced(baseline, sibling) is True


def test_depth_advanced_false_when_nothing_baseline_present_is_resolved():
    baseline = error_sites(load_json(FIXTURES / "compiler_messages_baseline.json"))
    # sibling == baseline: identical wall, no resolution, no advancement.
    assert depth_advanced(baseline, baseline) is False


def test_depth_advanced_false_when_sibling_only_adds_new_sites():
    baseline = {("crates/larql-boundary/src/lib.rs", 12, "E0433")}
    sibling = baseline | {("crates/larql-boundary/src/lib.rs", 99, "E0999")}
    assert depth_advanced(baseline, sibling) is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_promotion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.target_analysis_promotion'`

- [ ] **Step 4: Write minimal implementation**

```python
#!/usr/bin/env python3
"""The measurable-difference definition for the Secondary layer's
recursive observe/mutate loop.

Two distinct, separately-checked criteria — deliberately not one aggregate
metric:

1. Stage-level promotion: does a sibling mutation's own declared,
   named post-condition become true against the correct L1 source for
   that specific claim, where it was false in the baseline? A sibling
   promotes into the next round's shared upstream iff its postcondition
   is False on the baseline and True on the sibling.

2. Round-level depth advancement: did we get past a concrete blocker?
   Defined on the SET of (file, line, error-code) error sites, not a
   count and not a set of message categories. Depth advances iff at
   least one site present in the baseline is absent in the sibling —
   whether the sibling also reveals brand-new sites is irrelevant to
   this check and never counts against it. A naive "lower error count"
   check would punish exactly the case this layer exists to produce:
   clearing a blocker reveals a previously-hidden layer of problems,
   which can raise the count while still being genuine progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from scripts.target_analysis_common import unit_graph_units_named


def serde_features_ok(unit_graph: dict[str, Any]) -> bool:
    units = unit_graph_units_named(unit_graph, "serde")
    if not units:
        return False
    expected = {"alloc", "derive"}
    return all(set(unit.get("features", [])) == expected for unit in units)


def workspace_members_ok(metadata: dict[str, Any], expected_members: list[str]) -> bool:
    actual_names = {
        member.split(" ", 1)[0] for member in metadata.get("workspace_members", [])
    }
    return actual_names == set(expected_members)


def no_std_scaffold_ok(lib_rs_content: str) -> bool:
    return "#![no_std]" in lib_rs_content and "extern crate alloc;" in lib_rs_content


STAGE_POSTCONDITIONS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "stage-b": lambda state: no_std_scaffold_ok(state["lib_rs_content"]),
    "stage-b2": lambda state: serde_features_ok(state["unit_graph"]),
    "stage-b3": lambda state: workspace_members_ok(
        state["metadata"], state["expected_members"]
    ),
}


def stage_promotes(
    stage_name: str, baseline_state: dict[str, Any], sibling_state: dict[str, Any]
) -> bool:
    postcondition = STAGE_POSTCONDITIONS[stage_name]
    return postcondition(sibling_state) and not postcondition(baseline_state)


def depth_advanced(
    baseline_sites: set[tuple[str, int, str]], sibling_sites: set[tuple[str, int, str]]
) -> bool:
    return len(baseline_sites - sibling_sites) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_POSTCONDITIONS))
    parser.add_argument("--baseline-state-file", required=True, type=Path)
    parser.add_argument("--sibling-state-file", required=True, type=Path)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    baseline_state = json.loads(args.baseline_state_file.read_text(encoding="utf-8"))
    sibling_state = json.loads(args.sibling_state_file.read_text(encoding="utf-8"))
    promotes = stage_promotes(args.stage, baseline_state, sibling_state)

    print(json.dumps({"stage": args.stage, "promotes": promotes}))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"promotes={'true' if promotes else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_promotion.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py \
  tests/fixtures/target_analysis/unit_graph_serde_patched.json \
  tests/fixtures/target_analysis/cargo_metadata_full_workspace.json \
  tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json \
  tests/fixtures/target_analysis/compiler_messages_sibling_progress.json
git commit -m "feat: add stage-promotion and depth-advancement scripts (measurable-difference rule)"
```

---

### Task 5: Discovery job in the workflow

**Files:**
- Create: `.github/workflows/target-analysis-pipeline.yml` (new file — this task creates it with the `discovery` job only; later tasks add jobs to the same file)

**Interfaces:**
- Consumes: `scripts/target_analysis_discovery.py`'s CLI (Task 2): `python3 scripts/target_analysis_discovery.py --target-list-file <path> --requested-target <target-or-empty>`.
- Produces: job output `discovery.outputs.target-matrix` (a JSON array string) — every later job with a target-dependent matrix declares `needs: [discovery]` and reads `fromJSON(needs.discovery.outputs.target-matrix)`.

- [ ] **Step 1: Write the workflow header and discovery job**

```yaml
name: target-analysis-pipeline

on:
  push:
    branches:
      - "experiment/target-analysis-*"
  workflow_dispatch:
    inputs:
      target:
        description: "Single target triple to analyze (optional; omit for the full rustc target-list)"
        required: false
        type: string

jobs:
  discovery:
    name: Discover target matrix
    runs-on: ubuntu-latest
    outputs:
      target-matrix: ${{ steps.resolve.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: rustc --print target-list
        run: rustup run nightly rustc --print target-list > target-list.txt
      - name: Resolve target matrix
        id: resolve
        run: |
          MATRIX=$(python3 scripts/target_analysis_discovery.py \
            --target-list-file target-list.txt \
            --requested-target "${{ inputs.target }}")
          echo "matrix=$MATRIX" >> "$GITHUB_OUTPUT"
      - name: Upload raw target-list
        uses: actions/upload-artifact@v4
        with:
          name: discovery-target-list
          path: target-list.txt
```

Note: `--requested-target ""` (empty string, the default when `inputs.target` is unset) must be handled — `scripts/target_analysis_discovery.py`'s `argparse` default is `None`, but an empty string arrives here instead. Fix in this step, not a later one: change the script's CLI handling so an empty string is treated identically to `None`.

- [ ] **Step 2: Fix the empty-string CLI case with a test**

Add to `tests/test_target_analysis_discovery.py`:
```python
def test_main_treats_empty_string_requested_target_as_none(tmp_path, capsys):
    from scripts.target_analysis_discovery import main
    import sys

    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text(RAW, encoding="utf-8")
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
    ]
    assert main() == 0
    out = capsys.readouterr().out
    assert json.loads(out) == parse_target_list(RAW)
```

Add `import json` to the top of the test file.

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: FAIL — empty string is not `None`, so `resolve_target_matrix` treats `""` as a requested target and raises `ValueError`.

- [ ] **Step 3: Fix the implementation**

In `scripts/target_analysis_discovery.py`'s `main()`, change:
```python
    all_targets = parse_target_list(raw)
    matrix = resolve_target_matrix(all_targets, args.requested_target)
```
to:
```python
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml scripts/target_analysis_discovery.py tests/test_target_analysis_discovery.py
git commit -m "feat: add discovery job wiring target matrix into the new pipeline"
```

- [ ] **Step 6: Push and verify on a real runner**

Push to a branch matching `experiment/target-analysis-*`. Fetch the run's `discovery` job logs and confirm: the job succeeds, `target-list.txt` is uploaded as an artifact, and `steps.resolve.outputs.matrix` is a non-empty JSON array (check via the job's step output in the Actions UI or `gh run view --log`). This is the Task's actual test — per the spec's Validation approach, workflow-YAML behavior is proven only by a real run, never local simulation.

---

### Task 6: Batch the target matrix for GitHub Actions' 256-job cap

Task 5's real CI run empirically discovered 331 targets from `rustc --print
target-list` — independently reconfirmed by that task's reviewer via a downloaded
artifact, not assumed. GitHub Actions hard-caps `strategy.matrix` at 256 jobs, and
separately caps any single job's wall-clock at 6 hours. A `strategy.matrix: target:
fromJSON(needs.discovery.outputs.target-matrix)` fanning directly over the real,
unbatched 331-target list — as Tasks 7-9 and 18 originally would have — fails to
schedule. Worse, Task 8's build-attempt job crosses each target against 4
`build_std` modes × 3 `cargo_cmd`s × 2 feature configs × (typically ~5) crate types
— roughly 120 `cargo` invocations per target — so even a 256-target batch for that
specific job risks exceeding the 6-hour per-job limit. This task adds the batching
mechanism every later matrix-over-targets job needs, sized conservatively (12
targets per batch, uniformly across every batch-consuming job) so the heaviest job
(build-attempt) stays safely under both caps; Task 8's own real-run verification
step re-checks this estimate against actual job duration, since it's a reasoned
estimate, not a verified fact, until a real run confirms it.

This does not change `target-matrix` itself — Task 10's indexing job and Task 20's
next-round-baseline job keep consuming the unbatched form directly inside a Python
loop, which has no 256-element restriction.

**Files:**
- Modify: `scripts/target_analysis_discovery.py` (add `chunk_targets`, extend `main()`)
- Modify: `tests/test_target_analysis_discovery.py` (add tests for `chunk_targets` and the extended CLI)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (extend the `discovery` job's outputs and "Resolve target matrix" step)

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `chunk_targets(targets: list[str], max_size: int = 256) -> list[list[str]]`; job outputs `needs.discovery.outputs.batches` (JSON array of arrays, each ≤12 entries once the workflow step passes `--max-batch-size 12`) and `needs.discovery.outputs.batch-indices` (JSON array of ints `[0, 1, ...]`, one per chunk). Tasks 7, 8, 9, and 13 consume both of these instead of fanning `strategy.matrix` directly over `target-matrix`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_target_analysis_discovery.py`, below the existing tests:
```python
from scripts.target_analysis_discovery import chunk_targets


def test_chunk_targets_empty_list_returns_empty():
    assert chunk_targets([], max_size=3) == []


def test_chunk_targets_smaller_than_max_size_is_one_chunk():
    assert chunk_targets(["a", "b"], max_size=3) == [["a", "b"]]


def test_chunk_targets_exactly_max_size_is_one_chunk():
    assert chunk_targets(["a", "b", "c"], max_size=3) == [["a", "b", "c"]]


def test_chunk_targets_splits_into_multiple_chunks_with_remainder():
    targets = ["a", "b", "c", "d", "e", "f", "g"]
    assert chunk_targets(targets, max_size=3) == [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g"],
    ]


def test_chunk_targets_default_max_size_is_256():
    targets = [f"t{i}" for i in range(300)]
    chunks = chunk_targets(targets)
    assert len(chunks) == 2
    assert len(chunks[0]) == 256
    assert len(chunks[1]) == 44


def test_chunk_targets_rejects_non_positive_max_size():
    import pytest
    with pytest.raises(ValueError, match="max_size must be positive"):
        chunk_targets(["a"], max_size=0)


def test_main_writes_batches_and_batch_indices_to_github_output(tmp_path):
    from scripts.target_analysis_discovery import main
    import sys

    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("\n".join(f"t{i}" for i in range(300)), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    batches_line = next(line for line in output_text.splitlines() if line.startswith("batches="))
    batch_indices_line = next(line for line in output_text.splitlines() if line.startswith("batch-indices="))
    batches = json.loads(batches_line[len("batches="):])
    batch_indices = json.loads(batch_indices_line[len("batch-indices="):])
    assert len(batches) == 2
    assert batch_indices == [0, 1]
```
`import json` is already at the top of this test file (added in Task 5's Step 2) — do not add it twice.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_targets'` for the first 6 new tests; the CLI test also fails (it exercises `main()`, which doesn't yet accept `--github-output` or emit `batches`).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/target_analysis_discovery.py`, above `main()`:
```python
def chunk_targets(targets: list[str], max_size: int = 256) -> list[list[str]]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    return [targets[i : i + max_size] for i in range(0, len(targets), max_size)]
```

Replace `main()` (currently, from Task 5):
```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)
    print(json.dumps(matrix))
    return 0
```
with:
```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)
    batches = chunk_targets(matrix, max_size=args.max_batch_size)
    batch_indices = list(range(len(batches)))

    print(json.dumps(matrix))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"batches={json.dumps(batches)}\n")
            handle.write(f"batch-indices={json.dumps(batch_indices)}\n")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: 12 passed (5 existing from Tasks 2 and 5, plus 6 `chunk_targets` tests, plus 1 CLI test)

- [ ] **Step 5: Commit**

```bash
git add scripts/target_analysis_discovery.py tests/test_target_analysis_discovery.py
git commit -m "feat: add target-matrix batching for GitHub Actions' 256-job matrix cap"
```

- [ ] **Step 6: Extend the Discovery job's workflow step and declare the new outputs**

Modify `.github/workflows/target-analysis-pipeline.yml`'s `discovery` job to exactly:
```yaml
  discovery:
    name: Discover target matrix
    runs-on: ubuntu-latest
    outputs:
      target-matrix: ${{ steps.resolve.outputs.matrix }}
      batches: ${{ steps.resolve.outputs.batches }}
      batch-indices: ${{ steps.resolve.outputs.batch-indices }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: rustc --print target-list
        run: rustup run nightly rustc --print target-list > target-list.txt
      - name: Resolve target matrix
        id: resolve
        env:
          REQUESTED_TARGET: ${{ inputs.target }}
        run: |
          python3 scripts/target_analysis_discovery.py \
            --target-list-file target-list.txt \
            --requested-target "$REQUESTED_TARGET" \
            --max-batch-size 12 \
            --github-output "$GITHUB_OUTPUT" > matrix-stdout.json
          echo "matrix=$(cat matrix-stdout.json)" >> "$GITHUB_OUTPUT"
      - name: Upload raw target-list
        uses: actions/upload-artifact@v4
        with:
          name: discovery-target-list
          path: target-list.txt
```
What changed from Task 5's merged state: the script's `--github-output "$GITHUB_OUTPUT"` flag now writes `batches=`/`batch-indices=` directly (append-mode, per Step 3's implementation); the `matrix=` value is now captured to a file first (`> matrix-stdout.json`) rather than via `$(...)` command substitution, since the script writes to `$GITHUB_OUTPUT` as a side effect and stdout is now used only for the `matrix` value — then that file's content is written to `$GITHUB_OUTPUT` exactly as before. `--max-batch-size 12` is deliberately smaller than the 256-job cap itself, sized for Task 8's build-attempt job specifically (see this task's opening rationale) and applied uniformly to every batch-consuming job for simplicity, even though target-capability/dependency-graph/runtime-test could each tolerate much larger batches on their own.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: wire batches/batch-indices outputs into the discovery job"
```

- [ ] **Step 8: Push and verify on a real runner**

Push a follow-up commit to the same branch Task 5 pushed to (`experiment/target-analysis-pipeline`) — not a new branch. Fetch the real run's `discovery` job log and confirm: `steps.resolve.outputs.batches` is a JSON array of arrays, each of length ≤12 (given the real 331-target count from Task 5's run, expect `ceil(331/12) = 28` batches, the last one shorter — reconfirm the exact target count on this run too, since it could have changed by even one entry since Task 5 ran), and `steps.resolve.outputs.batch-indices` is `[0, 1, ..., 27]` (or whatever the real count implies). Confirm `target-matrix`'s value is unchanged in shape from Task 5's run (still the full unbatched array) — Tasks 10 and 19 still need it that way.

---

### Task 7: Target-capability probe job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add the `target-capability` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.batches`, `needs.discovery.outputs.batch-indices` (Task 6).
- Produces: per-batch artifact `target-capability-batch-<batch_index>`, containing one `target-spec-<target>.json` / `cfg-<target>.txt` / `supported-crate-types-<target>.txt` triple per target in that batch — consumed by Task 8 (build-attempt job) and Task 10 (indexing). Batching the artifact (not the target) is the direct consequence of Task 6's ruling: `actions/upload-artifact` is a `uses:` step and cannot itself be invoked in a per-target shell loop, so per-target granularity moves to the filename inside one per-batch artifact instead of to the artifact name itself — every file is still individually named and addressable, just grouped by batch at the artifact level.

- [ ] **Step 1: Add the job**

```yaml
  target-capability:
    name: "Target capability: batch ${{ matrix.batch_index }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        batch_index: ${{ fromJSON(needs.discovery.outputs.batch-indices) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: Probe every target in this batch
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p out
          echo "$BATCH_TARGETS" | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)))' > targets-in-batch.txt
          FAILED=0
          while IFS= read -r TARGET; do
            echo "=== target: $TARGET ==="
            if ! rustup run nightly rustc --print target-list | grep -qx "$TARGET"; then
              echo "::error::$TARGET not found in rustc --print target-list — discovery/capability disagreement"
              FAILED=1
              continue
            fi
            rustup run nightly rustc --print target-spec-json -Z unstable-options \
              --target "$TARGET" > "out/target-spec-$TARGET.json" || FAILED=1
            rustup run nightly rustc --print cfg --target "$TARGET" > "out/cfg-$TARGET.txt" || FAILED=1
            rustup run nightly rustc --print supported-crate-types --target "$TARGET" \
              > "out/supported-crate-types-$TARGET.txt" || FAILED=1
          done < targets-in-batch.txt
          exit $FAILED
      - name: Upload capability artifacts for this batch
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: target-capability-batch-${{ matrix.batch_index }}
          path: out/
```

Filenames embed the target triple directly (e.g. `out/target-spec-nvptx64-nvidia-cuda.json`); rustc target names are restricted to `[a-zA-Z0-9_.-]`, so no escaping is needed. `if: always()` on the upload step means a batch with one bad target still uploads the rest of that batch's good data (Standing Principle 7 — over-collection is correctable, under-collection is not; one target's probe failure must never silently drop an entire batch).

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add target-capability probe job, batched over the 256-job matrix cap"
```

- [ ] **Step 3: Push and verify on a real runner**

Push (same branch, follow-up commit). Confirm: `ceil(331/12) = 28` `target-capability` job instances (batch 0 through 27, matching Task 6's real split), each uploading one artifact containing one file-triple per target in its batch. Identify which batch contains `nvptx64-nvidia-cuda` (its position in the real, ordered 331-target list from Task 5/6's run determines this — download whichever batch artifact contains it) and confirm `target-spec-nvptx64-nvidia-cuda.json` shows `"std": false` and `supported-crate-types-nvptx64-nvidia-cuda.txt` shows `bin, cdylib, lib, rlib, staticlib` (matching this session's earlier empirical finding, and Standing Principle 5's canary: a clean/all-succeeding result here for nvptx would itself be a bug to chase).

---

### Task 8: Dependency-graph and build-attempt probe jobs

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `dependency-graph` and `build-attempt` jobs)

**Interfaces:**
- Consumes: `needs.discovery.outputs.batches`, `needs.discovery.outputs.batch-indices` (Task 6); `target-capability-batch-<batch_index>` artifacts (Task 7) for the crate-type list — note the batch INDEX lines up between jobs (batch `N`'s build-attempt work downloads batch `N`'s capability artifact, since both jobs consume the identical `batches` array from Discovery).
- Produces: per-batch artifacts `dependency-graph-batch-<batch_index>` and `build-attempt-batch-<batch_index>`, each containing one file (or file-group) per target — consumed by Task 10 (indexing).

Both jobs batch for the same reason as Task 7 (the real, empirically-discovered 331-target count exceeds GitHub Actions' 256-job matrix cap); build-attempt additionally batches at the small size (12 targets/batch, from Task 6) because it crosses each target against 4 `build_std` × 3 `cargo_cmd` × 2 `features` × (typically ~5) crate types — roughly 120 `cargo` invocations per target — which risks exceeding the 6-hour per-job wall-clock limit at a larger batch size.

- [ ] **Step 1: Add the dependency-graph job**

```yaml
  dependency-graph:
    name: "Dependency graph: batch ${{ matrix.batch_index }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        batch_index: ${{ fromJSON(needs.discovery.outputs.batch-indices) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: Dependency-graph probes for every target in this batch
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p out
          echo "$BATCH_TARGETS" | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)))' > targets-in-batch.txt
          cargo install cargo-deny --locked
          FAILED=0
          while IFS= read -r TARGET; do
            echo "=== target: $TARGET ==="
            cargo metadata --format-version 1 --filter-platform "$TARGET" \
              > "out/metadata-$TARGET.json" || FAILED=1
            cargo tree -p larql-cli -e features --target "$TARGET" \
              > "out/tree-features-$TARGET.txt" || FAILED=1
            cargo tree -p larql-cli --duplicates --target "$TARGET" \
              > "out/tree-duplicates-$TARGET.txt" || FAILED=1
            cargo +nightly build -Z unstable-options --unit-graph \
              -p larql-cli --target "$TARGET" \
              > "out/unit-graph-$TARGET.json" || FAILED=1
            cargo deny --config deny-nvptx.toml check bans licenses advisories sources \
              > "out/deny-$TARGET.txt" || true
          done < targets-in-batch.txt
          exit $FAILED
      - name: Upload dependency-graph artifacts for this batch
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: dependency-graph-batch-${{ matrix.batch_index }}
          path: out/
```

`cargo-deny check`'s own `|| true` is unchanged from the original design — its verdict is curated/L2 (Foundational framework) and must never fail the job; the other four commands use `|| FAILED=1` so one target's failure doesn't stop the loop from covering the rest of the batch, but the job itself still reports a real execution failure at the end (`exit $FAILED`) — this job is an observability probe (Error handling: "reflects execution success only"), so a real failure here (as opposed to `cargo-deny`'s curated verdict) is a genuine anomaly, not an expected build outcome.

- [ ] **Step 2: Add the build-attempt job**

```yaml
  build-attempt:
    name: "Build attempt: batch ${{ matrix.batch_index }}"
    needs: [discovery, target-capability]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        batch_index: ${{ fromJSON(needs.discovery.outputs.batch-indices) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add rust-src --toolchain nightly
      - name: Download this batch's target-capability artifact
        uses: actions/download-artifact@v4
        with:
          name: target-capability-batch-${{ matrix.batch_index }}
          path: capability
      - name: Build-attempt probes for every target/build_std/cmd/features/crate-type combination in this batch
        id: attempt
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p out
          echo "$BATCH_TARGETS" | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)))' > targets-in-batch.txt
          FAILED=0
          while IFS= read -r TARGET; do
            CRATE_TYPES=$(tr ',' '\n' < "capability/supported-crate-types-$TARGET.txt" | sed 's/^ *//; s/ *$//')
            for BUILD_STD in none std core,alloc core; do
              if [ "$BUILD_STD" = "none" ]; then BUILD_STD_FLAG=""; else BUILD_STD_FLAG="-Z build-std=$BUILD_STD"; fi
              for CARGO_CMD in check clippy build; do
                for FEATURES in default-features no-default-features; do
                  if [ "$FEATURES" = "no-default-features" ]; then FEATURES_FLAG="--no-default-features"; else FEATURES_FLAG=""; fi
                  OUTFILE="out/attempt-$TARGET-$BUILD_STD-$CARGO_CMD-$FEATURES.json"
                  : > "$OUTFILE"
                  for CRATE_TYPE in $CRATE_TYPES; do
                    echo "=== target=$TARGET build_std=$BUILD_STD cmd=$CARGO_CMD features=$FEATURES crate_type=$CRATE_TYPE ==="
                    cargo +nightly "$CARGO_CMD" -p larql-cli \
                      --target "$TARGET" --crate-type "$CRATE_TYPE" $FEATURES_FLAG $BUILD_STD_FLAG \
                      --keep-going --message-format=json >> "$OUTFILE" || FAILED=1
                  done
                done
              done
            done
          done < targets-in-batch.txt
          echo "batch-had-failure=$FAILED" >> "$GITHUB_OUTPUT"
      - name: Upload full diagnostic JSON for this batch
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: build-attempt-batch-${{ matrix.batch_index }}
          path: out/
      - name: Assert honest result
        run: |
          echo "batch ${{ matrix.batch_index }} had at least one real cargo failure: ${{ steps.attempt.outputs.batch-had-failure }}"
          if [ "${{ steps.attempt.outputs.batch-had-failure }}" = "1" ]; then
            echo "::notice::At least one real build failure in this batch, recorded as a finding in the uploaded JSON, not masked."
          fi
```

This job deliberately runs every `build_std` × `cargo_cmd` × `features` × `crate_type` combination against every target in the batch regardless of what `target-capability` reported about `std` (Standing Principle 3 — exhaustive unconditional fan-out; the spec's Build-attempt probes section is explicit that predictable outcomes are still real L1 data), iterating `supported-crate-types-$TARGET.txt`'s actual content (Task 7's output) rather than a crate-type list chosen in advance, per this session's `only-cdylib` field-name correction.

**Two probe-mechanism fixes folded into this rewrite, both caught before dispatch rather than after a wasted 28-batch run:**
- The crate-type loop's `cargo` invocation now passes `--crate-type "$CRATE_TYPE"` explicitly. Without it, every iteration of the crate-type loop ran the identical command, so the crate-type dimension was silently inert — the loop executed 5 times but never actually varied anything, meaning the spec's "iterate every empirically-supported crate type" requirement went unmet despite looking like it was covered. `cargo build`/`cargo check` (and, transitively, `cargo clippy`) support `--crate-type` directly; confirm on the real run that this composes cleanly with all three `cargo_cmd` values — if `clippy` specifically rejects the flag, that's itself a real, worth-recording finding about the probe mechanism, not something to guess at now.
- `rustup component add rust-src --toolchain nightly` is now installed alongside the toolchain. Every `-Z build-std` mode (`std`, `core,alloc`, `core`) requires the `rust-src` component to resolve `core`/`alloc`/`std` source for recompilation; without it, every build-std cell would fail with a missing-component error that looks identical to a real target-incompatibility finding but is actually a probe misconfiguration — exactly the kind of contradiction the spec's Error handling section requires distinguishing (`platform-limit-hit` and probe-execution-failure are different categories from a genuine compile error). Distinct from this: a tier-3 target genuinely lacking prebuilt `std` in `build_std=none` mode is a real finding, not a probe bug — the indexing categories need to be able to tell these two failure shapes apart, which is only possible once `rust-src` is actually present so a `none`-mode failure can't be misattributed to the missing component instead.

**Deliberate departure from the original per-combo honest-result pattern, noted explicitly so it doesn't read as an oversight:** the original (pre-batch) design used step-level `continue-on-error: true` plus an explicit assert step reading `steps.attempt.outcome`, one pair per (target, build_std, cmd, features) combination. Batching collapses every combination for a batch's targets into one shell loop inside one step, so there is no longer a distinct step per combination to read `.outcome` from. The loop therefore never lets `$FAILED` reach the step's own exit code (there is deliberately no `exit $FAILED` at the end of this step) — the batch job always reports success at the step/job level, and the real per-combination pass/fail signal moves entirely into the uploaded JSON file content (one `attempt-<target>-<build_std>-<cmd>-<features>.json` file per combination, `--message-format=json` preserved exactly as before), which Task 10's indexing job parses directly. This is not a weaker honest-result guarantee — the full compiler diagnostics are strictly more informative than the old per-combo boolean `.outcome` was, and nothing is masked, since the raw JSON is uploaded via `if: always()` regardless of the loop's outcome — but it is a real, deliberate change in *where* the honest signal lives (file content vs. step outcome), worth a reviewer's attention.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add dependency-graph and build-attempt probe jobs, batched over the 256-job matrix cap"
```

- [ ] **Step 4: Push and verify on a real runner**

Push and confirm: `dependency-graph` produces `ceil(331/12) = 28` batch artifacts, each containing all five per-target files for every target in that batch, non-empty content. `build-attempt` produces 28 batch artifacts; download whichever batch contains `nvptx64-nvidia-cuda` and confirm its `attempt-nvptx64-nvidia-cuda-none-*-*.json` files contain real compiler error messages (per the canary principle — `build_std=none` against a target with no `std` must show real errors, never a clean/empty result) and that the "Assert honest result" step's log shows the `::notice::` for that batch. Also record each `build-attempt` job's actual wall-clock duration from the run — this is the empirical check on Task 6's "12 targets per batch stays under 6 hours" estimate; if any batch approaches the 6-hour limit, the fix is lowering `--max-batch-size` in Task 6's Discovery step (not something to guess at now, only to verify once real duration data exists).

---

### Task 9: Runtime-test probe job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `runtime-test` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.batches`, `needs.discovery.outputs.batch-indices` (Task 6).
- Produces: per-batch artifact `runtime-test-batch-<batch_index>` containing one `runtime-test-result-<target>.json` file per target in that batch, recording either real test execution results or an explicit `"blocked: no runner available, reason: <cited>"` record — consumed by Task 10 (indexing).

- [ ] **Step 1: Add the job**

```yaml
  runtime-test:
    name: "Runtime test: batch ${{ matrix.batch_index }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        batch_index: ${{ fromJSON(needs.discovery.outputs.batch-indices) }}
    steps:
      - uses: actions/checkout@v4
      - name: Determine runner availability and execute or record block, for every target in this batch
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p out
          echo "$BATCH_TARGETS" | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)))' > targets-in-batch.txt
          while IFS= read -r TARGET; do
            case "$TARGET" in
              wasm32*|wasm64*)
                echo '{"status": "runnable", "runner": "wasmtime"}' > "out/runtime-test-result-$TARGET.json"
                # actual wasmtime execution wired in the wasm-family follow-on work;
                # this job records availability honestly either way.
                ;;
              x86_64-unknown-linux-gnu|aarch64-*-linux-*|aarch64-apple-darwin)
                echo '{"status": "runnable", "runner": "native"}' > "out/runtime-test-result-$TARGET.json"
                ;;
              nvptx64-nvidia-cuda)
                echo '{"status": "blocked", "reason": "no free GPU CI runner available for this project (confirmed 2026-08-16)"}' \
                  > "out/runtime-test-result-$TARGET.json"
                ;;
              *)
                echo '{"status": "blocked", "reason": "no known runner for this target"}' \
                  > "out/runtime-test-result-$TARGET.json"
                ;;
            esac
          done < targets-in-batch.txt
      - name: Upload runtime-test results for this batch
        uses: actions/upload-artifact@v4
        with:
          name: runtime-test-batch-${{ matrix.batch_index }}
          path: out/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add runtime-test probe job with explicit blocked-status recording, batched over the 256-job matrix cap"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: `ceil(331/12) = 28` `runtime-test-batch-<N>` artifacts, each containing one result file per target in that batch, and `nvptx64-nvidia-cuda`'s specifically records `"status": "blocked"` with a cited reason — never silently omitted, per the spec's Runtime-test probes section.

---

### Task 10: Indexing job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `indexing` job)

**Interfaces:**
- Consumes: `scripts/target_analysis_indexing.py`'s CLI (Task 3); every batch artifact from Tasks 7-9 via `actions/download-artifact`; `needs.discovery.outputs.target-matrix` and `needs.discovery.outputs.batches` (both unbatched-Python-loop-safe per Task 6's ruling).
- Produces: `index.json` artifact; job fails (`exit 1`) if any expected per-target file is missing.

Completeness now has to be checked at the **file** level, not the artifact-directory level: Tasks 7-9 upload one artifact per *batch* (e.g. `target-capability-batch-3`), each containing many per-target files inside it, rather than one artifact per target. The expected/actual comparison below reflects that directly — `actual` is built from filenames found inside every batch-prefixed artifact directory, not from the directory names themselves.

- [ ] **Step 1: Add the job**

```yaml
  indexing:
    name: Build navigation index
    needs: [discovery, target-capability, dependency-graph, build-attempt, runtime-test]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: artifacts
      - name: Compute expected vs. actual per-target files, and the target-to-batch map
        run: |
          python3 - <<'PYEOF'
          import json
          from pathlib import Path

          target_matrix = json.loads('''${{ needs.discovery.outputs.target-matrix }}''')
          batches = json.loads('''${{ needs.discovery.outputs.batches }}''')

          target_to_batch = {}
          for batch_index, batch_targets in enumerate(batches):
              for t in batch_targets:
                  target_to_batch[t] = batch_index

          expected = set()
          for t in target_matrix:
              expected.add(f"target-spec-{t}.json")
              expected.add(f"cfg-{t}.txt")
              expected.add(f"supported-crate-types-{t}.txt")
              expected.add(f"metadata-{t}.json")
              expected.add(f"tree-features-{t}.txt")
              expected.add(f"tree-duplicates-{t}.txt")
              expected.add(f"unit-graph-{t}.json")
              expected.add(f"deny-{t}.txt")
              expected.add(f"runtime-test-result-{t}.json")
              for build_std in ["none", "std", "core,alloc", "core"]:
                  for cmd in ["check", "clippy", "build"]:
                      for feat in ["default-features", "no-default-features"]:
                          expected.add(f"attempt-{t}-{build_std}-{cmd}-{feat}.json")

          batch_prefixes = (
              "target-capability-batch-", "dependency-graph-batch-",
              "build-attempt-batch-", "runtime-test-batch-",
          )
          actual = set()
          for batch_dir in Path("artifacts").iterdir():
              if batch_dir.is_dir() and batch_dir.name.startswith(batch_prefixes):
                  for f in batch_dir.iterdir():
                      if f.is_file():
                          actual.add(f.name)

          Path("expected-artifacts.json").write_text(json.dumps(sorted(expected)))
          Path("actual-artifacts.json").write_text(json.dumps(sorted(actual)))
          Path("target-to-batch.json").write_text(json.dumps(target_to_batch))
          PYEOF
      - name: Run indexing script per target
        run: |
          python3 - <<'PYEOF'
          import json
          import subprocess
          import sys
          from pathlib import Path

          target_matrix = json.loads('''${{ needs.discovery.outputs.target-matrix }}''')
          target_to_batch = json.loads(Path("target-to-batch.json").read_text())
          combined = {}
          any_missing = False
          for t in target_matrix:
              batch_index = target_to_batch.get(t)
              attempt_file = Path(f"artifacts/build-attempt-batch-{batch_index}/attempt-{t}-none-check-default-features.json")
              spec_file = Path(f"artifacts/target-capability-batch-{batch_index}/target-spec-{t}.json")
              if batch_index is None or not attempt_file.exists() or not spec_file.exists():
                  combined[t] = {"error": "missing required per-target file"}
                  any_missing = True
                  continue
              result = subprocess.run(
                  [
                      "python3", "scripts/target_analysis_indexing.py",
                      "--compiler-messages-file", str(attempt_file),
                      "--target-spec-file", str(spec_file),
                      "--std-mode-errors-file", "/dev/stdin",
                      "--expected-artifacts-file", "expected-artifacts.json",
                      "--actual-artifacts-file", "actual-artifacts.json",
                  ],
                  input="[]",
                  capture_output=True,
                  text=True,
              )
              combined[t] = json.loads(result.stdout)
              if result.returncode != 0:
                  any_missing = True

          Path("index.json").write_text(json.dumps(combined, indent=2))
          sys.exit(1 if any_missing else 0)
          PYEOF
      - name: Upload index
        uses: actions/upload-artifact@v4
        with:
          name: navigation-index
          path: index.json
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add indexing job with per-target file-level completeness enforcement across batched artifacts"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: the `indexing` job runs after all Primary-layer jobs (`needs:` + `!cancelled()`), downloads every batch artifact, produces `navigation-index` containing one entry per target (all 331) with `error_counts_by_target`, `contradictions`, and `missing_artifacts`. Then deliberately break completeness once (e.g. temporarily comment out the `runtime-test` job's artifact upload step in a throwaway commit) and confirm the `indexing` job fails loudly rather than silently indexing a partial set — then revert that throwaway commit.

---

### Task 11: Fix `unexpected_clean_std_build`'s field-path bug — the nvptx contradiction check has been inert since Task 3

**Real bug, surfaced by Task 10's own real-CI verification, traced back to a mistake in
this plan's own Task 3 brief.** `unexpected_clean_std_build(target_spec,
std_mode_errors)` checks `target_spec.get("std") is False` — but real `rustc --print
target-spec-json` output nests this field at `target_spec["metadata"]["std"]`, not at
the top level (confirmed directly, repeatedly, against real downloaded
`target-spec-<target>.json` files this session, e.g. nvptx64-nvidia-cuda's real file:
`"metadata": {"description": ..., "host_tools": false, "std": false, "tier": 2}`, no
top-level `"std"` key at all). `target_spec.get("std")` on real input therefore always
returns `None`, and `None is False` is always `False` — Standing Principle 5's whole
canary contradiction check has never actually fired against real data, for any target,
since Task 3 was written. Task 10's own wiring is correct (it passes the real,
downloaded file content through exactly as specified) — the bug is entirely in the
function itself, and in the fixture Task 3's own brief specified: `target_spec_nvptx
.json` puts `"std": false` at the top level, matching the buggy function's
expectation rather than real `rustc` output, so the existing tests have been passing
against a fixture shaped to match the bug, not against reality.

**Files:**
- Modify: `scripts/target_analysis_indexing.py` (`unexpected_clean_std_build`)
- Modify: `tests/fixtures/target_analysis/target_spec_nvptx.json` (nest `std` under `metadata`, matching real `rustc` output)
- Modify: `tests/test_target_analysis_indexing.py` (update the existing flat-shape test, add a regression test for the exact bug)

**Interfaces:**
- Consumes: nothing new.
- Produces: the same `unexpected_clean_std_build(target_spec: dict, std_mode_errors: list) -> bool` signature, unchanged — only its internal field lookup changes. No caller (Task 10's workflow wiring) needs any edit; once this function is fixed, the contradiction check starts working correctly against the exact same real input Task 10 already feeds it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_target_analysis_indexing.py`:
```python
def test_unexpected_clean_std_build_ignores_stray_top_level_std_key():
    # Real rustc target-spec-json nests std under metadata; a stray top-level
    # "std" key (the exact shape mismatch that made this check inert) must
    # never be mistaken for the real field.
    target_spec = {"std": False, "metadata": {}}
    assert unexpected_clean_std_build(target_spec, std_mode_errors=[]) is False
```

Also update `test_unexpected_clean_std_build_does_not_flag_targets_with_std` (currently
uses a flat, unreal shape) to:
```python
def test_unexpected_clean_std_build_does_not_flag_targets_with_std():
    host_spec = {"metadata": {"std": True}}
    assert unexpected_clean_std_build(host_spec, std_mode_errors=[]) is False
```

- [ ] **Step 2: Fix the fixture to match real `rustc` output shape**

Replace `tests/fixtures/target_analysis/target_spec_nvptx.json`:
```json
{
  "arch": "nvptx64",
  "os": "cuda",
  "std": false,
  "only-cdylib": true,
  "panic-strategy": "abort"
}
```
with:
```json
{
  "arch": "nvptx64",
  "os": "cuda",
  "only-cdylib": true,
  "panic-strategy": "abort",
  "metadata": {
    "std": false
  }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_indexing.py -v`
Expected: FAIL — `test_unexpected_clean_std_build_flags_the_nvptx_canary_contradiction`
and `test_unexpected_clean_std_build_does_not_flag_targets_with_std` now fail (the
fixture no longer has a top-level `std` key, and the buggy function still only checks
the top level), and the new regression test also fails (the current buggy
implementation actually returns `True` for `{"std": False, "metadata": {}}`, not the
required `False` — confirming the bug is real and exactly this shape-mismatch).

- [ ] **Step 4: Fix the implementation**

In `scripts/target_analysis_indexing.py`, replace:
```python
def unexpected_clean_std_build(target_spec: dict[str, Any], std_mode_errors: list[Any]) -> bool:
    return target_spec.get("std") is False and len(std_mode_errors) == 0
```
with:
```python
def unexpected_clean_std_build(target_spec: dict[str, Any], std_mode_errors: list[Any]) -> bool:
    return target_spec.get("metadata", {}).get("std") is False and len(std_mode_errors) == 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_indexing.py -v`
Expected: 7 passed (6 existing, now against the corrected fixture shape, plus the new regression test)

- [ ] **Step 6: Commit**

```bash
git add scripts/target_analysis_indexing.py tests/fixtures/target_analysis/target_spec_nvptx.json tests/test_target_analysis_indexing.py
git commit -m "fix: unexpected_clean_std_build was reading target-spec-json's std field at the wrong path"
```

- [ ] **Step 7: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real run's
`indexing` job output and confirm `navigation-index`'s `contradictions
.unexpected_clean_std_build` now reads `true` for `nvptx64-nvidia-cuda`'s entry — this
is the first real run where that canary check actually fires, per its originally
intended purpose (Standing Principle 5), rather than silently reading `false` for
every target the way it always has until now.

---

### Task 12: Scope Discovery to rustc tier 1+2 targets

**Real evidence driving this** (from Task 10's real run against the current, unscoped
331-target universe): downloading all 331 real `target-spec-json` files and tallying
rustc's own `metadata.tier` field shows 212 targets (64.0%) are tier 3 — rustc's own
"community-maintained, may or may not build" classification — and the large majority of
tier-2/tier-3 targets also lack `host_tools`. Building `larql-cli` (a real,
dependency-heavy CLI application) against these is near-certain to fail before revealing
anything specific to larql-cli's own no_std-readiness; the failure mostly just
re-confirms rustc's own tier classification, at real, recurring cost given this
pipeline's many-round recursive design (~2 hours/round, dominated by `build-attempt`).

Decided directly with the user: scope Discovery's target resolution to `tier <= 2` (119
targets: 8 tier-1 + 111 tier-2) — sourced from `target-spec-json`'s own `metadata.tier`
field, real L1 data, not agent judgment, so this does not violate Standing Principle 2's
no-agent-filtering rule (a mechanical threshold on rustc's own emitted classification is
categorically different from an agent's guess about "relevance"). Confirmed compatible
with Standing Principle 5 (`nvptx64-nvidia-cuda` as standing canary): nvptx is tier 2
(confirmed both via a real downloaded `target-spec-json` and cross-checked against
https://doc.rust-lang.org/rustc/platform-support.html#tier-2-without-host-tools), so it
survives this cut without needing a carve-out.

This changes Discovery's own resolution step only: before batching, Discovery queries
`target-spec-json`'s tier for every real target in `rustc --print target-list`, then
filters to tier ≤ 2 before computing `matrix`/`batches`/`batch-indices`. Every subsequent
round of this pipeline (after Task 10's already-in-flight run against the un-scoped
331-target universe, which stands as valid ground truth and is not retroactively
invalidated) fans out over 119 targets → `ceil(119/12) = 10` batches, not 331/28 — no
changes needed anywhere downstream (Tasks 7-10, 16-19 all derive everything from
Discovery's real output, never hardcoding 331/28 anywhere).

A `workflow_dispatch`-requested single target explicitly bypasses this filter — someone
deliberately asking to analyze one specific target (including a tier-3 one) is a real,
explicit request that should never be silently narrowed by a default-scope filter meant
for the automatic `push`-triggered path.

**Files:**
- Modify: `scripts/target_analysis_discovery.py` (add `filter_by_tier`, extend `main()`)
- Modify: `tests/test_target_analysis_discovery.py` (add tests for `filter_by_tier` and the extended CLI)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add a tier-query step to the `discovery` job, before "Resolve target matrix")

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `filter_by_tier(targets_with_tiers: dict[str, int | None], max_tier: int) -> list[str]`. The `discovery` job's new "Query tier for every real target" step produces `target-tiers.json` (a JSON object mapping target name → tier, or `null` if rustc reported none), consumed by `main()`'s new `--target-tiers-file`/`--max-tier` flags.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_target_analysis_discovery.py`:
```python
from scripts.target_analysis_discovery import filter_by_tier


def test_filter_by_tier_keeps_only_tier_at_or_below_max():
    targets_with_tiers = {"a": 1, "b": 2, "c": 3, "d": None}
    assert filter_by_tier(targets_with_tiers, max_tier=2) == ["a", "b"]


def test_filter_by_tier_excludes_null_tier():
    targets_with_tiers = {"a": None}
    assert filter_by_tier(targets_with_tiers, max_tier=3) == []


def test_filter_by_tier_max_tier_3_keeps_everything_with_a_tier():
    targets_with_tiers = {"a": 1, "b": 2, "c": 3}
    assert filter_by_tier(targets_with_tiers, max_tier=3) == ["a", "b", "c"]


def test_main_applies_tier_filter_when_no_target_requested(tmp_path):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("a\nb\nc\nd\n", encoding="utf-8")
    tiers_file = tmp_path / "tiers.json"
    tiers_file.write_text(json.dumps({"a": 1, "b": 2, "c": 3, "d": None}), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "",
        "--target-tiers-file", str(tiers_file),
        "--max-tier", "2",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    matrix_line = next(line for line in output_text.splitlines() if line.startswith("matrix="))
    matrix = json.loads(matrix_line[len("matrix="):])
    assert matrix == ["a", "b"]


def test_main_skips_tier_filter_when_specific_target_requested(tmp_path):
    target_list_file = tmp_path / "target-list.txt"
    target_list_file.write_text("a\nb\nc\n", encoding="utf-8")
    tiers_file = tmp_path / "tiers.json"
    tiers_file.write_text(json.dumps({"a": 1, "b": 2, "c": 3}), encoding="utf-8")
    github_output = tmp_path / "github_output.txt"
    sys.argv = [
        "target_analysis_discovery.py",
        "--target-list-file", str(target_list_file),
        "--requested-target", "c",
        "--target-tiers-file", str(tiers_file),
        "--max-tier", "2",
        "--github-output", str(github_output),
    ]
    assert main() == 0
    output_text = github_output.read_text(encoding="utf-8")
    matrix_line = next(line for line in output_text.splitlines() if line.startswith("matrix="))
    matrix = json.loads(matrix_line[len("matrix="):])
    assert matrix == ["c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: FAIL — `ImportError: cannot import name 'filter_by_tier'` for the first 3 new tests; the two `main()` tests also fail (no `--target-tiers-file`/`--max-tier` flags exist yet).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/target_analysis_discovery.py`, above `main()`:
```python
def filter_by_tier(targets_with_tiers: dict[str, int | None], max_tier: int) -> list[str]:
    return [
        target
        for target, tier in targets_with_tiers.items()
        if tier is not None and tier <= max_tier
    ]
```

Replace `main()` (currently, from Task 6):
```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)
    batches = chunk_targets(matrix, max_size=args.max_batch_size)
    batch_indices = list(range(len(batches)))

    print(json.dumps(matrix))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"batches={json.dumps(batches)}\n")
            handle.write(f"batch-indices={json.dumps(batch_indices)}\n")
    return 0
```
with:
```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list-file", required=True, type=Path)
    parser.add_argument("--requested-target", default=None)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--github-output", type=Path, default=None)
    parser.add_argument("--target-tiers-file", type=Path, default=None)
    parser.add_argument("--max-tier", type=int, default=None)
    args = parser.parse_args()

    raw = args.target_list_file.read_text(encoding="utf-8")
    all_targets = parse_target_list(raw)
    requested = args.requested_target or None
    matrix = resolve_target_matrix(all_targets, requested)

    if requested is None and args.target_tiers_file is not None and args.max_tier is not None:
        target_tiers = json.loads(args.target_tiers_file.read_text(encoding="utf-8"))
        matrix = filter_by_tier(target_tiers, max_tier=args.max_tier)

    batches = chunk_targets(matrix, max_size=args.max_batch_size)
    batch_indices = list(range(len(batches)))

    print(json.dumps(matrix))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"batches={json.dumps(batches)}\n")
            handle.write(f"batch-indices={json.dumps(batch_indices)}\n")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_target_analysis_discovery.py -v`
Expected: 17 passed (12 existing from Tasks 2/5/6 plus 5 new)

- [ ] **Step 5: Commit**

```bash
git add scripts/target_analysis_discovery.py tests/test_target_analysis_discovery.py
git commit -m "feat: add tier-based target filtering to discovery script"
```

- [ ] **Step 6: Add the tier-query step to the discovery job and wire the new flags**

Modify `.github/workflows/target-analysis-pipeline.yml`'s `discovery` job to exactly:
```yaml
  discovery:
    name: Discover target matrix
    runs-on: ubuntu-latest
    outputs:
      target-matrix: ${{ steps.resolve.outputs.matrix }}
      batches: ${{ steps.resolve.outputs.batches }}
      batch-indices: ${{ steps.resolve.outputs.batch-indices }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: rustc --print target-list
        run: rustup run nightly rustc --print target-list > target-list.txt
      - name: Query tier for every real target
        run: |
          : > tier-lines.txt
          while IFS= read -r TARGET; do
            TIER=$(rustup run nightly rustc --print target-spec-json -Z unstable-options --target "$TARGET" 2>/dev/null \
              | python3 -c 'import json, sys
          try:
              d = json.load(sys.stdin)
              print(d.get("metadata", {}).get("tier", ""))
          except Exception:
              print("")')
            printf '%s\t%s\n' "$TARGET" "$TIER" >> tier-lines.txt
          done < target-list.txt
          python3 -c "
          import json
          tiers = {}
          with open('tier-lines.txt') as f:
              for line in f:
                  target, tier = line.rstrip(chr(10)).split(chr(9), 1)
                  tiers[target] = int(tier) if tier else None
          with open('target-tiers.json', 'w') as f:
              json.dump(tiers, f)
          "
      - name: Resolve target matrix
        id: resolve
        env:
          REQUESTED_TARGET: ${{ inputs.target }}
        run: |
          python3 scripts/target_analysis_discovery.py \
            --target-list-file target-list.txt \
            --requested-target "$REQUESTED_TARGET" \
            --max-batch-size 12 \
            --target-tiers-file target-tiers.json \
            --max-tier 2 \
            --github-output "$GITHUB_OUTPUT" > /dev/null
      - name: Upload raw target-list and tiers
        uses: actions/upload-artifact@v4
        with:
          name: discovery-target-list
          path: |
            target-list.txt
            target-tiers.json
```
`printf '%s\t%s\n'` (not `echo`) is used for the tab-separated intermediate format specifically because target names and tier values never contain tabs, making a fixed single-tab-split parse in the Python conversion step unambiguous and simple — deliberately avoiding a CSV/JSON-per-line format that would need real escaping for this trivial two-field case.

The embedded Python (both the `python3 -c '...'` inside the loop and the `python3 -c "..."` after it) is indented to match the `run: |` block scalar's own minimum indentation (10 spaces here), not flush to column 0 — this was a real bug caught during this task's own dispatch: the plan originally had the embedded Python at column 0, which YAML parsers reject outright (a `ScannerError` — a line below the block's established minimum indentation ends the scalar early). The Python code's own semantics are unaffected either way (Python only cares about indentation *relative to itself*, and shell's single/double-quoted strings don't care about indentation at all) — only the YAML container's requirement changes what's valid here. Verified against the real, working, committed version of this file, not just reasoned about.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: query rustc's own tier classification and scope discovery to tier 1+2"
```

- [ ] **Step 8: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real run's `discovery` job log and confirm: the "Query tier for every real target" step completes for all 331 real targets (this step alone should take a few minutes — 331 lightweight `rustc --print` invocations, no compilation), `target-tiers.json` is uploaded alongside `target-list.txt`, and `steps.resolve.outputs.matrix` now contains exactly 119 targets (down from 331) — cross-check this count directly against this task's own real tier tally (8 tier-1 + 111 tier-2 = 119) rather than assuming the plan's number is still accurate, since the exact real target list can drift between runs. Confirm `nvptx64-nvidia-cuda` is still present in the filtered matrix (it must be — tier 2, the standing canary). Confirm `steps.resolve.outputs.batches`/`batch-indices` reflect `ceil(119/12) = 10` batches, not 28. This run will re-trigger the full pipeline (all jobs depend on `discovery`) — expect a real run against the new, smaller 119-target/10-batch universe end to end; this is the first real evidence of the pipeline running at its new, intended scope, not just Discovery in isolation.

---

### Task 13: Add a workflow-level concurrency group and least-privilege permissions to every job

**User directive: concurrency and least-privilege should always be used, not a
one-time fix.** Two real, separate findings motivate this:

1. **Concurrency.** Task 12's real run lost roughly 40 minutes to runner-queue
   contention from a stale, superseded prior run on the same branch — this workflow
   has no `concurrency:` group anywhere, so a new push doesn't cancel an in-flight
   run before starting a fresh one. GitHub's own mechanism for exactly this is a
   workflow-level `concurrency:` block with `cancel-in-progress: true`.
2. **Least privilege.** Of the six jobs currently in this workflow, only `indexing`
   (Task 10) declares an explicit `permissions:` block (`contents: read` +
   `actions: read`, the latter specifically because it calls `actions/download-
   artifact`). Every other job — `discovery`, `target-capability`, `dependency-graph`,
   `build-attempt`, `runtime-test` — has no `permissions:` block at all, meaning each
   one inherits whatever the *repository's own* default `GITHUB_TOKEN` permissions
   happen to be set to, rather than an explicit, auditable, minimal grant declared in
   the workflow file itself.

This is now a standing Global Constraint (added above), not just a fix for these six
jobs — every task from here on that adds a new job must declare its own explicit
`permissions:` block following the same pattern (`contents: read` floor; add
`actions: read` only if that job calls `actions/download-artifact`).

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add top-level `concurrency:`; add `permissions:` to the five jobs currently missing it)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — this task changes scheduling/authorization behavior only, no new outputs or artifacts.

- [ ] **Step 1: Add the workflow-level concurrency group**

Modify the top of `.github/workflows/target-analysis-pipeline.yml`, adding a
`concurrency:` block alongside the existing `on:` block:
```yaml
name: target-analysis-pipeline

on:
  push:
    branches:
      - "experiment/target-analysis-*"
  workflow_dispatch:
    inputs:
      target:
        description: "Single target triple to analyze (optional; omit for the full rustc target-list)"
        required: false
        type: string

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
```
Grouping by `github.ref` (not just `github.workflow`) means only pushes to the *same*
branch cancel each other — a `workflow_dispatch` run against a different target or a
push to a different `experiment/target-analysis-*` branch is unaffected.

- [ ] **Step 2: Add `permissions: {contents: read}` to the four jobs that never download artifacts**

Add this block immediately after each job's `runs-on: ubuntu-latest` line — for
`discovery`, `target-capability`, `dependency-graph`, and `runtime-test`:
```yaml
    permissions:
      contents: read   # actions/checkout
```

- [ ] **Step 3: Add `permissions: {contents: read, actions: read}` to `build-attempt`**

`build-attempt` is the one remaining job without a `permissions:` block that *does*
call `actions/download-artifact` (to fetch `target-capability-batch-<N>` for the
crate-type canary), so it needs the same two-line grant `indexing` already has:
```yaml
    permissions:
      contents: read   # actions/checkout
      actions: read    # actions/download-artifact
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add workflow-level concurrency group and least-privilege permissions to every job"
```

- [ ] **Step 5: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real
run and confirm every job still succeeds with its new, tightened `permissions:`
grant — if any job fails with a permission-denied error, that's real evidence it
needs something this task's minimal grant didn't anticipate; add exactly what the
error demands, nothing more, and re-verify. Also push a *second*, immediate
follow-up commit (a trivial no-op change, e.g. a comment) while the first run is
still in progress, and confirm via the real run list that the first run gets
cancelled (`conclusion: cancelled`) rather than left to queue-contend with the
second — this is the concurrency group's own real-world proof, not just a
plausible-looking YAML block.

---

### Task 14: Target-independent checks run before target-dependent jobs (fmt)

**User directive, general and forward-looking:** anything target-independent must run
before any target-dependent job. This mirrors the Task 16 restructure's own finding
(Stage A/B/B2/B3 never reference `--target`, so matrixing them per-target was pure
waste) generalized into a standing sequencing rule, not a one-off fix. `cargo fmt
--check` is the clearest, most immediately-buildable case: it operates on the AST/text
directly, takes no `--target` at all, and its outcome is identical regardless of which
of the 331 (soon 119) targets end up in scope — every target-dependent job in this
pipeline (`target-capability`, `dependency-graph`, `build-attempt`, `runtime-test`) is
still expected to run in full regardless of `fmt`'s own findings (Standing Principle 3
— nothing is skipped because an earlier result seems to already explain things); this
task only reorders *when* the target-independent work happens, using the exact
step-level `continue-on-error` + explicit `.outcome`-assertion honest-result pattern
already proven in Tasks 1-9 (not the batch-loop variant `build-attempt` needed — 5
crates need no such combinatorial workaround), so ordering costs nothing in
exhaustiveness: the job reports job-level success regardless of a real fmt violation
(exactly like every other honest-result job in this pipeline), so the downstream
`needs:` edges below never block or skip target-dependent work.

`cargo-semver-checks` (needs a chosen baseline — the last published crates.io version,
or a specific git rev — not yet decided), `cargo udeps`, `cargo miri` (its own
quasi-target, an interpreter rather than a real backend), `cargo hack`
(feature-powerset testing), and broadening `build-attempt`'s clippy invocation to the
spec's own stated lint breadth (`clippy::all`/`pedantic`/`nursery`/`cargo` — currently
running bare `cargo clippy` with no lint-group flags, a real, confirmed gap) are
explicitly **not** built here — flagged as follow-up work below, each needing its own
design decision this task doesn't make.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `target-independent-checks` job; add it to `needs:` on `target-capability`, `dependency-graph`, `build-attempt`, `runtime-test`)

**Interfaces:**
- Consumes: `needs: [discovery]` only (a real, minimal prerequisite — same pattern every other job already uses; `fmt` needs nothing from Discovery's own output, but waiting on Discovery's success is consistent with every other job in this pipeline).
- Produces: per-crate `target-independent-checks-<crate>` artifact containing `fmt-output.txt`. Deliberately NOT folded into Task 10's `navigation-index` — that index is organized per-target, and `fmt` findings are crate-level, not target-level, so they have no natural slot in it; this is a distinct, self-contained finding retrievable via its own artifact and the run's own honest-result log output, not an oversight.

- [ ] **Step 1: Add the job**

```yaml
  target-independent-checks:
    name: "Target-independent checks: fmt (${{ matrix.crate }})"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        crate: [larql-boundary, larql-vindex-spec, larql-models, larql-compute, larql-cli]
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add rustfmt --toolchain nightly
      - name: "cargo fmt --check: ${{ matrix.crate }}"
        id: fmt
        continue-on-error: true
        run: cargo +nightly fmt -p "${{ matrix.crate }}" --check > fmt-output.txt 2>&1
      - name: Upload fmt-check output
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: target-independent-checks-${{ matrix.crate }}
          path: fmt-output.txt
      - name: Assert honest result
        run: |
          echo "cargo fmt --check outcome: ${{ steps.fmt.outcome }}"
          if [ "${{ steps.fmt.outcome }}" = "failure" ]; then
            echo "::notice::Real fmt violation recorded as a finding, not masked."
          fi
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add target-independent fmt-check job"
```

- [ ] **Step 3: Wire it as a real prerequisite for every target-dependent job**

Modify each of the four target-dependent jobs' `needs:` line, exactly:

`target-capability` — change `needs: [discovery]` to `needs: [discovery, target-independent-checks]`.

`dependency-graph` — change `needs: [discovery]` to `needs: [discovery, target-independent-checks]`.

`build-attempt` — change `needs: [discovery, target-capability]` to `needs: [discovery, target-capability, target-independent-checks]`.

`runtime-test` — change `needs: [discovery]` to `needs: [discovery, target-independent-checks]`.

`target-independent-checks`'s own honest-result design (Step 1 — job-level success regardless of a real `fmt` violation, per the pattern every other job in this pipeline already uses) means these four `needs:` additions never block or skip a target-dependent job over a `fmt` finding — they only guarantee `fmt` starts and (for its own five small, fast job instances) generally finishes first, without an `if: !cancelled()` override being necessary anywhere.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: sequence target-dependent jobs after the target-independent fmt-check"
```

- [ ] **Step 5: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real run's job list and confirm: 5 `target-independent-checks` job instances (one per crate) start and typically finish well before `target-capability`/`dependency-graph`/`build-attempt`/`runtime-test` begin (check real start timestamps in the run's job list, not just declared `needs:` — GitHub schedules dependents only after their full `needs:` list completes, so this should hold structurally, but confirm it against real timing data since that's the actual, mechanical proof, not the YAML's stated intent). Confirm each `target-independent-checks-<crate>` artifact uploads `fmt-output.txt` regardless of whether `cargo fmt --check` found anything, and that a real fmt violation (if any exists in the current tree) produces the `::notice::` in that job's log without preventing `target-capability` et al. from running to completion afterward.

---

### Task 15: Trim build-attempt's `build_std` axis to `none` only in the Primary layer

**User-directed correction to already-merged, already-CI-verified work** (Task 8):
walking through what each of the four `build_std` modes actually reveals, three of
them turn out to add real, recurring cost with little or no marginal signal in the
Primary layer:

- `none` (no `-Z` flag, uses whatever std/core is actually available): the real,
  meaningful "does this build for this target" question. Works on stable for the 82
  `std: true` targets. For the 34 `std: false` targets it already produces the
  "can't find crate for `std`" failure Standing Principle 5's nvptx canary requires —
  no `-Z build-std` needed to get that result. **Kept, unconditionally.**
- `std` (recompile the full standard library from source): for `std: true` targets,
  recompiling `std` from source doesn't change whether `larql-cli` compiles against it
  — the API surface is identical to the prebuilt version, built by the same compiler.
  For `std: false` targets, this fails for the same structural, OS-support reason
  `none` already reveals, uninterestingly. No real justification found either way.
  **Dropped.**
- `core`/`core,alloc` (recompile a minimal, OS-less subset from source): the one mode
  that tests something `none` structurally cannot — but only once the source has
  actually been rewritten to be `#![no_std]`. Run against the *pristine, pre-mutation*
  checkout — what Task 8's `build-attempt` job does — every crate with any `use
  std::...` anywhere fails identically on every target ("unresolved import
  `std::...`"), regardless of the target's real capabilities: the failure is about the
  *source*, not the target. This mode's real value is concentrated exactly where it's
  already built: Task 17's Stage C, which runs `build_std=core,alloc` against the
  *mutated* tree, where the question actually differentiates by target. **Dropped from
  the Primary layer; unchanged in the Secondary layer (Task 17).**

This directly reduces `build-attempt`'s per-target combinations from `4 build_std × 3
cargo_cmd × 2 features = 24` to `1 × 3 × 2 = 6` — roughly a 4x reduction in the
dominant cost driver of the pipeline's real per-round wall-clock (`build-attempt`
averaged 54.0 min/batch across 28 batches in Task 8's real run).

**A related, confirmed-with-real-data refinement surfaced while discussing this:**
`wasm32-unknown-unknown` (tier 2, `std: true`, `host_tools: false`) is a third,
distinct standing-canary *category* from the two `std: false` canaries
(`nvptx64-nvidia-cuda`, `wasm32v1-none`) — its `std: true` doesn't mean "fully
OS-backed std" (no real filesystem, no real threads without special setup, no
sockets), the same kind of misleading-field-name situation this session already
caught once with `only-cdylib`. Once the Secondary layer's mutation exists, testing
the mutated, `core`/`alloc`-restricted code against `wasm32-unknown-unknown` asks a
genuinely different question than testing it against the two `std: false` canaries —
not "is there no OS at all" (already known for those two), but "does the mutation
also hold up against a target where `std` nominally exists but is OS-limited." Since
it's tier 2, it's already inside Task 12's rescoped 119-target universe and already
covered by Task 17's Stage C batch — no new task needed for coverage — but it's worth
naming explicitly here so it doesn't just blend anonymously into the batch.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`build-attempt` job, already merged in Task 8)

**Interfaces:**
- Consumes: nothing new.
- Produces: the same artifact naming scheme as before (`attempt-<target>-none-<cmd>-<features>.json`) — `none` is kept literally in the filename even though it's now the only value, specifically because Task 10's already-merged indexing job already hardcodes exactly this filename pattern (`attempt-{t}-none-check-default-features.json`) for its per-target parse — Task 10 never read the `std`/`core`/`core,alloc` files in the first place, so this change requires zero downstream edits.

- [ ] **Step 1: Trim the `build_std` loop**

In `.github/workflows/target-analysis-pipeline.yml`'s `build-attempt` job, replace:
```yaml
            for BUILD_STD in none std core,alloc core; do
              if [ "$BUILD_STD" = "none" ]; then BUILD_STD_FLAG=""; else BUILD_STD_FLAG="-Z build-std=$BUILD_STD"; fi
              for CARGO_CMD in check clippy build; do
                for FEATURES in default-features no-default-features; do
                  if [ "$FEATURES" = "no-default-features" ]; then FEATURES_FLAG="--no-default-features"; else FEATURES_FLAG=""; fi
                  OUTFILE="out/attempt-$TARGET-$BUILD_STD-$CARGO_CMD-$FEATURES.json"
                  echo "=== target=$TARGET build_std=$BUILD_STD cmd=$CARGO_CMD features=$FEATURES ==="
                  cargo +nightly "$CARGO_CMD" -p larql-cli \
                    --target "$TARGET" $FEATURES_FLAG $BUILD_STD_FLAG \
                    --keep-going --message-format=json > "$OUTFILE" || FAILED=1
                done
              done
            done
```
with:
```yaml
            BUILD_STD=none
            for CARGO_CMD in check clippy build; do
              for FEATURES in default-features no-default-features; do
                if [ "$FEATURES" = "no-default-features" ]; then FEATURES_FLAG="--no-default-features"; else FEATURES_FLAG=""; fi
                OUTFILE="out/attempt-$TARGET-$BUILD_STD-$CARGO_CMD-$FEATURES.json"
                echo "=== target=$TARGET build_std=$BUILD_STD cmd=$CARGO_CMD features=$FEATURES ==="
                cargo +nightly "$CARGO_CMD" -p larql-cli \
                  --target "$TARGET" $FEATURES_FLAG \
                  --keep-going --message-format=json > "$OUTFILE" || FAILED=1
              done
            done
```
`-Z build-std=$BUILD_STD` and its conditional are gone entirely — `none` mode never
used the flag, and it's the only mode left.

- [ ] **Step 2: Remove the now-unused `rust-src` component install**

`rust-src` was only ever needed for `-Z build-std`, which this job no longer invokes
anywhere. In the same job's "Install nightly Rust" step, remove:
```yaml
          rustup component add rust-src --toolchain nightly
```
leaving:
```yaml
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add clippy --toolchain nightly
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: trim build-attempt's build_std axis to none-only in the Primary layer"
```

- [ ] **Step 4: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real run's
`build-attempt` job durations and confirm they dropped substantially from Task 8's
real baseline (54.0 min/batch average) — roughly a 4x reduction is the mechanical
expectation given the combination count dropped from 24 to 6 per target, but confirm
the real number rather than assuming the estimate holds exactly. Confirm each
`attempt-<target>-none-<cmd>-<features>.json` file is still produced and non-empty,
and that `nvptx64-nvidia-cuda`'s `none`-mode files still show the same real `E0463`
compiler errors Task 8 already established. Confirm no `std`/`core`/`core,alloc`
files exist in the uploaded artifacts anymore (the axis is genuinely gone, not just
hidden).

---

### Task 16: Generalize the Secondary-layer mutation stages (A, B, B2, B3) — single job, no target matrix

**Restructured from the original two-job (per-crate × per-target) design.** All four mutation stages are target-independent: Stage A runs `clippy --fix` against the host target (never `--target`), Stage B is a pure text edit to `lib.rs`, and Stage B2/B3 are pure text edits to `Cargo.toml` files — none of them reference a target triple at all. The original draft matrixed this identical, target-independent work over all 331 targets (and, for Stage A/B, over 5 crates too — 1655 combinations), for no reason: target only enters the Secondary layer at Stage C. This version runs the whole mutation pipeline exactly once per pipeline run, producing a single patch that Task 17's batched Stage C job downloads and applies before checking against each target — this is also what fixes a real bug the original draft had: without an explicit patch-apply step, Stage C would have run against a fresh, unmutated checkout every time, silently checking pristine source instead of the mutation it was supposed to be evaluating.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-mutate` job, adapted from `experiment-cuda-nvptx.yml`'s existing `nostd-fix-attempt` job's Stage A/B/B2/B3 steps at `.github/workflows/experiment-cuda-nvptx.yml:477-624`)

**Interfaces:**
- Consumes: nothing target-specific. `needs: [discovery, indexing]` expresses a real ordering constraint even without consuming `target-matrix` directly — this stage's whole purpose (per Standing Principle 6) is to be validated against a Primary-layer baseline, so it still waits for the Primary layer's indexing to complete first.
- Produces: one `secondary-mutation` artifact containing `full-mutation.patch` (a single `git diff` of the whole tree after all four stages), per-crate `baseline-lib-rs-<crate>.txt` / `sibling-lib-rs-<crate>.txt` pairs (Stage B promotion input), and `baseline-metadata.json` (the unmutated `cargo metadata` output, captured before Stage B3 trims workspace members — Stage B3 promotion input). Consumed by Task 17.

- [ ] **Step 1: Add the job, reusing the existing workflow's proven Stage A/B/B2/B3 logic, target-independent**

```yaml
  secondary-mutate:
    name: "Secondary layer: mutate (Stages A, B, B2, B3)"
    needs: [discovery, indexing]
    runs-on: ubuntu-latest
    env:
      CRATES: "larql-boundary larql-vindex-spec larql-models larql-compute larql-cli"
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add clippy --toolchain nightly
      - name: Capture pre-mutation lib.rs content per crate and pre-mutation workspace metadata
        run: |
          mkdir -p out
          for CRATE in $CRATES; do
            cp "crates/$CRATE/src/lib.rs" "out/baseline-lib-rs-$CRATE.txt"
          done
          cargo metadata --format-version 1 --no-deps > out/baseline-metadata.json
      - name: "Stage A: mechanical std->core/alloc rewrite (host target), per crate"
        run: |
          for CRATE in $CRATES; do
            cargo +nightly clippy --fix --allow-dirty --allow-no-vcs \
              -p "$CRATE" -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc
          done
      - name: "Stage B: inject #![no_std] scaffold, per crate"
        run: |
          for CRATE in $CRATES; do
            python3 - "$CRATE" <<'PYEOF'
          import sys
          from pathlib import Path

          crate = sys.argv[1]
          lib_rs = Path(f"crates/{crate}/src/lib.rs")
          text = lib_rs.read_text(encoding="utf-8")
          lines = text.splitlines(keepends=True)

          insert_at = 0
          for i, line in enumerate(lines):
              stripped = line.strip()
              if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
                  insert_at = i + 1
              else:
                  break

          scaffold = "#![no_std]\nextern crate alloc;\n"
          lines.insert(insert_at, scaffold)
          lib_rs.write_text("".join(lines), encoding="utf-8")
          PYEOF
          done
      - name: Capture post-Stage-B lib.rs content per crate
        run: |
          for CRATE in $CRATES; do
            cp "crates/$CRATE/src/lib.rs" "out/sibling-lib-rs-$CRATE.txt"
          done
      - name: "Stage B2: patch std-defaulting dependency features"
        id: stage-b2
        background: true
        run: |
          find crates -name Cargo.toml -exec \
            sed -i -E 's/serde = \{ workspace = true([^}]*)\}/serde = { workspace = true, default-features = false, features = ["alloc", "derive"]\1}/' {} \;
      - name: "Stage B3: trim workspace members to larql-cli's real tree"
        id: stage-b3
        background: true
        run: |
          python3 - <<'PYEOF'
          import re
          from pathlib import Path

          root_toml = Path("Cargo.toml")
          text = root_toml.read_text(encoding="utf-8")
          reachable = [
              "crates/larql-cli", "crates/larql-boundary", "crates/larql-vindex-spec",
              "crates/larql-models", "crates/larql-compute",
          ]
          members_block = "members = [\n" + "".join(f'    "{c}",\n' for c in reachable) + "]\n"
          text = re.sub(r"members = \[[^\]]*\]\n", members_block, text, count=1)
          root_toml.write_text(text, encoding="utf-8")
          PYEOF
      - name: Wait for B2 and B3
        wait-all: [stage-b2, stage-b3]
      - name: Capture the full mutated-tree patch
        run: git diff > out/full-mutation.patch
      - uses: actions/upload-artifact@v4
        with:
          name: secondary-mutation
          path: out/
```

The scaffold-insertion Python is the exact fix this session already verified against a real 71-line doc comment (inserting after any leading `//!`/`#![`/blank-line block, never before it); the Stage B2 sed pattern and Stage B3 reachable-crate list are the exact ones this session verified against real CI output — all reused directly, not reinvented. `baseline-metadata.json` is captured before Stage B3 runs specifically so Task 17 never needs to reconstruct the unmutated workspace member list by parsing `git show`-retrieved TOML text — it's just read directly from this artifact.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add generalized Secondary-layer mutation job (Stages A, B, B2, B3), target-independent"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: the `secondary-mutate` job runs exactly once (no matrix), Stage A/B applies cleanly for every crate (no doc-comment corruption — check `larql-compute`'s output specifically, the crate with the largest leading doc comment), the `stage-b2`/`stage-b3` steps' start timestamps overlap (background/wait proven concurrent, matching Standing Principle 8's genuine-dependency-only sequencing), and the uploaded `secondary-mutation` artifact contains `full-mutation.patch`, 5 baseline/sibling lib.rs pairs, and `baseline-metadata.json`.

---

### Task 17: Stage C and the promotion/depth-advancement decision, batched

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-stage-c-and-promotion` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.batches`/`batch-indices` (Task 6); Task 16's `secondary-mutation` artifact (the patch plus Stage B/B3 baselines); Task 7's `target-capability-batch-<N>` and Task 8's `dependency-graph-batch-<N>` artifacts (the mechanically-grounded Primary-layer baselines for the Stage B2 promotion check — the per-target unmutated `unit-graph-<target>.json` this job would otherwise have no other source for); `scripts/target_analysis_promotion.py`'s CLI (Task 4).
- Produces: `promotion-decision-batch-<batch_index>` artifact containing, per target in that batch, the stage-b/b2/b3 promotion verdicts and the depth-advancement decision — the actual mechanical output this session's "measurable difference" discussion exists to produce.

**The critical fix this task makes over the original draft:** the original version did a fresh `actions/checkout@v4` and ran `cargo check` directly, with no step ever applying Task 16's mutation — Stage C would have silently checked pristine, unmutated source on every run, and the whole promotion/depth-advancement machinery would have been evaluating data that never reflected the mutation it claimed to evaluate. This version's very first non-checkout step downloads `secondary-mutation` and runs `git apply mutation/full-mutation.patch` before anything else.

- [ ] **Step 1: Add the job**

```yaml
  secondary-stage-c-and-promotion:
    name: "Secondary Stage C + promotion: batch ${{ matrix.batch_index }}"
    needs: [discovery, indexing, secondary-mutate, target-capability, dependency-graph]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        batch_index: ${{ fromJSON(needs.discovery.outputs.batch-indices) }}
    steps:
      - uses: actions/checkout@v4
      - name: Download the Secondary layer's mutation artifact
        uses: actions/download-artifact@v4
        with:
          name: secondary-mutation
          path: mutation
      - name: Apply the mutation patch to this job's checkout
        run: git apply mutation/full-mutation.patch
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add rust-src --toolchain nightly
      - name: Download this batch's target-capability and dependency-graph artifacts (Primary-layer baselines)
        uses: actions/download-artifact@v4
        with:
          pattern: "*-batch-${{ matrix.batch_index }}"
          path: primary
      - name: Stage C and promotion checks for every target in this batch
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p out
          echo "$BATCH_TARGETS" | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)))' > targets-in-batch.txt
          while IFS= read -r TARGET; do
            echo "=== Stage C: target=$TARGET ==="
            cargo +nightly check -p larql-cli --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "out/stage-c-$TARGET.json" || true

            python3 - "$TARGET" <<'PYEOF'
          import json
          import subprocess
          import sys
          from pathlib import Path

          target = sys.argv[1]

          # Stage B (no_std scaffold): real baseline/sibling lib.rs content
          # captured by the mutation job, target-independent (larql-cli's own file).
          baseline_b = {"lib_rs_content": Path("mutation/baseline-lib-rs-larql-cli.txt").read_text(encoding="utf-8")}
          sibling_b = {"lib_rs_content": Path("mutation/sibling-lib-rs-larql-cli.txt").read_text(encoding="utf-8")}
          Path(f"baseline-b-{target}.json").write_text(json.dumps(baseline_b))
          Path(f"sibling-b-{target}.json").write_text(json.dumps(sibling_b))

          # Stage B2 (serde features): baseline is the Primary layer's own
          # unmutated unit-graph for this target (dependency-graph-batch-N,
          # already downloaded — never recomputed, it's real mechanically-
          # grounded L1 data this job would otherwise have no other source
          # for); sibling is computed fresh here since the patch is now
          # applied to this job's own checkout.
          baseline_unit_graph_path = next(Path("primary").glob(f"dependency-graph-batch-*/unit-graph-{target}.json"))
          baseline_unit_graph = json.loads(baseline_unit_graph_path.read_text(encoding="utf-8"))
          sibling_unit_graph_raw = subprocess.run(
              ["cargo", "+nightly", "build", "-Z", "unstable-options", "--unit-graph",
               "-p", "larql-cli", "--target", target],
              capture_output=True, text=True,
          ).stdout
          Path(f"baseline-b2-{target}.json").write_text(json.dumps({"unit_graph": baseline_unit_graph}))
          Path(f"sibling-b2-{target}.json").write_text(json.dumps({"unit_graph": json.loads(sibling_unit_graph_raw)}))

          # Stage B3 (workspace trim): baseline is the mutation job's own
          # pre-Stage-B3 cargo metadata capture (target-independent — workspace
          # membership doesn't vary by target); sibling is this job's own
          # mutated-tree metadata.
          expected_members = ["larql-cli", "larql-boundary", "larql-vindex-spec", "larql-models", "larql-compute"]
          baseline_metadata = json.loads(Path("mutation/baseline-metadata.json").read_text(encoding="utf-8"))
          sibling_metadata = json.loads(subprocess.run(
              ["cargo", "metadata", "--format-version", "1", "--no-deps"],
              capture_output=True, text=True,
          ).stdout)
          Path(f"baseline-b3-{target}.json").write_text(json.dumps({"metadata": baseline_metadata, "expected_members": expected_members}))
          Path(f"sibling-b3-{target}.json").write_text(json.dumps({"metadata": sibling_metadata, "expected_members": expected_members}))
          PYEOF

            for STAGE in stage-b stage-b2 stage-b3; do
              python3 scripts/target_analysis_promotion.py --stage "$STAGE" \
                --baseline-state-file "baseline-$STAGE-$TARGET.json" \
                --sibling-state-file "sibling-$STAGE-$TARGET.json" \
                > "out/promotion-$STAGE-$TARGET.json"
            done

            python3 - "$TARGET" <<'PYEOF'
          import json
          import sys
          sys.path.insert(0, ".")
          from pathlib import Path
          from scripts.target_analysis_common import error_sites
          from scripts.target_analysis_promotion import depth_advanced

          target = sys.argv[1]
          with open(f"out/stage-c-{target}.json") as f:
              sibling_messages = [json.loads(line) for line in f if line.strip()]
          baseline_messages = []  # first round: no prior-round Stage C output exists yet (Task 20 wires round-over-round)

          baseline_sites = error_sites(baseline_messages)
          sibling_sites = error_sites(sibling_messages)
          result = {
              "target": target,
              "depth_advanced": depth_advanced(baseline_sites, sibling_sites),
              "baseline_site_count": len(baseline_sites),
              "sibling_site_count": len(sibling_sites),
              "resolved_sites": sorted(str(s) for s in (baseline_sites - sibling_sites)),
              "new_sites": sorted(str(s) for s in (sibling_sites - baseline_sites)),
          }
          Path(f"out/depth-decision-{target}.json").write_text(json.dumps(result, indent=2))
          PYEOF
          done < targets-in-batch.txt
      - uses: actions/upload-artifact@v4
        with:
          name: promotion-decision-batch-${{ matrix.batch_index }}
          path: out/
```

`primary/dependency-graph-batch-*/unit-graph-{target}.json` uses a glob rather than embedding `${{ matrix.batch_index }}` inside the Python heredoc — the `pattern:` download step already scoped `primary/` to exactly this job's batch index, so there's exactly one matching directory and the glob avoids mixing GitHub Actions expression interpolation into a Python string literal.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add Stage C job, batched, applying the mutation patch before checking, with mechanically-grounded b2/b3 baselines"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: 10 `promotion-decision-batch-<N>` artifacts are produced (Task 12's real 119-target/10-batch universe, not the stale pre-Task-12 `ceil(331/12) = 28`), each containing per-target `stage-c-<target>.json`, `promotion-stage-b-<target>.json`, `promotion-stage-b2-<target>.json`, `promotion-stage-b3-<target>.json`, and `depth-decision-<target>.json` files. Specifically check the batch containing `nvptx64-nvidia-cuda`: confirm `stage-c-nvptx64-nvidia-cuda.json` shows real compiler output against the *mutated* tree (spot-check that the file content differs from what Task 8's unmutated `build-attempt` probe recorded for the same target — this is the direct evidence the patch was actually applied, not skipped), and confirm all three `promotion-stage-b*-nvptx64-nvidia-cuda.json` files show real `"promotes": true/false` verdicts (not an error) — this first round has no real prior-round Stage C baseline yet (`baseline_messages = []`), so `depth_advanced` should read `true` for every target with any error at all (every site is "newly resolved" relative to an empty baseline is wrong — re-check this reasoning empirically against the real output: an empty baseline means `baseline_sites - sibling_sites` is always empty regardless of `sibling_sites`, since you cannot subtract from nothing, so `depth_advanced` should read `false` for every target on this first round; Task 20 wires the real round-over-round baseline that makes this check meaningful).

**Real-CI addendum (2026-08-20):** run `32409988443` confirmed exactly this — 10/10 batches green, 10/10 `promotion-decision-batch-N` artifacts, 119/119 targets, `depth_advanced` false for all 119, `stage-b` verdicts real (nvptx: `true`), `stage-b2` verdicts real and pre-authorized-false for all targets (`serde_core` still pulls `std` under this nightly toolchain even after the patch). One real, unauthorized bug surfaced by this run: Stage B3's verdict is unconditionally `false` for all 119 targets due to a bug in `workspace_members_ok()` (Task 4, already-merged) assuming an obsolete cargo PackageId string format — routed to a new dedicated task (Task 18) rather than left silently wrong. Task 18's own real-data verification (its Step 8) surfaced a second, distinct, deeper bug in the same area: Stage B3's `expected_members`/`reachable` lists (Task 16 and Task 17, already-merged) hand-declare 5 crates as "larql-cli's real tree," but Cargo's own documented rule ("All path dependencies residing in the workspace directory automatically become members," confirmed via direct WebFetch of Cargo's reference docs) means the real, trimmed workspace has 14 members, not 5 — confirmed independently via a full transitive-closure grep of the real Cargo.toml files, and matching the original proven `experiment-cuda-nvptx.yml` job's own 13-crate `keep` set plus `larql-compute-metal` (an optional path dependency Cargo pulls in regardless of `optional = true`). Routed to a new dedicated task (Task 19) rather than left silently wrong.

---

### Task 18: Fix `workspace_members_ok`'s real cargo-metadata PackageId-format bug — Stage B3's promotion signal has been unconditionally False since Task 4

**Real bug in already-merged, already-tested code (Task 4), surfaced by Task 17's first
real exercise of this function against actual `cargo metadata` output** (run
`32409988443`, all 119 targets). `workspace_members_ok()` computes `actual_names =
{member.split(" ", 1)[0] for member in metadata["workspace_members"]}`, assuming the
older cargo PackageId string format `"name version (source)"`. Real, modern cargo
(confirmed: 1.97.1, this repo's own toolchain, via direct `cargo metadata
--format-version 1 --no-deps` in this worktree) produces a URL-style format instead —
`path+file:///abs/path/to/crate#version`, e.g.
`path+file:///.../crates/larql-models#0.2.0` — with no space anywhere in the string, so
`.split(" ", 1)[0]` returns the entire unsplit string, which can never equal a bare crate
name like `"larql-cli"`. `workspace_members_ok()` is therefore unconditionally `False`
regardless of whether a workspace was actually trimmed — confirmed uniformly `False`
across all 119 real targets in Task 17's real run, not target-specific. Task 4's own
fixtures (`cargo_metadata_full_workspace.json`, `cargo_metadata_trimmed_workspace.json`)
hand-author the old, unreal format directly, which is why this has been invisible to
Task 4's own tests since the day it was written — the fixtures never reflected reality.

The robust fix does not parse the PackageId string at all: real `cargo metadata`'s own
top-level `packages` array gives an exact `id -> name` mapping (confirmed directly:
`{"id": "path+file:///.../crates/larql-models#0.2.0", "name": "larql-models", ...}`).
Building an `id_to_name` lookup from `packages` and mapping `workspace_members` through
it is immune to any future PackageId string-format change, unlike a path-parsing
heuristic on the id string itself.

**Files:**
- Modify: `scripts/target_analysis_promotion.py` (`workspace_members_ok`)
- Modify: `tests/fixtures/target_analysis/cargo_metadata_full_workspace.json`, `tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json` (replace the hand-authored, unreal PackageId format with the real, modern format, including a `packages` array)
- Modify: `tests/test_target_analysis_promotion.py` (add one explicit regression test)

**Interfaces:**
- Consumes: nothing new — `workspace_members_ok(metadata, expected_members)`'s signature is unchanged; it already receives the full `metadata` dict, which already contains `packages` (Task 17's own job already produces this via plain `cargo metadata --format-version 1 --no-deps`, no wiring change needed anywhere that calls this function).
- Produces: the same `bool` return value, now computed correctly against real data.

- [ ] **Step 1: Confirm the real format directly (do not trust memory or the old fixtures)**

Run in this worktree:
```bash
cargo metadata --format-version 1 --no-deps | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('workspace_members[0]:', repr(data['workspace_members'][0]))
print('packages[0]:', {'id': data['packages'][0]['id'], 'name': data['packages'][0]['name']})
"
```
Expected: `workspace_members[0]` is a `path+file:///...#<version>` string with no space anywhere in it; `packages[0]` shows the same id paired with the crate's real, bare name. Confirm this before writing any fixture or test — it's the ground truth the fix and the fixtures must match.

- [ ] **Step 2: Update the fixtures to the real format (this alone should make the existing tests fail against the OLD implementation — confirm that before touching the implementation)**

Replace `tests/fixtures/target_analysis/cargo_metadata_full_workspace.json` with:
```json
{
  "packages": [
    {"id": "path+file:///repo/crates/larql-cli#0.1.0", "name": "larql-cli"},
    {"id": "path+file:///repo/crates/larql-boundary#0.1.0", "name": "larql-boundary"},
    {"id": "path+file:///repo/crates/larql-python#0.1.0", "name": "larql-python"},
    {"id": "path+file:///repo/crates/larql-vindex-spec#0.1.0", "name": "larql-vindex-spec"}
  ],
  "workspace_members": [
    "path+file:///repo/crates/larql-cli#0.1.0",
    "path+file:///repo/crates/larql-boundary#0.1.0",
    "path+file:///repo/crates/larql-python#0.1.0",
    "path+file:///repo/crates/larql-vindex-spec#0.1.0"
  ]
}
```

Replace `tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json` with:
```json
{
  "packages": [
    {"id": "path+file:///repo/crates/larql-cli#0.1.0", "name": "larql-cli"},
    {"id": "path+file:///repo/crates/larql-boundary#0.1.0", "name": "larql-boundary"},
    {"id": "path+file:///repo/crates/larql-vindex-spec#0.1.0", "name": "larql-vindex-spec"}
  ],
  "workspace_members": [
    "path+file:///repo/crates/larql-cli#0.1.0",
    "path+file:///repo/crates/larql-boundary#0.1.0",
    "path+file:///repo/crates/larql-vindex-spec#0.1.0"
  ]
}
```

- [ ] **Step 3: Run the existing tests, confirm they now fail (RED) against the unfixed implementation**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -k workspace_members -v`
Expected: FAIL — `test_workspace_members_ok_true_for_trimmed_workspace` should now fail (the old `.split(" ", 1)[0]` logic can no longer produce a matching name from the new, realistic fixture content), proving the fixtures previously masked the real bug rather than exercising it.

- [ ] **Step 4: Fix `workspace_members_ok`**

In `scripts/target_analysis_promotion.py`, replace:
```python
def workspace_members_ok(metadata: dict[str, Any], expected_members: list[str]) -> bool:
    actual_names = {
        member.split(" ", 1)[0] for member in metadata.get("workspace_members", [])
    }
    return actual_names == set(expected_members)
```
with:
```python
def workspace_members_ok(metadata: dict[str, Any], expected_members: list[str]) -> bool:
    id_to_name = {pkg["id"]: pkg["name"] for pkg in metadata.get("packages", [])}
    actual_names = {id_to_name[member] for member in metadata.get("workspace_members", [])}
    return actual_names == set(expected_members)
```
No defensive `.get()`/try-except around the `id_to_name[member]` lookup — every `workspace_members` entry is guaranteed (by cargo's own `--no-deps` contract) to have a matching `packages` entry; a `KeyError` here would mean genuinely malformed metadata, which should fail loudly, not be silently swallowed.

- [ ] **Step 5: Add one explicit, self-contained regression test**

In `tests/test_target_analysis_promotion.py`, add:
```python
def test_workspace_members_ok_parses_real_modern_package_id_format():
    # Real cargo (this repo's toolchain: 1.97.1) package-id format is
    # "path+file:///abs/path/to/crate#version" -- no space anywhere, unlike the
    # older "name version (source)" format the fixtures previously (wrongly)
    # assumed, which is why this bug was invisible to this test file since Task 4.
    metadata = {
        "packages": [
            {"id": "path+file:///repo/crates/larql-cli#0.2.0", "name": "larql-cli"},
            {"id": "path+file:///repo/crates/larql-boundary#0.2.0", "name": "larql-boundary"},
        ],
        "workspace_members": [
            "path+file:///repo/crates/larql-cli#0.2.0",
            "path+file:///repo/crates/larql-boundary#0.2.0",
        ],
    }
    assert workspace_members_ok(metadata, ["larql-cli", "larql-boundary"]) is True
```

- [ ] **Step 6: Run the full test suite, confirm GREEN**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: all tests pass, including the two existing `workspace_members_ok` tests (now against realistic fixtures) and the new regression test.

- [ ] **Step 7: Commit**

```bash
git add scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py tests/fixtures/target_analysis/cargo_metadata_full_workspace.json tests/fixtures/target_analysis/cargo_metadata_trimmed_workspace.json
git commit -m "fix: workspace_members_ok assumed the old cargo PackageId format; parse via packages[].id->name instead"
```

- [ ] **Step 8: Real-data verification against Task 17's own already-captured artifacts (no new CI trigger required)**

Download Task 17's real `secondary-mutation` artifact from run `32409988443` (the `baseline-metadata.json` it contains is real, unmutated-tree `cargo metadata` output, captured on a GitHub-hosted runner) and, in a fresh checkout of the same commit Task 17 tested against, apply `full-mutation.patch` and run `cargo metadata --format-version 1 --no-deps` yourself to get the real, mutated-tree (trimmed) metadata. Feed both real JSON blobs through the now-fixed `workspace_members_ok()` (or `stage_promotes("stage-b3", ...)` directly) and confirm it returns `True` for this real, already-known-to-be-trimmed data — this is the actual proof the fix works against reality, not just synthetic fixtures. Per the Task 11 → Task 12 precedent (a real bug in already-merged code, fixed and unit-tested, with real-CI re-verification deferred to the next task's natural pipeline run rather than a dedicated new trigger), this task's own real-CI push is likewise deferred — Task 19 (inserted immediately after this task, fixing a second, distinct bug in the same area found during this very step) and then Task 20 ("Recursive-round baseline handoff") will exercise this fix live for the first time, together. Commit this task's fix; it will be pushed together with whatever is already staged in this worktree ahead of the last validated run (including the controller's own prior `ef8637f7` commit), not as a separate, dedicated push solely for this fix.

---

### Task 19: Fix Stage B3's `expected_members`/`reachable` closure — the real trimmed workspace has 14 members, not 5

**Real, distinct, deeper bug found during Task 18's own real-data verification (its Step
8), independently confirmed twice by the controller** (a full transitive-dependency
grep of the real Cargo.toml files, and a direct WebFetch of Cargo's own reference
documentation). Task 18 fixed `workspace_members_ok()`'s PackageId parsing so it can
now correctly compare real cargo-metadata output against an `expected_members` list —
but the `expected_members`/`reachable` lists themselves (Task 16's Stage B3 mutation
step, Task 17's Stage C `expected_members` literal) hand-declare only 5 crates
(`larql-cli, larql-boundary, larql-vindex-spec, larql-models, larql-compute`) as
"larql-cli's real tree." Cargo's own documented rule (confirmed via WebFetch of
`https://doc.rust-lang.org/cargo/reference/workspaces.html`): **"All path dependencies
residing in the workspace directory automatically become members"** — the only
override is an explicit `exclude` entry, which Stage B3 never adds. So every one of
those 5 crates' own path dependencies gets silently pulled back in as a real workspace
member regardless of the trimmed `members`/`default-members` arrays' literal contents.

A full transitive-dependency closure over the real Cargo.toml files, starting from the
5 hand-declared crates, was independently confirmed by the controller (direct grep of
every crate's `Cargo.toml`, three rounds until the frontier of new path dependencies was
empty) and separately by Task 18's own real-data replay (downloading Task 17's real
`secondary-mutation` artifact from run `32409988443`, applying `full-mutation.patch` in
a fresh checkout, and running real `cargo metadata --format-version 1 --no-deps`): the
real, trimmed workspace has exactly **14 members**, not 5:

```
larql-boundary, larql-cli, larql-compute, larql-compute-metal, larql-core,
larql-execution, larql-factory, larql-inference, larql-kv, larql-lql, larql-models,
larql-router-protocol, larql-vindex, larql-vindex-spec
```

This exactly matches the original, proven `experiment-cuda-nvptx.yml` job's own
13-crate `keep` set (`larql-boundary, larql-cli, larql-compute, larql-core,
larql-execution, larql-factory, larql-inference, larql-kv, larql-lql, larql-models,
larql-router-protocol, larql-vindex, larql-vindex-spec` — read directly from that job's
own Stage-B3-equivalent step during Task 16's pre-flight review this session, but not
cross-checked against the 5-crate list at the time — a real miss, now corrected here)
plus `larql-compute-metal`, an *optional* path dependency of `larql-cli`
(`larql-compute-metal = { path = "../larql-compute-metal", optional = true }`) that
Cargo's implicit-membership rule pulls in as a real workspace member regardless of
`optional = true` (optionality only affects whether it's compiled as a *dependency*,
not whether it counts as a workspace *member*).

**This is not a defect in the trim itself — it is a defect in the check's literal.**
The 5 crates genuinely, correctly dropped by Stage B3's trim (`larql-demos`,
`larql-server`, `larql-router`, `larql-python`, `model-compute`) are exactly the
feature-unification polluters the original `experiment-cuda-nvptx.yml` job's own
comments named as needing removal (`larql-server`'s HTTP stack, `larql-python`'s pyo3
bindings pulling in default/std features transitively). Cargo's implicit-member
mechanic didn't defeat Stage B3's purpose; it mechanically computed `larql-cli`'s true
reachable dependency closure and correctly excluded everything genuinely outside it.
No prior CI evidence from Tasks 16 or 17 is invalidated by this finding — this is
Stage B3's *check* catching up to what its *trim* already correctly does.

**Ruling on remediation (of three options Task 18's implementer raised):** fix
`expected_members` to the real 14-name closure, sourced as a single new, shared,
importable Python constant (following the `MUTATED_LIBRARY_CRATES` pattern already
established in this file), not a fresh hand-typed guess and not a second hardcoded
YAML literal. Adding an `exclude` list to force the trim down to literally 5 members
was rejected — it would mutate reality to fit a wrong literal, defeating the mechanism
for no benefit (the polluters are already gone). Weakening the postcondition to
subset/containment semantics was also rejected — it would blunt the exact-witness
promotion definition Task 4's whole design is built on.

**Files:**
- Modify: `scripts/target_analysis_promotion.py` (add `STAGE_B3_REACHABLE_CLOSURE`)
- Modify: `tests/test_target_analysis_promotion.py` (add three regression tests)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (Stage B3's `reachable` list in the `secondary-mutate` job; `expected_members` in the `secondary-stage-c-and-promotion` job; one stale `# ... (Task 18 wires round-over-round)` comment left over from before this session's task-renumbering, corrected to `Task 20`)

**Interfaces:**
- Consumes: `workspace_members_ok(metadata, expected_members)` (Task 4, fixed by Task 18) — signature unchanged.
- Produces: `STAGE_B3_REACHABLE_CLOSURE: tuple[str, ...]` (14 real crate names), importable from `scripts.target_analysis_promotion`, consumed by both the mutation step and the Stage C promotion step in the live workflow, and available to Task 20 if its round-over-round bookkeeping needs the same closure.

- [ ] **Step 1: Reproduce the real, trimmed-workspace member list directly — do not trust this brief's list blindly**

Download Task 17's real `secondary-mutation` artifact from run `32409988443` (the same
one Task 18's own Step 8 already used). In a fresh checkout of the same commit Task 17
tested against, apply `mutation/full-mutation.patch`, then run:
```bash
cargo metadata --format-version 1 --no-deps | python3 -c "
import json, sys
data = json.load(sys.stdin)
id_to_name = {p['id']: p['name'] for p in data['packages']}
names = sorted(id_to_name[m] for m in data['workspace_members'])
print(len(names))
print(names)
"
```
Expected: `14` and exactly `['larql-boundary', 'larql-cli', 'larql-compute', 'larql-compute-metal', 'larql-core', 'larql-execution', 'larql-factory', 'larql-inference', 'larql-kv', 'larql-lql', 'larql-models', 'larql-router-protocol', 'larql-vindex', 'larql-vindex-spec']`. Confirm this before writing anything — it is the ground truth both the controller's own independent transitive-closure grep and Task 18's real-data replay already confirm.

- [ ] **Step 2: Write the failing tests first (TDD) — reference the not-yet-existing constant**

In `tests/test_target_analysis_promotion.py`, add this import to the existing import block at the top of the file:
```python
from scripts.target_analysis_promotion import (
    STAGE_B3_REACHABLE_CLOSURE,
    depth_advanced,
    no_std_scaffold_ok,
    serde_features_ok,
    stage_b_lib_rs_filenames,
    stage_promotes,
    workspace_members_ok,
)
```
(this replaces the existing `from scripts.target_analysis_promotion import (...)` block — just add `STAGE_B3_REACHABLE_CLOSURE,` alphabetically before `depth_advanced,`)

Then add these three tests:
```python
def test_stage_b3_reachable_closure_matches_real_transitive_dependency_tree():
    # Cargo's own documented rule ("All path dependencies residing in the
    # workspace directory automatically become members") means the real,
    # trimmed workspace has 14 members, not the 5 originally hand-declared
    # as "larql-cli's real tree" (Task 16, Task 17) -- confirmed against
    # real captured CI data from run 32409988443 (Task 17/Task 18) and an
    # independent transitive-closure grep of the real Cargo.toml files,
    # exactly matching the original, proven experiment-cuda-nvptx.yml job's
    # own 13-crate `keep` set plus larql-compute-metal (an optional path
    # dependency Cargo pulls in as a member regardless of `optional = true`).
    assert set(STAGE_B3_REACHABLE_CLOSURE) == {
        "larql-boundary", "larql-cli", "larql-compute", "larql-compute-metal",
        "larql-core", "larql-execution", "larql-factory", "larql-inference",
        "larql-kv", "larql-lql", "larql-models", "larql-router-protocol",
        "larql-vindex", "larql-vindex-spec",
    }
    assert len(STAGE_B3_REACHABLE_CLOSURE) == 14


def test_workspace_members_ok_true_for_real_trimmed_workspace_against_full_closure():
    metadata = {
        "packages": [
            {"id": f"path+file:///repo/crates/{name}#0.1.0", "name": name}
            for name in STAGE_B3_REACHABLE_CLOSURE
        ],
        "workspace_members": [
            f"path+file:///repo/crates/{name}#0.1.0" for name in STAGE_B3_REACHABLE_CLOSURE
        ],
    }
    assert workspace_members_ok(metadata, STAGE_B3_REACHABLE_CLOSURE) is True


def test_workspace_members_ok_false_for_real_trimmed_workspace_against_old_5_crate_list():
    # The bug this task fixes: the OLD hand-declared 5-crate expected list
    # can never match the real, trimmed workspace's actual 14 members --
    # this is exactly why Stage B3's promotion signal was doomed to read
    # False even after Task 18's PackageId-parsing fix.
    metadata = {
        "packages": [
            {"id": f"path+file:///repo/crates/{name}#0.1.0", "name": name}
            for name in STAGE_B3_REACHABLE_CLOSURE
        ],
        "workspace_members": [
            f"path+file:///repo/crates/{name}#0.1.0" for name in STAGE_B3_REACHABLE_CLOSURE
        ],
    }
    old_wrong_expected = ["larql-cli", "larql-boundary", "larql-vindex-spec", "larql-models", "larql-compute"]
    assert workspace_members_ok(metadata, old_wrong_expected) is False
```

- [ ] **Step 3: Run the tests, confirm they fail with a collection-time `ImportError` (RED)**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'STAGE_B3_REACHABLE_CLOSURE' from 'scripts.target_analysis_promotion'`. This confirms the test file is exercising code that doesn't exist yet, not a typo.

- [ ] **Step 4: Add `STAGE_B3_REACHABLE_CLOSURE` to `scripts/target_analysis_promotion.py`**

Add this immediately after the existing `STAGE_B_REPRESENTATIVE_CRATE` constant (before the `stage_b_lib_rs_filenames` function):
```python
# The real, transitive workspace-membership closure once Stage B3's trim runs --
# NOT the 5 crates originally hand-declared as "larql-cli's real tree" (Task 16,
# Task 17). Cargo's own documented rule ("All path dependencies residing in the
# workspace directory automatically become members," confirmed against
# https://doc.rust-lang.org/cargo/reference/workspaces.html) pulls path-dependency
# crates back in regardless of what Stage B3's Cargo.toml edit lists in
# `members`/`default-members`. Derived by a full transitive-dependency closure
# over the real Cargo.toml files starting from the 5 originally hand-declared
# crates (confirmed against real captured CI data from run 32409988443), and
# independently matching the original, proven experiment-cuda-nvptx.yml job's
# own 13-crate `keep` set plus larql-compute-metal (an optional path dependency
# of larql-cli that Cargo pulls in as a member regardless of `optional = true`).
STAGE_B3_REACHABLE_CLOSURE = (
    "larql-boundary",
    "larql-cli",
    "larql-compute",
    "larql-compute-metal",
    "larql-core",
    "larql-execution",
    "larql-factory",
    "larql-inference",
    "larql-kv",
    "larql-lql",
    "larql-models",
    "larql-router-protocol",
    "larql-vindex",
    "larql-vindex-spec",
)
```

- [ ] **Step 5: Run the tests, confirm GREEN**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: all tests pass, including the three new ones.

- [ ] **Step 6: Wire the real closure into the live workflow — Stage B3's mutation step**

In `.github/workflows/target-analysis-pipeline.yml`, in the `secondary-mutate` job's `"Stage B3: trim workspace members to larql-cli's real tree"` step, replace:
```python
          import re
          from pathlib import Path

          root_toml = Path("Cargo.toml")
          text = root_toml.read_text(encoding="utf-8")
          reachable = [
              "crates/larql-cli", "crates/larql-boundary", "crates/larql-vindex-spec",
              "crates/larql-models", "crates/larql-compute",
          ]
```
with:
```python
          import re
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_promotion import STAGE_B3_REACHABLE_CLOSURE

          root_toml = Path("Cargo.toml")
          text = root_toml.read_text(encoding="utf-8")
          reachable = [f"crates/{name}" for name in STAGE_B3_REACHABLE_CLOSURE]
```
The rest of the step (the `members_block_fmt` computation and the `members`/`default-members` substitution loop) is unchanged — only the `reachable` list's source changes, from a hardcoded 5-name literal to the real 14-name closure.

- [ ] **Step 7: Wire the real closure into the live workflow — Stage C's `expected_members`**

In the same file, in the `secondary-stage-c-and-promotion` job's per-target Python heredoc (the one that writes `baseline-stage-b3-$TARGET.json`/`sibling-stage-b3-$TARGET.json`), add the import alongside the heredoc's existing imports:
```python
          import json
          import subprocess
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_promotion import STAGE_B3_REACHABLE_CLOSURE

          target = sys.argv[1]
```
(this adds two lines — `sys.path.insert(0, ".")` and the `from scripts...` import — right after the existing `from pathlib import Path` line and before `target = sys.argv[1]`)

Then replace:
```python
          expected_members = ["larql-cli", "larql-boundary", "larql-vindex-spec", "larql-models", "larql-compute"]
```
with:
```python
          expected_members = list(STAGE_B3_REACHABLE_CLOSURE)
```

- [ ] **Step 8: Fix the stale task-number comment left over from this session's renumbering**

In the same file, in the `depth_advanced` heredoc, replace:
```python
          baseline_messages = []  # first round: no prior-round Stage C output exists yet (Task 18 wires round-over-round)
```
with:
```python
          baseline_messages = []  # first round: no prior-round Stage C output exists yet (Task 20 wires round-over-round)
```
(this task inserted ahead of the plan's former Task 19 "Recursive-round baseline handoff," renumbering it to Task 20 — this comment predates that renumbering)

- [ ] **Step 9: Commit**

```bash
git add scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: Stage B3's expected_members/reachable list hand-declared 5 crates; the real transitive closure is 14"
```

- [ ] **Step 10: Real-data verification against Task 17's own already-captured artifacts (no new CI trigger required)**

Using the same real `baseline-metadata.json` (from Task 17's `secondary-mutation` artifact, run `32409988443`) and the same real, mutated-tree `cargo metadata` output from Step 1 above, feed both through `stage_promotes("stage-b3", baseline_state, sibling_state)` with `expected_members = list(STAGE_B3_REACHABLE_CLOSURE)` in both states. Expected: `workspace_members_ok` reads `False` for the baseline (19 real unmutated members ≠ the 14-name closure) and `True` for the sibling (14 real trimmed members == the 14-name closure exactly), so `stage_promotes("stage-b3", ...)` reads `True` — the first genuine, real `False → True` stage promotion this pipeline has produced. This is the actual proof the fix works against reality, not just synthetic fixtures. Per the Task 11 → Task 12 and Task 17 → Task 18 precedent, this task's own real-CI push is deferred to Task 20's next natural pipeline run — commit this task's fix now; it will be pushed together with the controller's own already-staged, unpushed `ef8637f7` and Task 18's `3e5ad217` commits, not as a separate, dedicated push solely for this fix.

---

### Task 20: Recursive-round baseline handoff

**Real gap this task closes, caught before dispatch (pre-flight review, same
standard as Tasks 15-17's brief corrections).** This task's original text
only ever *produced* a `round-baseline` artifact — it never wired anything
to *consume* a prior round's `round-baseline` back in as the next round's
actual Stage C baseline. `secondary-stage-c-and-promotion` (Task 17,
already merged) hardcodes `baseline_messages = []` unconditionally, with a
comment reading "first round: no prior-round Stage C output exists yet."
Nothing in the original Task 20 text ever made that comment stop being
true for round 2 onward. As specced, the recursive loop this whole
Secondary layer exists to run — "observation and mutation inform each
other round over round" — never actually closes: `round-baseline` would be
produced every run and read by nothing, forever. This revision adds the
missing read-back half.

**Design, confirmed against real GitHub documentation before being written
here** (same standard this plan already held itself to for the
`background:`/`wait-all:` correction and Cargo's workspace-membership
rule): `actions/download-artifact@v4` supports fetching an artifact from a
*different* workflow run via `run-id:` + `github-token:` inputs (confirmed
directly against the action's own v4 README — "The id of the workflow run
where the desired download artifact was uploaded from... If github-token is
specified, this is the run that artifacts will be downloaded from"). Finding
*which* prior run to fetch from uses GitHub's REST API `GET
/repos/{owner}/{repo}/actions/artifacts` endpoint (confirmed directly
against GitHub's own REST API docs), which supports a `name` query filter
and returns each artifact's `expired` flag and its `workflow_run` object
(`id`, `head_branch`). The API's own docs do not specify a sort order, so
the candidate list is sorted client-side by `created_at`, newest first —
never assumed to arrive pre-sorted.

**Placement: the exotic cross-run mechanism lives in exactly one small,
dedicated job**, not spread across `secondary-mutate` or all 10
`secondary-stage-c-and-promotion` batch jobs. That job fetches once and
republishes what it found as an ordinary same-run artifact; every other
job downloads that same-run artifact the normal way, with no cross-run
awareness of its own.

**No silent fallback.** A first-ever run (or a run on a branch with no
prior `round-baseline`) is a real, expected case, not an error — but it
must be visible, not silently indistinguishable from "compared against a
real prior round and found nothing new." The fetch job's own output
records whether it found a prior run and, if so, which run ID; that
provenance is written into the republished artifact so every downstream
consumer — and any human reading `round-baseline.json` later — can tell
which case occurred.

**Payload: carry full per-target site sets, recomputed from the real
`stage-c-<target>.json` compiler output already present in each batch's
own `out/` directory** (confirmed: `secondary-stage-c-and-promotion`'s
final `upload-artifact` step uploads `path: out/` in full, and
`stage-c-$TARGET.json` is written into `out/` earlier in that same job —
Task 17, already merged). The existing `depth-decision-<target>.json`
records (`resolved_sites`, `new_sites`) are *differences*, not the full
site set, and are serialized as `sorted(str(s) for s in ...)` —
stringified tuples like `"('crates/foo/src/lib.rs', 12, 'E0433')"` — which
would require a fragile `eval`/regex to parse back into a real tuple. The
fold step instead recomputes each target's actual `sibling_sites` directly
from that target's own `stage-c-<target>.json`, via the same `error_sites()`
call `secondary-stage-c-and-promotion`'s own depth-advancement step
already uses, and stores it as a real JSON list of `[file, line, code]`
triples — the next round restores it as a set of tuples with
`{tuple(s) for s in ...}`, never by parsing a stringified repr. Skipping
this tuple-restore step would silently produce empty-looking set
intersections downstream with no local test to catch it, since no test in
this repo exercises the live workflow's own heredocs end-to-end.

**Scope note: "fold promoted diffs" means verdict records (JSON), never
source patches.** This task carries forward `depth-decision`/promotion
JSON and recomputed site sets — never a `git diff`/patch of the mutated
source tree. Committing a patch across runs would violate this plan's
explicit no-CI-commits, no-caching-of-mutated-source Global Constraint;
each round's mutation is re-derived fresh from Stage A/B/B2/B3 every run,
only the *verdicts* about it persist round-to-round.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` — add a new `fetch-prior-round-baseline` job; add a new `next-round-baseline` job (this task's original deliverable, now enriched with real per-target site payloads); modify the already-merged `secondary-stage-c-and-promotion` job's `depth_advanced` heredoc to read the fetched baseline back in, replacing the `baseline_messages = []` hardcode.

**Interfaces:**
- Consumes: `promotion-decision-batch-<batch_index>` artifacts (Task 17), each containing per-target `depth-decision-<target>.json`, `promotion-stage-*-<target>.json`, and `stage-c-<target>.json` files (the last of these newly consumed by this task's fold step, though it was already being uploaded).
- Produces: `round-baseline` artifact (this run's own fold, for the *next* run to find) and `prior-round-baseline` artifact (this run's own same-run republish of whatever the fetch job found, consumed by `secondary-stage-c-and-promotion` within this same run).

- [ ] **Step 1: Add the `fetch-prior-round-baseline` job**

```yaml
  fetch-prior-round-baseline:
    name: Fetch prior round's baseline (cross-run artifact lookup)
    needs: [discovery]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    permissions:
      contents: read   # actions/checkout
      actions: read    # gh api /actions/artifacts, cross-run actions/download-artifact
    steps:
      - uses: actions/checkout@v4
      - name: Find the most recent prior run's round-baseline artifact on this branch, if any
        id: find
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          python3 - <<'PYEOF'
          import json
          import os
          import subprocess

          repo = os.environ["GITHUB_REPOSITORY"]
          branch = os.environ["GITHUB_REF_NAME"]
          current_run_id = os.environ["GITHUB_RUN_ID"]

          raw = subprocess.run(
              ["gh", "api", f"/repos/{repo}/actions/artifacts", "--paginate",
               "-f", "name=round-baseline"],
              capture_output=True, text=True, check=True,
          ).stdout
          # --paginate with multiple pages emits one JSON object per page;
          # each page's "artifacts" list is concatenated here.
          artifacts = []
          for line in raw.splitlines():
              if line.strip():
                  artifacts.extend(json.loads(line)["artifacts"])

          candidates = [
              a for a in artifacts
              if not a["expired"]
              and a.get("workflow_run") is not None
              and a["workflow_run"]["head_branch"] == branch
              and str(a["workflow_run"]["id"]) != str(current_run_id)
          ]
          candidates.sort(key=lambda a: a["created_at"], reverse=True)

          with open(os.environ["GITHUB_OUTPUT"], "a") as f:
              if candidates:
                  f.write("found=true\n")
                  f.write(f"prior_run_id={candidates[0]['workflow_run']['id']}\n")
              else:
                  f.write("found=false\n")
                  f.write("prior_run_id=\n")
          PYEOF
      - name: Download the prior run's round-baseline artifact, if one was found
        if: steps.find.outputs.found == 'true'
        uses: actions/download-artifact@v4
        with:
          name: round-baseline
          run-id: ${{ steps.find.outputs.prior_run_id }}
          github-token: ${{ github.token }}
          path: prior-round-baseline-download
      - name: Republish as a same-run artifact, self-describing what was found
        run: |
          mkdir -p out
          if [ "${{ steps.find.outputs.found }}" = "true" ]; then
            cp prior-round-baseline-download/round-baseline.json out/round-baseline.json
            printf '{"prior_run_id": %s}' "${{ steps.find.outputs.prior_run_id }}" > out/provenance.json
          else
            printf '{"promoted": {}, "preserved_not_promoted": {}}' > out/round-baseline.json
            printf '{"prior_run_id": null}' > out/provenance.json
          fi
      - uses: actions/upload-artifact@v4
        with:
          name: prior-round-baseline
          path: out/
```

`--paginate`'s exact output framing (one JSON object per page vs. one
combined array) is a real `gh` CLI behavior to confirm against the first
real run's step output, not assumed — if it emits a single combined JSON
array instead of one object per page, the parsing loop above needs
`json.loads(raw)["artifacts"]` directly instead. Note this explicitly when
verifying Step 4 below.

- [ ] **Step 2: Add the `next-round-baseline` job, enriched with real per-target site payloads**

```yaml
  next-round-baseline:
    name: Fold promoted diffs into next round's baseline
    needs: [discovery, secondary-stage-c-and-promotion]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    permissions:
      contents: read   # actions/checkout
      actions: read    # actions/download-artifact
    steps:
      - uses: actions/checkout@v4
      - name: Download all promotion-decision batch artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: "promotion-decision-batch-*"
          path: promotion-decisions
      - name: Fold promoted results, preserve non-promoted results separately, carry real site sets forward
        run: |
          python3 - <<'PYEOF'
          import json
          import re
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_common import error_sites

          promoted = {}
          preserved = {}
          for batch_dir in Path("promotion-decisions").iterdir():
              if not batch_dir.is_dir():
                  continue
              for depth_file in batch_dir.glob("depth-decision-*.json"):
                  target = re.sub(r"^depth-decision-|\.json$", "", depth_file.name)
                  depth_decision = json.loads(depth_file.read_text())
                  stage_verdicts = {}
                  for stage in ("stage-b", "stage-b2", "stage-b3"):
                      stage_file = batch_dir / f"promotion-{stage}-{target}.json"
                      if stage_file.exists():
                          stage_verdicts[stage] = json.loads(stage_file.read_text())

                  stage_c_file = batch_dir / f"stage-c-{target}.json"
                  sibling_messages = [
                      json.loads(line) for line in stage_c_file.read_text().splitlines() if line.strip()
                  ]
                  sibling_sites = sorted(list(s) for s in error_sites(sibling_messages))

                  record = {**depth_decision, "stage_promotions": stage_verdicts, "sibling_sites": sibling_sites}
                  if depth_decision["depth_advanced"]:
                      promoted[target] = record
                  else:
                      preserved[target] = record

          Path("round-baseline.json").write_text(
              json.dumps({"promoted": promoted, "preserved_not_promoted": preserved}, indent=2)
          )
          PYEOF
      - uses: actions/upload-artifact@v4
        with:
          name: round-baseline
          path: round-baseline.json
```

`sibling_sites` is carried forward for every target regardless of
promotion status — a target that didn't promote this round still has a
real error wall the next round needs as its comparison baseline; only
`depth_advanced`'s own true/false split (not this payload) distinguishes
promoted from preserved. Per the spec's Data flow open question, cross-run
artifact retention beyond GitHub's default 90-day expiry is explicitly not
resolved by this plan — sufficient for the recursive loop's own mechanics
without deciding the longer-term history question.

- [ ] **Step 3: Wire the read-back into `secondary-stage-c-and-promotion` (already-merged, Task 17)**

Add `fetch-prior-round-baseline` to this job's `needs:` list (currently
`needs: [discovery, indexing, secondary-mutate, dependency-graph]` — add
`fetch-prior-round-baseline` to that list) and add a download step
alongside its existing `dependency-graph-batch-<N>` download:

```yaml
      - name: Download this run's fetched prior-round baseline
        uses: actions/download-artifact@v4
        with:
          name: prior-round-baseline
          path: prior-round-baseline
```

Then, in the per-target `depth_advanced` heredoc, replace:
```python
          baseline_messages = []  # first round: no prior-round Stage C output exists yet (Task 20 wires round-over-round)

          baseline_sites = error_sites(baseline_messages)
          sibling_sites = error_sites(sibling_messages)
```
with:
```python
          prior_round = json.loads(Path("prior-round-baseline/round-baseline.json").read_text())
          prior_record = (
              prior_round.get("promoted", {}).get(target)
              or prior_round.get("preserved_not_promoted", {}).get(target)
          )
          baseline_sites = (
              {tuple(s) for s in prior_record["sibling_sites"]} if prior_record is not None else set()
          )
          sibling_sites = error_sites(sibling_messages)
```
`prior_record is None` covers both real, expected cases without treating
either as an error: a genuine first-ever run (no prior round-baseline
existed at all, per Step 1's own explicit fallback) and a target newly
appearing in this round's matrix that had no corresponding entry in the
prior round's fold. Both correctly start that target's `baseline_sites` at
the empty set, matching the pre-existing "first round" semantics exactly —
only now real for round 2+ too, instead of hardcoded forever.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: wire real cross-run baseline handoff -- fetch, fold, and read back prior round's Stage C sites"
```

- [ ] **Step 5: Push and verify on a real runner — two runs, not one**

This task's own real-CI verification requires **two separate pushes**, not
one, and both must be checked before this task is considered proven — a
single green run only exercises the fallback half:

*Run A (this task's own push, combined with the already-staged, unpushed
`ef8637f7`, `3e5ad217`, and `166851a9` commits per the Task 11 → 12 and
Task 18/19 precedent):* confirm `fetch-prior-round-baseline` reports
`found=false` (no `round-baseline` artifact has ever existed on this
branch — this job and `next-round-baseline` are both new in this same
push), confirm `secondary-stage-c-and-promotion` still runs correctly with
`baseline_sites = set()` for every target (identical observable behavior
to before this task, since `prior_record` is `None` for everyone), and
confirm `round-baseline` and `prior-round-baseline` artifacts are both
produced with the enriched `sibling_sites` payload present for every
target. This run proves the fallback path and first-ever production of a
real, enriched `round-baseline` — it does **not** prove the read-back path
works, since nothing exists yet for it to read.

*Run B (the next natural push after Run A — expected to be supplied by
Task 21's own work, not a separate dedicated trigger for this task alone):*
confirm `fetch-prior-round-baseline` reports `found=true` with a real
`prior_run_id` matching Run A's actual run ID, confirm at least one
target's `baseline_sites` in `secondary-stage-c-and-promotion` is
genuinely non-empty (reconstructed from Run A's real `sibling_sites`, not
`set()`), and confirm `depth_advanced` for that target is computed against
that real, non-trivial baseline rather than vacuously reading `true` for
every non-empty `sibling_sites` (the empty-baseline case Task 17 already
flagged this exact failure mode for). **Do not declare this task's loop
"closed" — in the ledger or otherwise — until Run B's read-back path is
confirmed on real CI; Run A alone only proves the parts of this task that
would have been trivially true even without it.**

**Known, already-flagged interaction, not new scope:** cross-round site
comparison inherits the unpinned-nightly toolchain drift this plan already
flagged as a follow-up (each job independently runs `rustup toolchain
install nightly`, which can resolve to a different nightly build across
runs) — a spurious site appearing or disappearing between Run A and Run B
could be toolchain drift, not real mutation progress, until that follow-up
is addressed. Noted here as a real caveat on how to read Run B's result,
not a blocker for this task or a scope change to it.

---

### Task 21: Secondary-layer test suite — noise floor, blast-radius containment, ephemerality

**Three real defects found and fixed in this task's own original text before dispatch**
(the same pre-flight standard already applied to Tasks 15-17 and 20):

1. **The ephemerality check is self-triggering — confirmed by direct simulation, not
   assumed.** The original check greps the whole workflow file for the literal
   substring `git (commit|push)`. But the check's own step name
   (`"Ephemerality: assert no git commit/push exists anywhere in this workflow
   file"`) and its own error message (`"Found a git commit/push in the pipeline
   workflow"`) both *contain that exact substring* — the moment this step is added
   to the file, the grep matches the step's own descriptive text and the check
   fails unconditionally, on every run, regardless of whether any real git-mutating
   command exists anywhere. Confirmed directly: extracting the planned step's own
   YAML text and running the planned grep against it reproduces the false positive.
   Fixed below by (a) rewording the step's own name/messages to not contain the
   literal adjacent phrase, and (b) filtering out `echo`/`::error::`/comment lines
   before searching, so future additions to this file don't reintroduce the same
   self-match.
2. **Blast-radius containment tests a hand-duplicated copy of Stage B's real logic,
   not Stage B itself, and uses a different crate than the pipeline actually uses.**
   The original check re-implements the scaffold-insertion algorithm inline, a
   second, independent copy of the exact code already in the `secondary-mutate`
   job's real "Stage B: inject `#![no_std]` scaffold" step (Task 16, already
   merged) — any future change to the real algorithm could silently desync from
   this copy, so the test would keep passing while testing something Stage B no
   longer actually does. It also targets `larql-boundary`, not
   `STAGE_B_REPRESENTATIVE_CRATE` (`larql-compute`, Task 14's own deliberate choice
   as the hardest real case) — an arbitrary, inconsistent choice. Fixed below by
   extracting the algorithm into a single, shared, importable function
   (`insert_no_std_scaffold`) that both the real Stage B step and this self-test
   call — eliminating the duplication and letting this test genuinely exercise the
   real mechanism, not a twin of it — and by using `larql-compute` throughout.
3. Golden-fixture crates and cross-target/native comparison remain **explicitly out
   of scope for this task** (already flagged as separate, not-yet-implemented
   follow-ups in the Self-Review below) — not a defect in this task's own text,
   noted here only so it isn't mistaken for one.

**Files:**
- Modify: `scripts/target_analysis_promotion.py` (extract `insert_no_std_scaffold`)
- Modify: `tests/test_target_analysis_promotion.py` (regression test for the extracted function)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (Stage B step now calls the shared function instead of its own inline copy; add `secondary-layer-self-test` job)

**Interfaces:**
- Consumes: nothing from earlier jobs — this job validates the Secondary-layer mechanism itself, independent of any specific crate/target result, per the spec's Testing section.
- Produces: `insert_no_std_scaffold(text: str) -> str`, importable from `scripts.target_analysis_promotion`, consumed by both the real Stage B step and this self-test.

- [ ] **Step 1: Write a failing test for the not-yet-extracted function (TDD)**

Add this import to the top of `tests/test_target_analysis_promotion.py` (alphabetically, before `no_std_scaffold_ok`):
```python
from scripts.target_analysis_promotion import (
    STAGE_B3_REACHABLE_CLOSURE,
    depth_advanced,
    insert_no_std_scaffold,
    no_std_scaffold_ok,
    serde_features_ok,
    stage_b_lib_rs_filenames,
    stage_promotes,
    workspace_members_ok,
)
```
Then add:
```python
def test_insert_no_std_scaffold_inserts_after_leading_doc_comment_and_attributes():
    text = "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\npub fn f() {}\n"
    result = insert_no_std_scaffold(text)
    assert result == (
        "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\n"
        "#![no_std]\nextern crate alloc;\n"
        "pub fn f() {}\n"
    )
    assert no_std_scaffold_ok(result) is True


def test_insert_no_std_scaffold_inserts_at_top_when_no_leading_doc_comment():
    text = "pub fn f() {}\n"
    result = insert_no_std_scaffold(text)
    assert result == "#![no_std]\nextern crate alloc;\npub fn f() {}\n"
```

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'insert_no_std_scaffold'`.

- [ ] **Step 2: Extract `insert_no_std_scaffold` into `scripts/target_analysis_promotion.py`**

Add this immediately after `no_std_scaffold_ok`:
```python
def insert_no_std_scaffold(text: str) -> str:
    lines = text.splitlines(keepends=True)

    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
            insert_at = i + 1
        else:
            break

    scaffold = "#![no_std]\nextern crate alloc;\n"
    lines.insert(insert_at, scaffold)
    return "".join(lines)
```
This is a byte-for-byte extraction of the real algorithm already proven in this
session's real CI runs (Task 16) — the insertion-position loop and the scaffold
string are copied verbatim, not rewritten.

- [ ] **Step 3: Run the tests, confirm GREEN**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: all tests pass, including the two new ones.

- [ ] **Step 4: Point the real Stage B step at the shared function**

In `.github/workflows/target-analysis-pipeline.yml`, in the `secondary-mutate`
job's `"Stage B: inject #![no_std] scaffold, per crate"` step, replace:
```python
          import sys
          from pathlib import Path

          crate = sys.argv[1]
          lib_rs = Path(f"crates/{crate}/src/lib.rs")
          text = lib_rs.read_text(encoding="utf-8")
          lines = text.splitlines(keepends=True)

          insert_at = 0
          for i, line in enumerate(lines):
              stripped = line.strip()
              if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
                  insert_at = i + 1
              else:
                  break

          scaffold = "#![no_std]\nextern crate alloc;\n"
          lines.insert(insert_at, scaffold)
          lib_rs.write_text("".join(lines), encoding="utf-8")
```
with:
```python
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_promotion import insert_no_std_scaffold

          crate = sys.argv[1]
          lib_rs = Path(f"crates/{crate}/src/lib.rs")
          lib_rs.write_text(
              insert_no_std_scaffold(lib_rs.read_text(encoding="utf-8")),
              encoding="utf-8",
          )
```
This is a pure extraction — same algorithm, same output, now shared instead of
duplicated. Real-CI verification in Step 8 below confirms this produces
byte-identical `sibling-lib-rs-*.txt` output to before this change, not just
"looks equivalent."

- [ ] **Step 5: Add the self-test job**

```yaml
  secondary-layer-self-test:
    name: Secondary-layer self-test (noise floor, blast radius, ephemerality)
    needs: [discovery]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    permissions:
      contents: read   # actions/checkout
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: "Noise floor: run unmutated Stage C twice, confirm stable findings"
        run: |
          for i in 1 2; do
            cargo +nightly check -p larql-cli --target nvptx64-nvidia-cuda \
              --message-format=json --keep-going > noise-floor-run-$i.json || true
          done
          python3 - <<'PYEOF'
          import json
          import sys
          sys.path.insert(0, ".")
          from scripts.target_analysis_common import error_sites

          def load_messages(path):
              with open(path) as f:
                  return [json.loads(line) for line in f if line.strip()]

          run1 = error_sites(load_messages("noise-floor-run-1.json"))
          run2 = error_sites(load_messages("noise-floor-run-2.json"))
          if run1 != run2:
              print(f"::error::Noise floor violated: run1={run1} run2={run2}")
              sys.exit(1)
          print(f"Noise floor stable: {len(run1)} identical error sites across both runs.")
          PYEOF
      - name: "Blast-radius containment: assert Stage B only touches its declared scope"
        run: |
          python3 - <<'PYEOF'
          import sys
          sys.path.insert(0, ".")
          from pathlib import Path
          from scripts.target_analysis_promotion import insert_no_std_scaffold, STAGE_B_REPRESENTATIVE_CRATE

          lib_rs = Path(f"crates/{STAGE_B_REPRESENTATIVE_CRATE}/src/lib.rs")
          lib_rs.write_text(insert_no_std_scaffold(lib_rs.read_text(encoding="utf-8")), encoding="utf-8")
          PYEOF
          CHANGED=$(git diff --name-only)
          if [ "$CHANGED" != "crates/larql-compute/src/lib.rs" ]; then
            echo "::error::Stage B touched files outside its declared scope: $CHANGED"
            exit 1
          fi
          echo "Blast radius contained to the declared file."
      - name: "Ephemerality: assert no forbidden git-mutating command exists in this workflow file"
        run: |
          if grep -v -E '(echo |::error::|::warning::|^\s*#)' .github/workflows/target-analysis-pipeline.yml | grep -Eq 'git (commit|push)'; then
            echo "::error::Found a forbidden git-mutating command in the pipeline workflow — mutations must stay ephemeral to the job checkout."
            exit 1
          fi
          echo "No forbidden git-mutating command found in the pipeline workflow."
```
This step's own name and echoed messages deliberately avoid the literal adjacent
phrase `git commit`/`git push` (confirmed by direct simulation before this text was
written: the ORIGINAL wording self-triggered the check the moment it was added to
the file) — write any future edit to this step the same way, or re-run the same
simulation before changing its wording.

- [ ] **Step 6: Run the full local test suite, confirm GREEN**

Run: `python3 -m pytest -v`
Expected: all tests pass, including the two new ones from Step 1.

- [ ] **Step 7: Commit**

```bash
git add scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add Secondary-layer self-test job; extract shared insert_no_std_scaffold to stop duplicating Stage B's real logic"
```

- [ ] **Step 8: Push and verify on a real runner**

Push and confirm: all three self-test checks pass on the real pipeline. Specifically
for the blast-radius check: confirm it now runs against `larql-compute`, and confirm
`secondary-mutate`'s own real Stage B step (now calling the shared function) still
produces the exact same `sibling-lib-rs-larql-compute.txt` content as Task 17's last
real run (run `32418628575`, or whichever is most recent) — a byte-for-byte diff
against that prior artifact's content, not merely "the job stayed green," is the
actual proof this was a safe, behavior-preserving extraction. Then deliberately
introduce a Stage B bug that touches an extra file (e.g., also modify `Cargo.toml`
inside the blast-radius test's own inline script) in a throwaway commit, confirm the
blast-radius check fails loudly, then revert that throwaway commit.

**This push is also Run B for Task 20's recursive-round baseline handoff** — the
first real chance to prove the read-back path Task 20 could only fall back on.
Confirm, as part of this same push's verification: `fetch-prior-round-baseline`
reports `found=true` with `prior_run_id` matching Run A's real run ID (`32418628575`
unless a further push has landed first), at least one target's `secondary-stage-c-
and-promotion` computes a genuinely non-empty `baseline_sites` (reconstructed from
Run A's real `sibling_sites`, not `set()`), and `depth_advanced` for that target is
computed against that real, non-trivial baseline. Do not close out Task 20's own
outstanding concern in the ledger until this is confirmed — record the result against
Task 20's entry as well as this task's own.

---

## Self-Review

**Spec coverage** — every spec section maps to a task:
- Foundational framework / Standing design principles → Global Constraints (verbatim-sourced).
- Discovery job → Task 5, extended by Task 6 (batching — a real GitHub Actions platform limit discovered empirically on Task 5's own real run, per Standing Principle 1).
- Target-capability probes → Task 7.
- Dependency-graph probes → Task 8 (dependency-graph half).
- Build-attempt probes (including the `only-cdylib` crate-type correction, and the `--crate-type`/`rust-src` fixes caught before dispatch) → Task 8 (build-attempt half).
- Runtime-test probes → Task 9 wires the job and its artifact contract, but the job never executes an actual test on any target (including host targets, where it would cost nothing extra) — it only emits a hardcoded `{"status": "runnable", "runner": ...}` verdict per target, which is curated L2 judgment, not a real measurement. This was not caught as a gap until the final whole-branch review; flagged here so it isn't miscounted as done, and added to the follow-up list below alongside the other honestly-flagged gaps.
- Indexing (structural extraction, contradiction rule, completeness enforcement, now at file-level across batched artifacts) → Tasks 3 and 10. The contradiction rule's real field-path bug (`unexpected_clean_std_build` checked `target_spec["std"]` instead of the real `target_spec["metadata"]["std"]`, inert against real data since Task 3, surfaced by Task 10's own real-CI verification) → Task 11.
- Discovery scope (tier 1+2 only, 119 of 331 real targets — a user-directed, real-evidence-driven correction to the original "every rustc target" design, mechanically grounded in `target-spec-json`'s own `metadata.tier` field, not agent judgment) → Task 12.
- Concurrency (a workflow-level group with `cancel-in-progress`, motivated by ~40 real minutes lost to queue contention on Task 12's own run) and least-privilege permissions on every job (user directive, now a standing Global Constraint, not a one-time fix) → Task 13.
- Target-independent-before-target-dependent sequencing (user directive, generalizing the Task 16 mutation-job restructure's own finding into a standing rule) → Task 14 (`cargo fmt --check`, the first concrete case).
- Build-attempt `build_std` axis (user-directed correction to already-merged Task 8 work: of the four modes, only `none` has real, unconditional justification in the Primary layer — `std` was never justified either way, `core`/`core,alloc` are only meaningful post-mutation, which is exactly where Task 17's Stage C already runs them) → Task 15.
- Secondary-layer mutation stages A/B/B2/B3 (with `background`/`wait` concurrency, target-independent, single job) → Task 16. Stage C (batched, applying the mutation patch — the critical bug this restructure fixes over the original per-target-matrixed draft) → Task 17.
- The measurable-difference / promotion rule (this session's immediate deliverable) → Task 4 (script) and Task 17 (wiring, with mechanically-grounded b2/b3 baselines sourced from the Primary layer's own artifacts rather than fabricated).
- Recursive round-over-round baseline handoff → Task 20. A real bug in the promotion rule's own already-merged code (`workspace_members_ok()` assumed an obsolete cargo PackageId string format, making Stage B3's verdict unconditionally `False` since Task 4 — surfaced by Task 17's first real exercise against actual `cargo metadata` output) → Task 18. A second, distinct, deeper bug in the same area — Stage B3's `expected_members`/`reachable` lists hand-declare 5 crates, but Cargo's own documented implicit-workspace-membership rule means the real, trimmed tree has 14 members — surfaced by Task 18's own real-data verification → Task 19.
- **A real, load-bearing limit of the recursive loop as built, found by the final whole-branch review, not by any single task's own reviewer:** the loop verifiably *closes* round-to-round (Task 20/21's real-CI evidence: a real prior round's baseline is fetched, and `baseline_site_count` matches the prior round's real `sibling_sites` length for 119/119 targets) — but it can only ever *detect drift*, never *mechanically advance depth by its own action*. Stages A/B/B2/B3 (`secondary-mutate`) are fixed and unparameterized by round; `secondary-mutate` has no `needs:` on `fetch-prior-round-baseline` and reads no baseline. So round N+1 applies a byte-identical mutation to a byte-identical checkout as round N, and `depth_advanced` can only ever fire from toolchain drift, dependency drift, or a human source change between rounds — never from the recursive loop's own mutation. Task 20's ruling that "fold promoted diffs" means verdict records, not source patches, is correct and not being revisited here — but this is the actual, previously-unstated consequence of that ruling, and a reader of this plan should know it. Advancing depth mechanically, round over round, requires round-parameterized mutation stages — this is exactly the "waterfall"/subgraph-branching redesign the user raised earlier in this plan's own history and which remains an open, parked architectural question, not yet decided.
- Error handling (honest-result pattern, retries narrow to network calls, platform-limit category) → reused directly from the proven `experiment-cuda-nvptx.yml` patterns in Tasks 8 and 17; the retry/platform-limit categories are not separately re-implemented since Tasks 7-9's probes don't call rate-limited external APIs beyond `rustc`/`cargo` — Discovery's crates.io/GitHub SBOM calls (mentioned in the spec's Discovery job description) are the one place a narrow retry would apply and are flagged here as **not yet implemented**: Task 5 only wires `rustc --print target-list`, not the crates.io/SBOM ecosystem-discovery calls. This is a real gap — added as a follow-up task below rather than silently left out.
- Testing (noise floor, blast-radius, golden fixtures, ephemerality, cross-target/native comparison) → Task 21 covers ephemerality fully; noise floor and blast-radius only partially. The noise-floor check proves within-job, same-toolchain-install determinism (two `cargo check` runs back to back) — it does NOT observe the actual noise source flagged elsewhere in this section (cross-run/cross-toolchain-install drift), since both its runs share one `rustup toolchain install nightly`. Blast-radius containment covers Stage B only; Stages A, B2, and B3 have no containment check, and Stage A additionally has no promotion postcondition of any kind (`STAGE_POSTCONDITIONS` only covers stage-b/b2/b3). Golden fixtures and cross-target/native comparison remain **not yet implemented** — flagged below, alongside the narrower noise-floor/blast-radius gaps just described.
- Explicitly not doing (no caching, no CI commits, no agent curation presented as L1) → Global Constraints + Task 21's ephemerality check enforces the no-commits rule structurally.

**Follow-up tasks not included in this plan** (real gaps, not placeholders — each needs its own task the way Tasks 1-21 are written, deferred here because this plan's immediate trigger was the promotion-rule definition, not full spec closure):
- Discovery job's crates.io/GitHub SBOM ecosystem-discovery calls, with narrow bounded retry on those specific network calls (spec: Components/Discovery job, Error handling/Retries).
- Golden-fixture crates with a planted, known-in-advance outcome, generalizing `serde-nostd-probe` (spec: Testing).
- Cross-target/cross-native comparison job, once the target matrix includes both nvptx and at least one native target's Stage C result for the same underlying finding (spec: Testing) — this is naturally sequenced after Tasks 1-21 produce enough real round data to compare, not before.
- The target-family tooling registry (curated, labeled L2, e.g. `os: cuda` → CUDA toolkit tooling) mentioned in Components/Discovery job.
- Toolchain-pinning across jobs within a single run: each job independently runs `rustup toolchain install nightly`, which can resolve to different nightly builds if a run straddles a nightly release boundary (typically UTC midnight), producing spurious cross-job disagreement that Task 21's own noise-floor test is specifically designed to catch but not fix. Not blocking for this plan (the batching correction above already re-verified everything against real CI evidence); worth a dedicated fix (Discovery resolves and pins a specific nightly date, passed to every downstream job) before this pipeline is trusted for long-running, many-round recursive use.
- `cargo-semver-checks` as a second target-independent check (Task 14's pattern) — needs a chosen baseline (last published crates.io version, or a specific git rev) before it can be built; not decided in this plan.
- `cargo udeps` as a further target-independent check (Task 14's pattern), straightforward to add the same way as `fmt`.
- `cargo miri`, corrected after an initial mischaracterization: Miri genuinely supports cross-target interpretation (`cargo miri test --target s390x-unknown-linux-gnu` is Miri's own documented example for big-endian testing — confirmed against Miri's real README, not assumed), so it belongs as a real axis crossed with `--target` (like `build_std`), not a single target-independent job like `fmt`. Detects out-of-bounds/use-after-free, uninitialized reads, misaligned access, invalid enum/bool discriminants, aliasing violations, memory leaks, and — directly relevant given probable async/concurrent components in this project — data races and weak-memory violations. Two real, confirmed limitations shape its scope: it does not support networking at all, and has very limited FFI access, meaning it will very likely fail against the native-link dependencies already confirmed real in this project (`openssl-sys`, `protobuf-src`, `onig_sys`, `ring`) — those failure points are themselves a mechanical way to locate exactly where behavior stops being portable/host-independent, not an incidental gap to route around. User directed: survey which of the 119 tier 1+2 targets Miri can actually interpret for (narrower than rustc's full codegen list) and which of `larql-cli`'s real dependencies hit the FFI/networking wall, before designing this as a task — not yet done.
- `wasm32-unknown-unknown` (tier 2, `std: true`, `host_tools: false`) as a third, explicitly-named standing-canary category alongside the two `std: false` canaries — its `std: true` doesn't mean fully OS-backed std (no real filesystem/threads/sockets without special setup), so post-mutation `core`/`alloc` testing against it asks a different question than testing against the `std: false` canaries. Already covered by Task 17's batch (tier 2, so inside Task 12's rescoped universe) — flagged here so it's named, not just anonymously blended into the batch.
- `cargo hack` feature-powerset testing — current feature coverage is only `default-features`/`no-default-features` (Task 8's `build-attempt`); the full feature powerset is unexplored.
- Broadening `build-attempt`'s clippy invocation to the spec's own stated lint breadth (`clippy::all`/`pedantic`/`nursery`/`cargo`) — currently runs bare `cargo clippy` with no lint-group flags at all, a real, confirmed gap between what's specified and what's built (found while surveying target-independent/target-dependent axes with the user).
- Target-side axes beyond std-availability/crate-type, surveyed with the user but not yet built into any probe: panic-strategy (abort vs. unwind — both canary targets are abort-only, no unwind contrast exists), atomics (`max-atomic-width`, including targets with none at all, e.g. `thumbv6m-none-eabi`), endianness (`target-endian`, e.g. `s390x-unknown-linux-gnu` — tier 2, real host tools, currently unexercised as a big-endian canary), pointer width, `host_tools`-derived runnability (currently hand-coded as a case statement in `runtime-test` rather than read from this real field), OS/environment family (WASI/UEFI/RTOS semantics), and libc/vendor variant for the same architecture (gnu/musl/msvc, `crt-static` default).
- Real runtime-test execution: native `cargo test`/`cargo run` for host targets, wasmtime execution for the wasm family (the `.cargo/config.toml` wasmtime runner wiring already exists in this repo and is currently unused by this pipeline) — currently the job only emits a curated runnability verdict, never actually runs anything (found by the final whole-branch review).
- Per-stage blast-radius containment for Stages A, B2, and B3 (currently only Stage B is checked), and a declared promotion postcondition for Stage A (currently `STAGE_POSTCONDITIONS` only covers stage-b/b2/b3, so the one stage that rewrites arbitrary source across all four mutated crates produces no measurable verdict at all) — found by the final whole-branch review.
- Extending the noise-floor self-test to compare against a prior round's real `sibling_sites` (already downloaded by `secondary-stage-c-and-promotion` via `prior-round-baseline`) so cross-run/cross-toolchain-install drift becomes observable, not just within-job determinism — found by the final whole-branch review; the toolchain-pinning fix listed elsewhere in this section remains the actual root-cause fix.
- Making the Secondary layer's per-target Stage C loop resilient to a single target's failure (currently `bash -e` with no per-target guards, so one bad target aborts the rest of its batch) — a further robustness improvement on top of this review's Fix 2, which converts the resulting silent data loss into a loud failure but does not reduce how often that loud failure occurs.

**Placeholder scan:** no "TBD"/"TODO" remain; the one inline placeholder note in Task 10 Step 1 (in the original single-target-invocation draft) was resolved before this task's content was finalized, not deferred.

**Type consistency:** `stage_promotes(stage_name, baseline_state, sibling_state)`'s CLI (`--stage`, `--baseline-state-file`, `--sibling-state-file`) and `depth_advanced(baseline_sites, sibling_sites)` introduced in Task 4 are used identically in Task 17's workflow wiring (same dict-shaped state objects keyed exactly as `STAGE_POSTCONDITIONS` expects: `lib_rs_content`, `unit_graph`, `metadata`+`expected_members`). `error_sites()` and `unit_graph_units_named()` from Task 1 are imported by name, unchanged, in Tasks 3, 4, and 12's inline scripts. Artifact-naming consistency re-verified after the batching restructure: Task 7 uploads `target-spec-<target>.json`/`cfg-<target>.txt`/`supported-crate-types-<target>.txt` inside `target-capability-batch-<N>`; Task 8 reads `supported-crate-types-$TARGET.txt` from that same download and uploads `unit-graph-<target>.json` inside `dependency-graph-batch-<N>`; Task 10's expected-file computation and Task 17's `next(Path("primary").glob(...))` baseline lookup both reference these exact filenames — checked directly against Tasks 7/8's `path: out/` upload blocks, not assumed.
