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

---

## File Structure

- `.github/workflows/target-analysis-pipeline.yml` — the new, generalized pipeline: `discovery` (also computes batches, Task 6), `target-capability`, `dependency-graph`, `build-attempt`, `runtime-test` (all four batched over `batch_index`, Tasks 7-9), `indexing` (Task 10), `secondary-mutate` (Stages A/B/B2/B3, single job, target-independent, Task 13), `secondary-stage-c-and-promotion` (batched, applies the mutation patch, Task 14), `next-round-baseline` (Task 15), `secondary-layer-self-test` (Task 16).
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
- Produces: `serde_features_ok(unit_graph: dict) -> bool`; `workspace_members_ok(metadata: dict, expected_members: list[str]) -> bool`; `no_std_scaffold_ok(lib_rs_content: str) -> bool`; `stage_promotes(stage_name: str, baseline_state: dict, sibling_state: dict) -> bool`; `depth_advanced(baseline_sites: set[tuple[str, int, str]], sibling_sites: set[tuple[str, int, str]]) -> bool`. Task 14 (Secondary-layer promotion wiring) calls `stage_promotes` once per stage (`stage-b`, `stage-b2`, `stage-b3`) and `depth_advanced` once per round to decide what folds into the next round's shared baseline.

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
unbatched 331-target list — as Tasks 7-9 and 15 originally would have — fails to
schedule. Worse, Task 8's build-attempt job crosses each target against 4
`build_std` modes × 3 `cargo_cmd`s × 2 feature configs × (typically ~5) crate types
— roughly 120 `cargo` invocations per target — so even a 256-target batch for that
specific job risks exceeding the 6-hour per-job limit. This task adds the batching
mechanism every later matrix-over-targets job needs, sized conservatively (12
targets per batch, uniformly across every batch-consuming job) so the heaviest job
(build-attempt) stays safely under both caps; Task 8's own real-run verification
step re-checks this estimate against actual job duration, since it's a reasoned
estimate, not a verified fact, until a real run confirms it.

This does not change `target-matrix` itself — Task 10's indexing job and Task 16's
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

Push a follow-up commit to the same branch Task 5 pushed to (`experiment/target-analysis-pipeline`) — not a new branch. Fetch the real run's `discovery` job log and confirm: `steps.resolve.outputs.batches` is a JSON array of arrays, each of length ≤12 (given the real 331-target count from Task 5's run, expect `ceil(331/12) = 28` batches, the last one shorter — reconfirm the exact target count on this run too, since it could have changed by even one entry since Task 5 ran), and `steps.resolve.outputs.batch-indices` is `[0, 1, ..., 27]` (or whatever the real count implies). Confirm `target-matrix`'s value is unchanged in shape from Task 5's run (still the full unbatched array) — Tasks 10 and 16 still need it that way.

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

### Task 11: Scope Discovery to rustc tier 1+2 targets

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
changes needed anywhere downstream (Tasks 7-10, 13-16 all derive everything from
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

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: query rustc's own tier classification and scope discovery to tier 1+2"
```

- [ ] **Step 8: Push and verify on a real runner**

Push a follow-up commit to `experiment/target-analysis-pipeline`. Fetch the real run's `discovery` job log and confirm: the "Query tier for every real target" step completes for all 331 real targets (this step alone should take a few minutes — 331 lightweight `rustc --print` invocations, no compilation), `target-tiers.json` is uploaded alongside `target-list.txt`, and `steps.resolve.outputs.matrix` now contains exactly 119 targets (down from 331) — cross-check this count directly against this task's own real tier tally (8 tier-1 + 111 tier-2 = 119) rather than assuming the plan's number is still accurate, since the exact real target list can drift between runs. Confirm `nvptx64-nvidia-cuda` is still present in the filtered matrix (it must be — tier 2, the standing canary). Confirm `steps.resolve.outputs.batches`/`batch-indices` reflect `ceil(119/12) = 10` batches, not 28. This run will re-trigger the full pipeline (all jobs depend on `discovery`) — expect a real run against the new, smaller 119-target/10-batch universe end to end; this is the first real evidence of the pipeline running at its new, intended scope, not just Discovery in isolation.

---

### Task 12: Target-independent checks run before target-dependent jobs (fmt)

**User directive, general and forward-looking:** anything target-independent must run
before any target-dependent job. This mirrors the Task 13 restructure's own finding
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

### Task 13: Generalize the Secondary-layer mutation stages (A, B, B2, B3) — single job, no target matrix

**Restructured from the original two-job (per-crate × per-target) design.** All four mutation stages are target-independent: Stage A runs `clippy --fix` against the host target (never `--target`), Stage B is a pure text edit to `lib.rs`, and Stage B2/B3 are pure text edits to `Cargo.toml` files — none of them reference a target triple at all. The original draft matrixed this identical, target-independent work over all 331 targets (and, for Stage A/B, over 5 crates too — 1655 combinations), for no reason: target only enters the Secondary layer at Stage C. This version runs the whole mutation pipeline exactly once per pipeline run, producing a single patch that Task 14's batched Stage C job downloads and applies before checking against each target — this is also what fixes a real bug the original draft had: without an explicit patch-apply step, Stage C would have run against a fresh, unmutated checkout every time, silently checking pristine source instead of the mutation it was supposed to be evaluating.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-mutate` job, adapted from `experiment-cuda-nvptx.yml`'s existing `nostd-fix-attempt` job's Stage A/B/B2/B3 steps at `.github/workflows/experiment-cuda-nvptx.yml:477-624`)

**Interfaces:**
- Consumes: nothing target-specific. `needs: [discovery, indexing]` expresses a real ordering constraint even without consuming `target-matrix` directly — this stage's whole purpose (per Standing Principle 6) is to be validated against a Primary-layer baseline, so it still waits for the Primary layer's indexing to complete first.
- Produces: one `secondary-mutation` artifact containing `full-mutation.patch` (a single `git diff` of the whole tree after all four stages), per-crate `baseline-lib-rs-<crate>.txt` / `sibling-lib-rs-<crate>.txt` pairs (Stage B promotion input), and `baseline-metadata.json` (the unmutated `cargo metadata` output, captured before Stage B3 trims workspace members — Stage B3 promotion input). Consumed by Task 14.

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

The scaffold-insertion Python is the exact fix this session already verified against a real 71-line doc comment (inserting after any leading `//!`/`#![`/blank-line block, never before it); the Stage B2 sed pattern and Stage B3 reachable-crate list are the exact ones this session verified against real CI output — all reused directly, not reinvented. `baseline-metadata.json` is captured before Stage B3 runs specifically so Task 14 never needs to reconstruct the unmutated workspace member list by parsing `git show`-retrieved TOML text — it's just read directly from this artifact.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add generalized Secondary-layer mutation job (Stages A, B, B2, B3), target-independent"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: the `secondary-mutate` job runs exactly once (no matrix), Stage A/B applies cleanly for every crate (no doc-comment corruption — check `larql-compute`'s output specifically, the crate with the largest leading doc comment), the `stage-b2`/`stage-b3` steps' start timestamps overlap (background/wait proven concurrent, matching Standing Principle 8's genuine-dependency-only sequencing), and the uploaded `secondary-mutation` artifact contains `full-mutation.patch`, 5 baseline/sibling lib.rs pairs, and `baseline-metadata.json`.

---

### Task 14: Stage C and the promotion/depth-advancement decision, batched

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-stage-c-and-promotion` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.batches`/`batch-indices` (Task 6); Task 13's `secondary-mutation` artifact (the patch plus Stage B/B3 baselines); Task 7's `target-capability-batch-<N>` and Task 8's `dependency-graph-batch-<N>` artifacts (the mechanically-grounded Primary-layer baselines for the Stage B2 promotion check — the per-target unmutated `unit-graph-<target>.json` this job would otherwise have no other source for); `scripts/target_analysis_promotion.py`'s CLI (Task 4).
- Produces: `promotion-decision-batch-<batch_index>` artifact containing, per target in that batch, the stage-b/b2/b3 promotion verdicts and the depth-advancement decision — the actual mechanical output this session's "measurable difference" discussion exists to produce.

**The critical fix this task makes over the original draft:** the original version did a fresh `actions/checkout@v4` and ran `cargo check` directly, with no step ever applying Task 13's mutation — Stage C would have silently checked pristine, unmutated source on every run, and the whole promotion/depth-advancement machinery would have been evaluating data that never reflected the mutation it claimed to evaluate. This version's very first non-checkout step downloads `secondary-mutation` and runs `git apply mutation/full-mutation.patch` before anything else.

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
          baseline_messages = []  # first round: no prior-round Stage C output exists yet (Task 15 wires round-over-round)

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

Push and confirm: `ceil(331/12) = 28` `promotion-decision-batch-<N>` artifacts are produced, each containing per-target `stage-c-<target>.json`, `promotion-stage-b-<target>.json`, `promotion-stage-b2-<target>.json`, `promotion-stage-b3-<target>.json`, and `depth-decision-<target>.json` files. Specifically check the batch containing `nvptx64-nvidia-cuda`: confirm `stage-c-nvptx64-nvidia-cuda.json` shows real compiler output against the *mutated* tree (spot-check that the file content differs from what Task 8's unmutated `build-attempt` probe recorded for the same target — this is the direct evidence the patch was actually applied, not skipped), and confirm all three `promotion-stage-b*-nvptx64-nvidia-cuda.json` files show real `"promotes": true/false` verdicts (not an error) — this first round has no real prior-round Stage C baseline yet (`baseline_messages = []`), so `depth_advanced` should read `true` for every target with any error at all (every site is "newly resolved" relative to an empty baseline is wrong — re-check this reasoning empirically against the real output: an empty baseline means `baseline_sites - sibling_sites` is always empty regardless of `sibling_sites`, since you cannot subtract from nothing, so `depth_advanced` should read `false` for every target on this first round; Task 15 wires the real round-over-round baseline that makes this check meaningful).

---

### Task 15: Recursive-round baseline handoff

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `next-round-baseline` job)

**Interfaces:**
- Consumes: `promotion-decision-batch-<batch_index>` artifacts (Task 14), each containing multiple per-target `depth-decision-<target>.json` and `promotion-stage-*-<target>.json` files.
- Produces: `round-baseline` artifact — the folded-forward set of promoted stage diffs plus the full set of non-promoted results preserved separately — retrievable by the next `push` to the same branch pattern (the mechanism by which round N+1 begins from round N's baseline, per this session's mutual-recursion finding: observation and mutation inform each other round over round, neither completes independently).

- [ ] **Step 1: Add the job**

```yaml
  next-round-baseline:
    name: Fold promoted diffs into next round's baseline
    needs: [discovery, secondary-stage-c-and-promotion]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download all promotion-decision batch artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: "promotion-decision-batch-*"
          path: promotion-decisions
      - name: Fold promoted results, preserve non-promoted results separately
        run: |
          python3 - <<'PYEOF'
          import json
          import re
          from pathlib import Path

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
                  record = {**depth_decision, "stage_promotions": stage_verdicts}
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

Per the spec's Data flow open question, cross-run artifact retention beyond GitHub's default 90-day expiry is explicitly not resolved by this plan — `round-baseline` is retrievable within that window via `actions/download-artifact` from the prior run, which is sufficient for the recursive loop's own mechanics without deciding the longer-term history question.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add next-round-baseline job folding promoted diffs forward per the measurable-difference rule"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: `round-baseline` artifact is produced, `promoted` contains only targets where `depth_advanced` was `true`, and `preserved_not_promoted` contains the rest — nothing is discarded, matching the promotion rule: a sibling producing zero measurable difference is preserved as its own artifact, not folded forward and not deleted.

---

### Task 16: Secondary-layer test suite — noise floor, blast-radius containment, ephemerality

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-layer-self-test` job)

**Interfaces:**
- Consumes: nothing from earlier jobs — this job validates the Secondary-layer mechanism itself, independent of any specific crate/target result, per the spec's Testing section.

- [ ] **Step 1: Add the noise-floor check**

```yaml
  secondary-layer-self-test:
    name: Secondary-layer self-test (noise floor, blast radius, ephemerality)
    needs: [discovery]
    runs-on: ubuntu-latest
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
          git status --porcelain > before-stage-b.txt
          python3 - <<'PYEOF'
          import re
          from pathlib import Path

          lib_rs = Path("crates/larql-boundary/src/lib.rs")
          text = lib_rs.read_text(encoding="utf-8")
          lines = text.splitlines(keepends=True)
          insert_at = 0
          for i, line in enumerate(lines):
              stripped = line.strip()
              if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
                  insert_at = i + 1
              else:
                  break
          lines.insert(insert_at, "#![no_std]\nextern crate alloc;\n")
          lib_rs.write_text("".join(lines), encoding="utf-8")
          PYEOF
          CHANGED=$(git diff --name-only)
          if [ "$CHANGED" != "crates/larql-boundary/src/lib.rs" ]; then
            echo "::error::Stage B touched files outside its declared scope: $CHANGED"
            exit 1
          fi
          echo "Blast radius contained to the declared file."
      - name: "Ephemerality: assert no git commit/push exists anywhere in this workflow file"
        run: |
          if grep -Eq "git (commit|push)" .github/workflows/target-analysis-pipeline.yml; then
            echo "::error::Found a git commit/push in the pipeline workflow — mutations must stay ephemeral to the job checkout."
            exit 1
          fi
          echo "No git commit/push found in the pipeline workflow."
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add Secondary-layer self-test job (noise floor, blast radius, ephemerality)"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: all three checks pass on the real pipeline as currently written. Then deliberately introduce a Stage B bug that touches an extra file (e.g., also modify `Cargo.toml` in the blast-radius test's inline script) in a throwaway commit, confirm the blast-radius check fails loudly, then revert.

---

## Self-Review

**Spec coverage** — every spec section maps to a task:
- Foundational framework / Standing design principles → Global Constraints (verbatim-sourced).
- Discovery job → Task 5, extended by Task 6 (batching — a real GitHub Actions platform limit discovered empirically on Task 5's own real run, per Standing Principle 1).
- Target-capability probes → Task 7.
- Dependency-graph probes → Task 8 (dependency-graph half).
- Build-attempt probes (including the `only-cdylib` crate-type correction, and the `--crate-type`/`rust-src` fixes caught before dispatch) → Task 8 (build-attempt half).
- Runtime-test probes → Task 9.
- Indexing (structural extraction, contradiction rule, completeness enforcement, now at file-level across batched artifacts) → Tasks 3 and 10.
- Discovery scope (tier 1+2 only, 119 of 331 real targets — a user-directed, real-evidence-driven correction to the original "every rustc target" design, mechanically grounded in `target-spec-json`'s own `metadata.tier` field, not agent judgment) → Task 11.
- Target-independent-before-target-dependent sequencing (user directive, generalizing the Task 13 mutation-job restructure's own finding into a standing rule) → Task 12 (`cargo fmt --check`, the first concrete case).
- Secondary-layer mutation stages A/B/B2/B3 (with `background`/`wait` concurrency, target-independent, single job) → Task 13. Stage C (batched, applying the mutation patch — the critical bug this restructure fixes over the original per-target-matrixed draft) → Task 14.
- The measurable-difference / promotion rule (this session's immediate deliverable) → Task 4 (script) and Task 14 (wiring, with mechanically-grounded b2/b3 baselines sourced from the Primary layer's own artifacts rather than fabricated).
- Recursive round-over-round baseline handoff → Task 15.
- Error handling (honest-result pattern, retries narrow to network calls, platform-limit category) → reused directly from the proven `experiment-cuda-nvptx.yml` patterns in Tasks 8 and 14; the retry/platform-limit categories are not separately re-implemented since Tasks 7-9's probes don't call rate-limited external APIs beyond `rustc`/`cargo` — Discovery's crates.io/GitHub SBOM calls (mentioned in the spec's Discovery job description) are the one place a narrow retry would apply and are flagged here as **not yet implemented**: Task 5 only wires `rustc --print target-list`, not the crates.io/SBOM ecosystem-discovery calls. This is a real gap — added as a follow-up task below rather than silently left out.
- Testing (noise floor, blast-radius, golden fixtures, ephemerality, cross-target/native comparison) → Task 16 covers noise floor, blast radius, ephemerality directly. Golden fixtures (`serde-nostd-probe`-style planted-outcome crates) and cross-target/native comparison are **not yet implemented** — flagged below.
- Explicitly not doing (no caching, no CI commits, no agent curation presented as L1) → Global Constraints + Task 16's ephemerality check enforces the no-commits rule structurally.

**Follow-up tasks not included in this plan** (real gaps, not placeholders — each needs its own task the way Tasks 1-16 are written, deferred here because this plan's immediate trigger was the promotion-rule definition, not full spec closure):
- Discovery job's crates.io/GitHub SBOM ecosystem-discovery calls, with narrow bounded retry on those specific network calls (spec: Components/Discovery job, Error handling/Retries).
- Golden-fixture crates with a planted, known-in-advance outcome, generalizing `serde-nostd-probe` (spec: Testing).
- Cross-target/cross-native comparison job, once the target matrix includes both nvptx and at least one native target's Stage C result for the same underlying finding (spec: Testing) — this is naturally sequenced after Tasks 1-16 produce enough real round data to compare, not before.
- The target-family tooling registry (curated, labeled L2, e.g. `os: cuda` → CUDA toolkit tooling) mentioned in Components/Discovery job.
- Toolchain-pinning across jobs within a single run: each job independently runs `rustup toolchain install nightly`, which can resolve to different nightly builds if a run straddles a nightly release boundary (typically UTC midnight), producing spurious cross-job disagreement that Task 16's own noise-floor test is specifically designed to catch but not fix. Not blocking for this plan (the batching correction above already re-verified everything against real CI evidence); worth a dedicated fix (Discovery resolves and pins a specific nightly date, passed to every downstream job) before this pipeline is trusted for long-running, many-round recursive use.
- `cargo-semver-checks` as a second target-independent check (Task 12's pattern) — needs a chosen baseline (last published crates.io version, or a specific git rev) before it can be built; not decided in this plan.
- `cargo udeps` and `cargo miri` as further target-independent checks (Task 12's pattern) — udeps is straightforward to add the same way as `fmt`; miri is its own quasi-target (an interpreter, not a real backend) and needs its own design pass, not just a slot in the existing per-crate matrix.
- `cargo hack` feature-powerset testing — current feature coverage is only `default-features`/`no-default-features` (Task 8's `build-attempt`); the full feature powerset is unexplored.
- Broadening `build-attempt`'s clippy invocation to the spec's own stated lint breadth (`clippy::all`/`pedantic`/`nursery`/`cargo`) — currently runs bare `cargo clippy` with no lint-group flags at all, a real, confirmed gap between what's specified and what's built (found while surveying target-independent/target-dependent axes with the user).
- Target-side axes beyond std-availability/crate-type, surveyed with the user but not yet built into any probe: panic-strategy (abort vs. unwind — both canary targets are abort-only, no unwind contrast exists), atomics (`max-atomic-width`, including targets with none at all, e.g. `thumbv6m-none-eabi`), endianness (`target-endian`, e.g. `s390x-unknown-linux-gnu` — tier 2, real host tools, currently unexercised as a big-endian canary), pointer width, `host_tools`-derived runnability (currently hand-coded as a case statement in `runtime-test` rather than read from this real field), OS/environment family (WASI/UEFI/RTOS semantics), and libc/vendor variant for the same architecture (gnu/musl/msvc, `crt-static` default).

**Placeholder scan:** no "TBD"/"TODO" remain; the one inline placeholder note in Task 10 Step 1 (in the original single-target-invocation draft) was resolved before this task's content was finalized, not deferred.

**Type consistency:** `stage_promotes(stage_name, baseline_state, sibling_state)`'s CLI (`--stage`, `--baseline-state-file`, `--sibling-state-file`) and `depth_advanced(baseline_sites, sibling_sites)` introduced in Task 4 are used identically in Task 14's workflow wiring (same dict-shaped state objects keyed exactly as `STAGE_POSTCONDITIONS` expects: `lib_rs_content`, `unit_graph`, `metadata`+`expected_members`). `error_sites()` and `unit_graph_units_named()` from Task 1 are imported by name, unchanged, in Tasks 3, 4, and 12's inline scripts. Artifact-naming consistency re-verified after the batching restructure: Task 7 uploads `target-spec-<target>.json`/`cfg-<target>.txt`/`supported-crate-types-<target>.txt` inside `target-capability-batch-<N>`; Task 8 reads `supported-crate-types-$TARGET.txt` from that same download and uploads `unit-graph-<target>.json` inside `dependency-graph-batch-<N>`; Task 10's expected-file computation and Task 14's `next(Path("primary").glob(...))` baseline lookup both reference these exact filenames — checked directly against Tasks 7/8's `path: out/` upload blocks, not assumed.
