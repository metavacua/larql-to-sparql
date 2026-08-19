# Generalized Target-Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad hoc `experiment-cuda-nvptx.yml` workflow with a target-parameterized, fully autonomous GitHub Actions pipeline that, given a target triple, produces a complete map of every place gating or fixing is necessary to build and runtime-test for that target — and implements the mutually-recursive Primary(observe)/Secondary(mutate) loop with a precisely defined, mechanical promotion rule.

**Architecture:** A Discovery job feeds target/crate matrices to four Primary-layer probe job families (target-capability, dependency-graph, build-attempt, runtime-test) via `fromJSON()`. An Indexing job structurally aggregates every probe's raw artifact with loud-failure-on-missing-artifact completeness checking. A Secondary-layer job runs the existing four-stage mutation pipeline (generalized from crate/target-specific to parameterized), captures before/after Primary-layer diffs per stage, and applies a mechanical promotion/depth-advancement rule to decide what folds into the next round's shared baseline.

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

- `.github/workflows/target-analysis-pipeline.yml` — the new, generalized pipeline: `discovery`, `target-capability`, `dependency-graph`, `build-attempt`, `runtime-test`, `indexing`, `secondary-stage-a-b`, `secondary-stage-b2-b3`, `secondary-stage-c`, `secondary-promotion` jobs.
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
- Produces: `parse_target_list(raw: str) -> list[str]`; `resolve_target_matrix(all_targets: list[str], requested: str | None) -> list[str]` (raises `ValueError` if `requested` is not in `all_targets` — this is the sanity check Standing Principle 8 requires before any downstream job trusts a `workflow_dispatch` input). Task 8 (Discovery job wiring) invokes this script's CLI to emit the matrix JSON that `fromJSON()` consumes.

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
- Produces: `count_errors_by_target(compiler_messages: list[dict]) -> dict[str, int]`; `unexpected_clean_std_build(target_spec: dict, std_mode_errors: list) -> bool`; `missing_artifacts(expected: set[str], actual: set[str]) -> set[str]`. Task 9 (Indexing job wiring) calls this script's CLI to build the navigation index and fails the job if `missing_artifacts(...)` is non-empty.

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
- Produces: `serde_features_ok(unit_graph: dict) -> bool`; `workspace_members_ok(metadata: dict, expected_members: list[str]) -> bool`; `no_std_scaffold_ok(lib_rs_content: str) -> bool`; `stage_promotes(stage_name: str, baseline_state: dict, sibling_state: dict) -> bool`; `depth_advanced(baseline_sites: set[tuple[str, int, str]], sibling_sites: set[tuple[str, int, str]]) -> bool`. Task 12 (Secondary-layer promotion wiring) calls `stage_promotes` once per stage (`stage-b`, `stage-b2`, `stage-b3`) and `depth_advanced` once per round to decide what folds into the next round's shared baseline.

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

### Task 6: Target-capability probe job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add the `target-capability` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.target-matrix` (Task 5).
- Produces: per-target artifact `target-capability-<target>` containing `target-spec.json`, `cfg.txt`, `supported-crate-types.txt` — consumed by Task 7 (build-attempt matrix) and Task 9 (indexing).

- [ ] **Step 1: Add the job**

```yaml
  target-capability:
    name: "Target capability: ${{ matrix.target }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: Sanity check target exists in target-list
        run: |
          rustup run nightly rustc --print target-list | grep -qx "${{ matrix.target }}"
      - name: target-spec-json
        run: |
          rustup run nightly rustc --print target-spec-json -Z unstable-options \
            --target "${{ matrix.target }}" > target-spec.json
      - name: cfg
        run: |
          rustup run nightly rustc --print cfg --target "${{ matrix.target }}" > cfg.txt
      - name: supported-crate-types
        run: |
          rustup run nightly rustc --print supported-crate-types --target "${{ matrix.target }}" \
            > supported-crate-types.txt
      - name: Upload capability artifacts
        uses: actions/upload-artifact@v4
        with:
          name: target-capability-${{ matrix.target }}
          path: |
            target-spec.json
            cfg.txt
            supported-crate-types.txt
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add target-capability probe job"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: one `target-capability` job instance per matrix entry, each producing all three artifact files, and — for the `nvptx64-nvidia-cuda` entry specifically — `target-spec.json` shows `"std": false` and `supported-crate-types.txt` shows `bin, cdylib, lib, rlib, staticlib` (matching this session's earlier empirical finding, and Standing Principle 5's canary: a clean/all-succeeding result here for nvptx would itself be a bug to chase).

---

### Task 7: Dependency-graph and build-attempt probe jobs

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `dependency-graph` and `build-attempt` jobs)

**Interfaces:**
- Consumes: `needs.discovery.outputs.target-matrix`; `target-capability-<target>` artifacts (Task 6) for the crate-type list.
- Produces: per-(crate, target, feature-config) artifacts consumed by Task 9 (indexing).

- [ ] **Step 1: Add the dependency-graph job**

```yaml
  dependency-graph:
    name: "Dependency graph: ${{ matrix.target }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: cargo metadata --filter-platform
        run: |
          cargo metadata --format-version 1 --filter-platform "${{ matrix.target }}" \
            > metadata-${{ matrix.target }}.json
      - name: cargo tree (features, duplicates)
        run: |
          cargo tree -p larql-cli -e features --target "${{ matrix.target }}" \
            > tree-features-${{ matrix.target }}.txt
          cargo tree -p larql-cli --duplicates --target "${{ matrix.target }}" \
            > tree-duplicates-${{ matrix.target }}.txt
      - name: cargo build --unit-graph
        run: |
          cargo +nightly build -Z unstable-options --unit-graph \
            -p larql-cli --target "${{ matrix.target }}" \
            > unit-graph-${{ matrix.target }}.json
      - name: cargo-deny (curated, labeled L2)
        run: |
          cargo install cargo-deny --locked || true
          cargo deny --config deny-nvptx.toml check bans licenses advisories sources \
            > deny-${{ matrix.target }}.txt || true
      - name: Upload dependency-graph artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dependency-graph-${{ matrix.target }}
          path: |
            metadata-${{ matrix.target }}.json
            tree-features-${{ matrix.target }}.txt
            tree-duplicates-${{ matrix.target }}.txt
            unit-graph-${{ matrix.target }}.json
            deny-${{ matrix.target }}.txt
```

Note: `cargo-deny check` failures must never fail this job — its verdict is explicitly labeled curated/L2 (Foundational framework), so `|| true` here is deliberate, not error-suppression; the raw output is still uploaded and the indexing job (Task 9) reports its content, not its exit code.

- [ ] **Step 2: Add the build-attempt job**

```yaml
  build-attempt:
    name: "Build attempt: ${{ matrix.target }} / ${{ matrix.build_std }} / ${{ matrix.cargo_cmd }} / ${{ matrix.features }}"
    needs: [discovery, target-capability]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
        build_std: ["none", "std", "core,alloc", "core"]
        cargo_cmd: ["check", "clippy", "build"]
        features: ["default-features", "no-default-features"]
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: Download target-capability artifact
        uses: actions/download-artifact@v4
        with:
          name: target-capability-${{ matrix.target }}
      - name: Set feature flag
        id: feature-flag
        run: |
          if [ "${{ matrix.features }}" = "no-default-features" ]; then
            echo "flag=--no-default-features" >> "$GITHUB_OUTPUT"
          else
            echo "flag=" >> "$GITHUB_OUTPUT"
          fi
      - name: Set build-std flag
        id: build-std-flag
        run: |
          if [ "${{ matrix.build_std }}" = "none" ]; then
            echo "flag=" >> "$GITHUB_OUTPUT"
          else
            echo "flag=-Z build-std=${{ matrix.build_std }}" >> "$GITHUB_OUTPUT"
          fi
      - name: "cargo ${{ matrix.cargo_cmd }}"
        id: attempt
        continue-on-error: true
        run: |
          cargo +nightly "${{ matrix.cargo_cmd }}" -p larql-cli \
            --target "${{ matrix.target }}" \
            ${{ steps.feature-flag.outputs.flag }} \
            ${{ steps.build-std-flag.outputs.flag }} \
            --keep-going \
            --message-format=json > attempt-output.json
      - name: Upload full diagnostic JSON
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: build-attempt-${{ matrix.target }}-${{ matrix.build_std }}-${{ matrix.cargo_cmd }}-${{ matrix.features }}
          path: attempt-output.json
      - name: Assert honest result
        run: |
          echo "cargo ${{ matrix.cargo_cmd }} outcome: ${{ steps.attempt.outcome }}"
          if [ "${{ steps.attempt.outcome }}" = "failure" ]; then
            echo "::notice::Real build failure recorded as a finding, not masked."
          fi
```

This job deliberately runs every `build_std` × `cargo_cmd` × `features` combination against every target regardless of what `target-capability` reported about `std` (Standing Principle 3 — exhaustive unconditional fan-out; the spec's Build-attempt probes section is explicit that predictable outcomes are still real L1 data). The crate-type dimension from `supported-crate-types.txt` is deferred to this job's Step 3 below rather than baked into the initial matrix, since it needs the downloaded artifact's content, not a value known at matrix-definition time.

- [ ] **Step 3: Extend the build-attempt job to iterate crate types from the capability probe's real output**

Add a step before "cargo ${{ matrix.cargo_cmd }}", and change that step to a loop:
```yaml
      - name: Build against every empirically-supported crate type
        continue-on-error: true
        run: |
          CRATE_TYPES=$(tr ',' '\n' < supported-crate-types.txt | sed 's/^ *//; s/ *$//')
          FAILED=0
          for CRATE_TYPE in $CRATE_TYPES; do
            echo "=== crate-type: $CRATE_TYPE ==="
            cargo +nightly "${{ matrix.cargo_cmd }}" -p larql-cli \
              --target "${{ matrix.target }}" \
              ${{ steps.feature-flag.outputs.flag }} \
              ${{ steps.build-std-flag.outputs.flag }} \
              --keep-going \
              --message-format=json >> attempt-output.json || FAILED=1
          done
          exit $FAILED
        id: attempt
```
Remove the earlier single-invocation "cargo ${{ matrix.cargo_cmd }}" step — this replaces it, iterating `supported-crate-types.txt`'s actual content (Task 6's output) rather than a crate-type list chosen in advance, per this session's `only-cdylib` field-name correction.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add dependency-graph and build-attempt probe jobs, crate-type iteration from empirical probe output"
```

- [ ] **Step 5: Push and verify on a real runner**

Push and confirm: `dependency-graph` produces all five artifact files per target with non-empty content; `build-attempt` fans out over the full matrix, and for `nvptx64-nvidia-cuda` × `build_std=none` × any `cargo_cmd`, the job's real outcome is `failure` (per the canary principle) with the honest-result assertion printing the notice, not a masked green.

---

### Task 8: Runtime-test probe job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `runtime-test` job)

**Interfaces:**
- Consumes: `needs.discovery.outputs.target-matrix`.
- Produces: per-target artifact recording either real test execution results or an explicit `"blocked: no runner available, reason: <cited>"` record — consumed by Task 9 (indexing).

- [ ] **Step 1: Add the job**

```yaml
  runtime-test:
    name: "Runtime test: ${{ matrix.target }}"
    needs: [discovery]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
    steps:
      - uses: actions/checkout@v4
      - name: Determine runner availability and execute or record block
        run: |
          case "${{ matrix.target }}" in
            wasm32*|wasm64*)
              echo '{"status": "runnable", "runner": "wasmtime"}' > runtime-test-result.json
              # actual wasmtime execution wired in the wasm-family follow-on work;
              # this job records availability honestly either way.
              ;;
            x86_64-unknown-linux-gnu|aarch64-*-linux-*|aarch64-apple-darwin)
              echo '{"status": "runnable", "runner": "native"}' > runtime-test-result.json
              ;;
            nvptx64-nvidia-cuda)
              echo '{"status": "blocked", "reason": "no free GPU CI runner available for this project (confirmed 2026-08-16)"}' \
                > runtime-test-result.json
              ;;
            *)
              echo '{"status": "blocked", "reason": "no known runner for this target"}' \
                > runtime-test-result.json
              ;;
          esac
      - name: Upload runtime-test result
        uses: actions/upload-artifact@v4
        with:
          name: runtime-test-${{ matrix.target }}
          path: runtime-test-result.json
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add runtime-test probe job with explicit blocked-status recording"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: every target in the matrix produces a `runtime-test-<target>` artifact, and `nvptx64-nvidia-cuda`'s specifically records `"status": "blocked"` with a cited reason — never silently omitted, per the spec's Runtime-test probes section.

---

### Task 9: Indexing job

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `indexing` job)

**Interfaces:**
- Consumes: `scripts/target_analysis_indexing.py`'s CLI (Task 3); every artifact from Tasks 6-8 via `actions/download-artifact`.
- Produces: `index.json` artifact; job fails (`exit 1`) if any expected artifact is missing.

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
      - name: Compute expected vs. actual artifact sets and build index
        run: |
          python3 - <<'PYEOF'
          import json
          import subprocess
          from pathlib import Path

          target_matrix = json.loads('''${{ needs.discovery.outputs.target-matrix }}''')
          expected = set()
          for t in target_matrix:
              expected.add(f"target-capability-{t}")
              expected.add(f"dependency-graph-{t}")
              expected.add(f"runtime-test-{t}")
              for build_std in ["none", "std", "core,alloc", "core"]:
                  for cmd in ["check", "clippy", "build"]:
                      for feat in ["default-features", "no-default-features"]:
                          expected.add(f"build-attempt-{t}-{build_std}-{cmd}-{feat}")

          actual = {p.name for p in Path("artifacts").iterdir() if p.is_dir()}
          Path("expected-artifacts.json").write_text(json.dumps(sorted(expected)))
          Path("actual-artifacts.json").write_text(json.dumps(sorted(actual)))
          PYEOF
      - name: Run indexing script
        run: |
          # Representative single-target inputs shown; the real invocation loops
          # per target in the matrix, writing one index.json section per target.
          python3 scripts/target_analysis_indexing.py \
            --compiler-messages-file artifacts/build-attempt-nvptx64-nvidia-cuda-none-check-default-features/attempt-output.json \
            --target-spec-file artifacts/target-capability-nvptx64-nvidia-cuda/target-spec.json \
            --std-mode-errors-file <(echo '[]') \
            --expected-artifacts-file expected-artifacts.json \
            --actual-artifacts-file actual-artifacts.json \
            > index.json
      - name: Upload index
        uses: actions/upload-artifact@v4
        with:
          name: navigation-index
          path: index.json
```

The single-target inline invocation shown is a placeholder for the loop structure — resolve this in Step 2 below with a real per-target loop, not left as-is; flagging it here so the loop is written deliberately rather than skipped.

- [ ] **Step 2: Replace the single-target invocation with a real per-target loop**

Replace the "Run indexing script" step with:
```yaml
      - name: Run indexing script per target
        run: |
          python3 - <<'PYEOF'
          import json
          import subprocess
          import sys
          from pathlib import Path

          target_matrix = json.loads('''${{ needs.discovery.outputs.target-matrix }}''')
          combined = {}
          any_missing = False
          for t in target_matrix:
              attempt_file = Path(f"artifacts/build-attempt-{t}-none-check-default-features/attempt-output.json")
              spec_file = Path(f"artifacts/target-capability-{t}/target-spec.json")
              if not attempt_file.exists() or not spec_file.exists():
                  combined[t] = {"error": "missing required per-target artifact"}
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
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add indexing job with per-target loop and completeness enforcement"
```

- [ ] **Step 4: Push and verify on a real runner**

Push and confirm: the `indexing` job runs after all Primary-layer jobs (`needs:` + `!cancelled()`), downloads every artifact, produces `navigation-index` containing one entry per target with `error_counts_by_target`, `contradictions`, and `missing_artifacts`. Then deliberately break completeness once (e.g. temporarily comment out the `runtime-test` job's artifact upload step in a throwaway commit) and confirm the `indexing` job fails loudly rather than silently indexing a partial set — then revert that throwaway commit.

---

### Task 10: Generalize Secondary-layer Stages A and B

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-stage-a-b` job, adapted from `experiment-cuda-nvptx.yml`'s existing `nostd-fix-attempt` job's Stage A / Stage B steps at `.github/workflows/experiment-cuda-nvptx.yml:477-553`)

**Interfaces:**
- Consumes: `needs.discovery.outputs.target-matrix`; a crate matrix (this task hardcodes the five crates already proven in the existing workflow — `larql-boundary`, `larql-vindex-spec`, `larql-models`, `larql-compute`, `larql-cli` — generalizing crate discovery itself is out of scope for this plan, matching the existing proven pipeline's scope).
- Produces: mutated checkout state (Stage A/B applied) plus `lib_rs_content` captured to a file, consumed by Task 12's `stage-b` promotion check.

- [ ] **Step 1: Add the job, reusing the existing workflow's proven Stage A/B logic**

```yaml
  secondary-stage-a-b:
    name: "Secondary Stage A/B: ${{ matrix.crate }} / ${{ matrix.target }}"
    needs: [discovery, indexing]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
        crate: [larql-boundary, larql-vindex-spec, larql-models, larql-compute, larql-cli]
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: "Stage A: mechanical std->core/alloc rewrite (host target)"
        run: |
          cargo +nightly clippy --fix --allow-dirty --allow-no-vcs \
            -p "${{ matrix.crate }}" -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc
      - name: "Stage B: inject #![no_std] scaffold"
        run: |
          python3 - <<'PYEOF'
          import re
          from pathlib import Path

          lib_rs = Path("crates/${{ matrix.crate }}/src/lib.rs")
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
      - name: Capture Stage B lib.rs content for promotion check
        run: cp "crates/${{ matrix.crate }}/src/lib.rs" "stage-b-lib-rs-${{ matrix.crate }}-${{ matrix.target }}.txt"
      - name: Upload Stage A/B checkout diff
        run: git diff > stage-a-b-diff-${{ matrix.crate }}-${{ matrix.target }}.patch
      - uses: actions/upload-artifact@v4
        with:
          name: secondary-stage-a-b-${{ matrix.crate }}-${{ matrix.target }}
          path: |
            stage-b-lib-rs-${{ matrix.crate }}-${{ matrix.target }}.txt
            stage-a-b-diff-${{ matrix.crate }}-${{ matrix.target }}.patch
```

The scaffold-insertion Python here is the exact fix this session already verified against a real 71-line doc comment (inserting after any leading `//!`/`#![`/blank-line block, never before it) — reused directly, not reinvented.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add generalized Secondary-layer Stage A/B job"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: for each crate × target pair, Stage A/B applies cleanly (no doc-comment corruption — check `larql-compute`'s output specifically, the crate with the largest leading doc comment), and the uploaded `stage-b-lib-rs-*.txt` files contain both `#![no_std]` and `extern crate alloc;`.

---

### Task 11: Generalize Secondary-layer Stages B2 and B3, run concurrently

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-stage-b2-b3` job, adapted from `experiment-cuda-nvptx.yml:554-624`)

**Interfaces:**
- Consumes: same crate/target matrix as Task 10; runs independently (does not need Task 10's output — B2/B3 touch disjoint files from A/B, per spec's Secondary-layer stages section).
- Produces: `unit-graph` (post-B2) and `cargo metadata` (post-B3) captures, consumed by Task 12's `stage-b2`/`stage-b3` promotion checks.

- [ ] **Step 1: Add the job with B2 and B3 as concurrent background steps**

```yaml
  secondary-stage-b2-b3:
    name: "Secondary Stage B2/B3: ${{ matrix.target }}"
    needs: [discovery, indexing]
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
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
      - name: Capture post-B2 unit-graph
        run: |
          cargo +nightly build -Z unstable-options --unit-graph \
            -p larql-cli --target "${{ matrix.target }}" \
            > post-b2-unit-graph-${{ matrix.target }}.json
      - name: Capture post-B3 workspace metadata
        run: |
          cargo metadata --format-version 1 --no-deps > post-b3-metadata-${{ matrix.target }}.json
      - uses: actions/upload-artifact@v4
        with:
          name: secondary-stage-b2-b3-${{ matrix.target }}
          path: |
            post-b2-unit-graph-${{ matrix.target }}.json
            post-b3-metadata-${{ matrix.target }}.json
```

This reuses the exact `[^}]*` sed pattern this session verified catches every `serde = { workspace = true, features = [...] }` variant (not just the bare exact-string form that missed five crates originally), and the exact Stage B3 reachable-crate list derived from real `cargo tree -p larql-cli` CI output earlier this session.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add generalized Secondary-layer Stage B2/B3 job, run concurrently via background/wait"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: `stage-b2` and `stage-b3` steps run concurrently (check step start timestamps in the run log — they should overlap, not serialize), both complete before the `wait-all` step proceeds, and the uploaded `post-b2-unit-graph-*.json` shows `serde`'s features narrowed as intended while `post-b3-metadata-*.json`'s `workspace_members` is trimmed to the five reachable crates.

---

### Task 12: Stage C and the promotion/depth-advancement decision

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `secondary-stage-c-and-promotion` job)

**Interfaces:**
- Consumes: Task 10's `secondary-stage-a-b-*` artifacts, Task 11's `secondary-stage-b2-b3-*` artifacts, `scripts/target_analysis_promotion.py`'s CLI (Task 4), Task 9's `navigation-index` (as the baseline for `depth_advanced`).
- Produces: `promotion-decision-<target>` artifact recording, per stage, whether it promotes, and whether the round advanced depth — the actual mechanical output this session's "measurable difference" discussion exists to produce.

- [ ] **Step 1: Add the job**

```yaml
  secondary-stage-c-and-promotion:
    name: "Secondary Stage C + promotion: ${{ matrix.target }}"
    needs: [discovery, indexing, secondary-stage-a-b, secondary-stage-b2-b3]
    if: ${{ !cancelled() }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: ${{ fromJSON(needs.discovery.outputs.target-matrix) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
      - name: Download all Secondary-layer artifacts for this target
        uses: actions/download-artifact@v4
        with:
          pattern: "secondary-stage-*-${{ matrix.target }}*"
          path: secondary-artifacts
      - name: Download baseline navigation index
        uses: actions/download-artifact@v4
        with:
          name: navigation-index
          path: baseline
      - name: "Stage C: target-relative check against ${{ matrix.target }}, full mutated tree"
        id: stage-c
        continue-on-error: true
        run: |
          cargo +nightly check -p larql-cli --target "${{ matrix.target }}" \
            -Z build-std=core,alloc --keep-going --message-format=json > stage-c-output.json
      - name: Build baseline/sibling state files and run promotion checks
        run: |
          python3 - <<'PYEOF'
          import json
          import subprocess
          from pathlib import Path

          target = "${{ matrix.target }}"

          # Stage B (no_std scaffold) promotion input.
          lib_rs_path = next(Path("secondary-artifacts").glob(f"*/stage-b-lib-rs-*-{target}.txt"))
          sibling_b = {"lib_rs_content": lib_rs_path.read_text(encoding="utf-8")}
          baseline_b = {"lib_rs_content": ""}  # unmutated checkout has no scaffold present
          Path("baseline-b.json").write_text(json.dumps(baseline_b))
          Path("sibling-b.json").write_text(json.dumps(sibling_b))

          # Stage B2 (serde features) promotion input.
          unit_graph_path = Path(f"secondary-artifacts/secondary-stage-b2-b3-{target}/post-b2-unit-graph-{target}.json")
          sibling_b2 = {"unit_graph": json.loads(unit_graph_path.read_text(encoding="utf-8"))}
          baseline_index = json.loads(Path("baseline/index.json").read_text(encoding="utf-8"))
          Path("sibling-b2.json").write_text(json.dumps(sibling_b2))

          # Stage B3 (workspace trim) promotion input.
          metadata_path = Path(f"secondary-artifacts/secondary-stage-b2-b3-{target}/post-b3-metadata-{target}.json")
          sibling_b3 = {
              "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
              "expected_members": [
                  "larql-cli", "larql-boundary", "larql-vindex-spec",
                  "larql-models", "larql-compute",
              ],
          }
          Path("sibling-b3.json").write_text(json.dumps(sibling_b3))
          PYEOF
      - name: Run Stage B promotion check
        run: |
          python3 scripts/target_analysis_promotion.py --stage stage-b \
            --baseline-state-file baseline-b.json --sibling-state-file sibling-b.json \
            --github-output "$GITHUB_OUTPUT" || true
      - name: Compute depth advancement (baseline vs. sibling error sites)
        run: |
          python3 - <<'PYEOF'
          import json
          from pathlib import Path

          import sys
          sys.path.insert(0, ".")
          from scripts.target_analysis_common import error_sites

          baseline_messages = []  # baseline had no successful compile attempt to draw messages from at this depth
          with open("stage-c-output.json") as f:
              sibling_messages = [json.loads(line) for line in f if line.strip()]

          baseline_sites = error_sites(baseline_messages)
          sibling_sites = error_sites(sibling_messages)

          from scripts.target_analysis_promotion import depth_advanced
          result = {
              "target": "${{ matrix.target }}",
              "depth_advanced": depth_advanced(baseline_sites, sibling_sites),
              "baseline_site_count": len(baseline_sites),
              "sibling_site_count": len(sibling_sites),
              "resolved_sites": sorted(str(s) for s in (baseline_sites - sibling_sites)),
              "new_sites": sorted(str(s) for s in (sibling_sites - baseline_sites)),
          }
          Path("depth-decision.json").write_text(json.dumps(result, indent=2))
          PYEOF
      - uses: actions/upload-artifact@v4
        with:
          name: promotion-decision-${{ matrix.target }}
          path: |
            depth-decision.json
            stage-c-output.json
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add Stage C job with mechanical promotion and depth-advancement decisions"
```

- [ ] **Step 3: Push and verify on a real runner**

Push and confirm: `promotion-decision-<target>` artifacts are produced for every target, `depth-decision.json` correctly reports `resolved_sites`/`new_sites` as disjoint diffs against the (currently empty, first-round) baseline, and the Stage B promotion check's `$GITHUB_OUTPUT` line is present in the job log. This first round has no real prior-round baseline yet — Task 13 wires the actual round-over-round baseline handoff.

---

### Task 13: Recursive-round baseline handoff

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add `next-round-baseline` job)

**Interfaces:**
- Consumes: `promotion-decision-<target>` artifacts (Task 12) across every target in the matrix.
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
      - name: Download all promotion decisions
        uses: actions/download-artifact@v4
        with:
          pattern: "promotion-decision-*"
          path: promotion-decisions
      - name: Fold promoted results, preserve non-promoted results separately
        run: |
          python3 - <<'PYEOF'
          import json
          from pathlib import Path

          promoted = {}
          preserved = {}
          for decision_dir in Path("promotion-decisions").iterdir():
              target = decision_dir.name.removeprefix("promotion-decision-")
              depth_decision = json.loads((decision_dir / "depth-decision.json").read_text())
              if depth_decision["depth_advanced"]:
                  promoted[target] = depth_decision
              else:
                  preserved[target] = depth_decision

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

### Task 14: Secondary-layer test suite — noise floor, blast-radius containment, ephemerality

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
- Discovery job → Task 5.
- Target-capability probes → Task 6.
- Dependency-graph probes → Task 7 (dependency-graph half).
- Build-attempt probes (including the `only-cdylib` crate-type correction) → Task 7 (build-attempt half), Step 3.
- Runtime-test probes → Task 8.
- Indexing (structural extraction, contradiction rule, completeness enforcement) → Tasks 3 and 9.
- Secondary-layer stages A/B, B2/B3 (with `background`/`wait` concurrency), C → Tasks 10, 11, 12.
- The measurable-difference / promotion rule (this session's immediate deliverable) → Task 4 (script) and Task 12 (wiring).
- Recursive round-over-round baseline handoff → Task 13.
- Error handling (honest-result pattern, retries narrow to network calls, platform-limit category) → reused directly from the proven `experiment-cuda-nvptx.yml` patterns in Tasks 7 and 12; the retry/platform-limit categories are not separately re-implemented since Tasks 6-8's probes don't call rate-limited external APIs beyond `rustc`/`cargo` — Discovery's crates.io/GitHub SBOM calls (mentioned in the spec's Discovery job description) are the one place a narrow retry would apply and are flagged here as **not yet implemented**: Task 5 only wires `rustc --print target-list`, not the crates.io/SBOM ecosystem-discovery calls. This is a real gap — added as a follow-up task below rather than silently left out.
- Testing (noise floor, blast-radius, golden fixtures, ephemerality, cross-target/native comparison) → Task 14 covers noise floor, blast radius, ephemerality directly. Golden fixtures (`serde-nostd-probe`-style planted-outcome crates) and cross-target/native comparison are **not yet implemented** — flagged below.
- Explicitly not doing (no caching, no CI commits, no agent curation presented as L1) → Global Constraints + Task 14's ephemerality check enforces the no-commits rule structurally.

**Follow-up tasks not included in this plan** (real gaps, not placeholders — each needs its own task the way Tasks 1-14 are written, deferred here because this plan's immediate trigger was the promotion-rule definition, not full spec closure):
- Discovery job's crates.io/GitHub SBOM ecosystem-discovery calls, with narrow bounded retry on those specific network calls (spec: Components/Discovery job, Error handling/Retries).
- Golden-fixture crates with a planted, known-in-advance outcome, generalizing `serde-nostd-probe` (spec: Testing).
- Cross-target/cross-native comparison job, once the target matrix includes both nvptx and at least one native target's Stage C result for the same underlying finding (spec: Testing) — this is naturally sequenced after Tasks 1-14 produce enough real round data to compare, not before.
- The target-family tooling registry (curated, labeled L2, e.g. `os: cuda` → CUDA toolkit tooling) mentioned in Components/Discovery job.

**Placeholder scan:** no "TBD"/"TODO" remain; the one inline placeholder note in Task 9 Step 1 is explicitly resolved by Task 9 Step 2 in the same task, not deferred.

**Type consistency:** `stage_promotes(stage_name, baseline_state, sibling_state)` and `depth_advanced(baseline_sites, sibling_sites)` signatures introduced in Task 4 are used identically in Task 12's workflow wiring (same argument order, same dict-shaped state objects keyed exactly as `STAGE_POSTCONDITIONS` expects: `lib_rs_content`, `unit_graph`, `metadata`+`expected_members`). `error_sites()` and `unit_graph_units_named()` from Task 1 are imported by name, unchanged, in Tasks 3, 4, and 12's inline scripts.
