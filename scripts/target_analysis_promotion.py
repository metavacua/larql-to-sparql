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

from scripts.target_analysis_common import load_json, unit_graph_units_named


# The exact feature set Stage B2 (secondary-mutate job) patches every
# `serde = { workspace = true }` dependency line to -- single source of
# truth for both the mutation's sed replacement and this postcondition.
STAGE_B2_SERDE_FEATURES = ("alloc", "derive")


def serde_features_ok(unit_graph: dict[str, Any]) -> bool:
    units = unit_graph_units_named(unit_graph, "serde")
    if not units:
        return False
    expected = set(STAGE_B2_SERDE_FEATURES)
    return all(set(unit.get("features", [])) == expected for unit in units)


def workspace_members_ok(metadata: dict[str, Any], expected_members: list[str]) -> bool:
    id_to_name = {pkg["id"]: pkg["name"] for pkg in metadata.get("packages", [])}
    actual_names = {id_to_name[member] for member in metadata.get("workspace_members", [])}
    return actual_names == set(expected_members)


def no_std_scaffold_ok(lib_rs_content: str) -> bool:
    return "#![no_std]" in lib_rs_content and "extern crate alloc;" in lib_rs_content


def _leading_block_end(text: str) -> int:
    """The line index immediately after any leading //! doc comments, #![...]
    inner attributes, and blank lines -- the correct insertion point for a
    module-level attribute or extern crate declaration, so it lands before
    any real code but after the crate's own existing doc/attributes."""
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
            insert_at = i + 1
        else:
            break
    return insert_at


def insert_alloc_extern_crate(text: str) -> str:
    """Runs before Stage A's clippy --fix. A plain, still-std-linked crate
    can declare `extern crate alloc;` without `#![no_std]` -- alloc ships in
    every target's sysroot regardless -- so std:: references Stage A does
    NOT touch keep working exactly as before, while std::X paths clippy
    suggests rewriting to alloc::X now have something to resolve against.
    Real CI evidence (2026-08-21): without this, Stage A's fix for any
    std::->alloc::-only path (BTreeMap, BTreeSet, Arc, VecDeque) fails to
    compile once applied and clippy's own safety check silently reverts it
    -- confirmed for 4 of 5 mutated crates."""
    lines = text.splitlines(keepends=True)
    lines.insert(_leading_block_end(text), "extern crate alloc;\n")
    return "".join(lines)


def insert_no_std_attribute(text: str) -> str:
    """Runs after Stage A. Only #![no_std] remains to insert -- extern
    crate alloc; was already added by insert_alloc_extern_crate() before
    Stage A ran."""
    lines = text.splitlines(keepends=True)
    lines.insert(_leading_block_end(text), "#![no_std]\n")
    return "".join(lines)


def insert_no_std_scaffold(text: str) -> str:
    """DEPRECATED -- kept only because `secondary-layer-self-test`'s
    "Blast-radius containment" step (out of this task's scope; Task 11's,
    per the 2026-08-21-stage-c-measurement-validity plan) still imports and
    calls this name directly, and that step is explicitly left untouched by
    this task. Provably equivalent to the old combined-scaffold single pass
    for ANY input: `extern crate alloc;` never matches the `//!`/`#![`/blank
    classification `_leading_block_end` scans for, so it doesn't shift where
    the scan halts -- composing insert_alloc_extern_crate then
    insert_no_std_attribute always lands #![no_std] immediately above
    extern crate alloc;, byte-for-byte matching this function's old
    single-pass output. Task 11 removes this function and its one remaining
    call site together."""
    return insert_no_std_attribute(insert_alloc_extern_crate(text))


# The spec's required positive control (Testing section): a crate with a
# known-in-advance outcome, run through the real Stage A->C pipeline, so a
# broken measurement is distinguishable from a genuine experimental result.
GOLDEN_FIXTURE_CRATE = "larql-nostd-canary"

# The crates the Secondary-layer mutation job (Stage A/B) actually produces a
# baseline/sibling lib.rs pair for. larql-cli is deliberately absent: it is
# bin-only (no src/lib.rs), so there is nothing for Stage A/B to mutate.
MUTATED_LIBRARY_CRATES = (
    "larql-boundary",
    "larql-vindex-spec",
    "larql-models",
    "larql-compute",
    GOLDEN_FIXTURE_CRATE,
)

# larql-compute is the chosen Stage-B representative: it has the largest
# leading doc comment of the four mutated crates, the shape that broke the
# scaffold-insertion script's first, naive version, making it the sentinel
# most likely to actually catch a real regression rather than pass vacuously.
STAGE_B_REPRESENTATIVE_CRATE = "larql-compute"

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


def stage_b_lib_rs_filenames(crate: str = STAGE_B_REPRESENTATIVE_CRATE) -> tuple[str, str]:
    if crate not in MUTATED_LIBRARY_CRATES:
        raise ValueError(
            f"{crate!r} is not one of the crates the Secondary-layer mutation "
            f"job produces lib.rs baseline/sibling pairs for {MUTATED_LIBRARY_CRATES}. "
            "larql-cli specifically is bin-only (no src/lib.rs) and is never mutated."
        )
    return f"baseline-lib-rs-{crate}.txt", f"sibling-lib-rs-{crate}.txt"


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
    args = parser.parse_args()

    baseline_state = load_json(args.baseline_state_file)
    sibling_state = load_json(args.sibling_state_file)
    promotes = stage_promotes(args.stage, baseline_state, sibling_state)

    print(json.dumps({"stage": args.stage, "promotes": promotes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
