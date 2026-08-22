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
