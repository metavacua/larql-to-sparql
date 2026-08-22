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

# Standing Principle 5 (the design spec's "primary, always-on validation
# mechanism for the Primary layer") specifically names this target as its
# own reference canary for unexpected_clean_std_build's contradiction
# check. Its own metadata.std is also False, so the guaranteed-failure
# short-circuit below would otherwise skip it too -- silently making
# that contradiction check permanently unreachable for the one target
# it's built to protect. Exempted here, deliberately, so it keeps
# running the real attempt SP5's contradiction check depends on.
SP5_CANARY_TARGET = "nvptx64-nvidia-cuda"


def guaranteed_std_failure(target_spec: dict[str, Any], target: str) -> bool:
    return target != SP5_CANARY_TARGET and target_spec.get("metadata", {}).get("std") is False


def skip_record(target: str) -> dict[str, Any]:
    return {
        SKIP_MARKER_KEY: True,
        "target": target,
        "skip_reason": SKIP_REASON,
    }
