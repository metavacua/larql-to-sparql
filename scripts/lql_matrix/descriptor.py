#!/usr/bin/env python3
"""Read a produced vindex's index.json and emit its structural descriptor as JSON,
compared against the leg's expected quant. Descriptive only — records what the
produce recipe actually yielded (family/dtype/quant/has_model_weights, and whether
a bitnet ternary layout was retained), so extraction/conversion breakage or silent
dequantization is observed on the current binary rather than assumed.

Generation-aware: index.json's `version` field is the sole discriminator (per
crates/larql-vindex/src/format/generation.rs) — 1-2 is VINDEX2 (the v1 manifest,
larql-vindex-spec), 3-4 is VINDEX3 (crates/larql-vindex/src/format/vindex3/index.rs's
Vindex3Index — a catalogue of `representations`, not a single quant/dtype scalar).

Usage: descriptor.py <vindex_dir> <leg_name> <expect_quant>
"""
import json
import os
import sys

V3_MIN_VERSION = 3


def describe_v2(idx, expect):
    d = {
        "generation": "v2",
        "family": idx.get("family"),
        "dtype": idx.get("dtype"),
        "quant": idx.get("quant"),
        "has_model_weights": idx.get("has_model_weights"),
        "num_layers": idx.get("num_layers"),
        "hidden_size": idx.get("hidden_size"),
        "bitnet_layout": idx.get("bitnet_layout") is not None,
    }
    # ternary is signalled by the bitnet_layout sidecar, not the quant field
    observed = "ternary" if d["bitnet_layout"] else d.get("quant")
    d["observed_quant"] = observed
    d["quant_match"] = (observed == expect)
    return d


def describe_v3(idx, expect):
    # Vindex3Index has no single quant/dtype scalar — quant is per-representation
    # (RepresentationEntry.encoding). Report the distinct encodings observed;
    # this matrix never requests --quant under --generation v3 (CLI refuses the
    # combination today), so `quant_match` here only asserts "no representation
    # claims an encoding" is false, i.e. something was actually produced — it is
    # NOT yet a real match assertion the way V2's is. Extend once V3 quantization
    # is reachable from this workflow.
    reps = idx.get("representations") or {}
    encodings = sorted({r.get("encoding") for r in reps.values() if r.get("encoding")})
    return {
        "generation": "v3",
        "family": idx.get("family"),
        "dtype": None,
        "quant": None,
        "has_model_weights": len(reps) > 0,
        "num_layers": idx.get("num_layers"),
        "hidden_size": idx.get("hidden_size"),
        "bitnet_layout": False,
        "observed_quant": "+".join(encodings) if encodings else None,
        "quant_match": len(reps) > 0,
        "num_representations": len(reps),
        "num_profiles": len(idx.get("profiles") or []),
    }


def main():
    vindex, name, expect = sys.argv[1], sys.argv[2], sys.argv[3]
    d = {"name": name, "expect_quant": expect, "produced": os.path.isdir(vindex)}
    idx_path = os.path.join(vindex, "index.json")
    try:
        idx = json.load(open(idx_path, encoding="utf-8"))
        version = idx.get("version")
        is_v3 = isinstance(version, int) and version >= V3_MIN_VERSION
        d.update(describe_v3(idx, expect) if is_v3 else describe_v2(idx, expect))
    except Exception as e:  # missing/malformed index.json is itself a finding
        d.update({"error": str(e), "observed_quant": None, "quant_match": False})
    print(json.dumps(d))


if __name__ == "__main__":
    main()
