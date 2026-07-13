#!/usr/bin/env python3
"""Read a produced vindex's index.json and emit its structural descriptor as JSON,
compared against the leg's expected quant. Descriptive only — records what the
produce recipe actually yielded (family/dtype/quant/has_model_weights, and whether
a bitnet ternary layout was retained), so extraction/conversion breakage or silent
dequantization is observed on the current binary rather than assumed.

Usage: descriptor.py <vindex_dir> <leg_name> <expect_quant>
"""
import json
import os
import sys


def main():
    vindex, name, expect = sys.argv[1], sys.argv[2], sys.argv[3]
    d = {"name": name, "expect_quant": expect, "produced": os.path.isdir(vindex)}
    idx_path = os.path.join(vindex, "index.json")
    try:
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        d.update({
            "family": idx.get("family"),
            "dtype": idx.get("dtype"),
            "quant": idx.get("quant"),
            "has_model_weights": idx.get("has_model_weights"),
            "num_layers": idx.get("num_layers"),
            "hidden_size": idx.get("hidden_size"),
            "bitnet_layout": idx.get("bitnet_layout") is not None,
        })
        # ternary is signalled by the bitnet_layout sidecar, not the quant field
        observed = "ternary" if d["bitnet_layout"] else d.get("quant")
        d["observed_quant"] = observed
        d["quant_match"] = (observed == expect)
    except Exception as e:  # missing/malformed index.json is itself a finding
        d.update({"error": str(e), "observed_quant": None, "quant_match": False})
    print(json.dumps(d))


if __name__ == "__main__":
    main()
