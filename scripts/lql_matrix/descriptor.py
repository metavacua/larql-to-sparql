#!/usr/bin/env python3
"""Read a produced vindex's index.json and emit what it structurally IS.

Reporting only. It records the fields the produce recipe actually yielded —
family, dtype, quant, layer/hidden geometry, whether a bitnet ternary layout
sidecar is present — and stops there.

It does NOT compare them against an expectation. The old version carried
`expect_quant` and emitted `quant_match`, which made this a verdict: a
`quant_match: false` reads as a failure when it may equally be a stale
expectation in the leg table. Whether an observed dtype/quant is right is a
judgement about the data, and judgement happens outside the harness.

Usage: descriptor.py <vindex_dir> <leg_name>
"""
import json
import os
import sys


def main():
    vindex, name = sys.argv[1], sys.argv[2]
    d = {"name": name, "produced": os.path.isdir(vindex)}
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
            # ternary is signalled by the sidecar, not the quant field — both
            # are reported; which one "counts" is not this script's call.
            "bitnet_layout": idx.get("bitnet_layout") is not None,
        })
    except Exception as e:  # a missing/malformed index.json is itself the report
        d["error"] = str(e)
    print(json.dumps(d))


if __name__ == "__main__":
    main()
