#!/usr/bin/env python3
"""Per-(layer, position) delta table across a span boundary.

`larql shannon layer-diff` collapses each plane to one number per layer,
which is the right summary for whole-trace parity and the wrong one for a
boundary experiment: a sliding window that first excludes a key at
position P changes *only* positions >= P, and a per-layer maximum cannot
show that.

    python scripts/span_boundary_report.py --a DUMP_A --b DUMP_B \\
        --positions 2046 2047 2048 2049 --layers 7 8 9

The fingerprint a correct span mutation produces has two coordinates. If
layer L is switched from `Sliding(w)` to `Full`:

    layer < L                  unchanged at every position
    layer = L, position <  P   unchanged
    layer = L, position >= P   diverges
    layer > L, position <  P   still unchanged
    layer > L, position >= P   may inherit the divergence

Earlier positions stay clean at *every* depth because attention is
causal: a changed key at a later position can never reach an earlier
query. That asymmetry is the proof — "the run changed" is not.

Reported per cell is relative RMS, `||a - b|| / ||b||`, over that
position's hidden vector. `max_abs` is deliberately absent: it is an
extreme-value statistic, so it inflates with the number of elements
compared and is not comparable across runs of different length.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MANIFEST_NAME = "layer_dump.json"

# Below this, two f32 traces of the same program are indistinguishable
# from reassociation noise. Well under any real masking effect.
SAME_REL_RMS = 1e-6


def load(dump: Path) -> dict:
    manifest = json.loads((dump / MANIFEST_NAME).read_text())
    return {"dir": dump, "manifest": manifest}


def plane(side: dict, index: int) -> np.ndarray:
    m = side["manifest"]
    name = m["planes"][index]
    raw = np.fromfile(side["dir"] / name, dtype="<f4")
    return raw.reshape(m["seq_len"], m["hidden_size"]).astype(np.float64)


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return 0.0 if np.linalg.norm(a) == 0.0 else float("inf")
    return float(np.linalg.norm(a - b) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--positions", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layer indices to report; default every layer")
    ap.add_argument("--threshold", type=float, default=SAME_REL_RMS)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    ma, mb = a["manifest"], b["manifest"]
    if ma["token_ids"] != mb["token_ids"]:
        raise SystemExit("the two dumps ran different token windows — refusing to compare")
    if (ma["seq_len"], ma["hidden_size"]) != (mb["seq_len"], mb["hidden_size"]):
        raise SystemExit("geometry mismatch between dumps")
    seq = ma["seq_len"]
    for p in args.positions:
        if p >= seq:
            raise SystemExit(f"position {p} is past the fixture's {seq} tokens")

    print(f"a: {ma['engine']}")
    print(f"b: {mb['engine']}")
    print(f"{seq} tokens, hidden_size {ma['hidden_size']}, "
          f"{len(ma['planes'])} planes, threshold rel_rms {args.threshold:g}")
    print()

    # Plane 0 is the embedding; plane i+1 is the output of layer i.
    layers = args.layers if args.layers is not None else range(ma["num_layers"])
    header = "layer".ljust(8) + "".join(f"pos {p:>6}      ".rjust(18) for p in args.positions)
    print(header)
    print("-" * len(header))
    for layer in layers:
        pa, pb = plane(a, layer + 1), plane(b, layer + 1)
        cells = []
        for p in args.positions:
            value = rel_rms(pa[p], pb[p])
            verdict = "same" if value <= args.threshold else "DIVERGE"
            cells.append(f"{value:>10.3e} {verdict:<7}")
        print(f"{layer:<8}" + "".join(c.rjust(18) for c in cells))


if __name__ == "__main__":
    main()
