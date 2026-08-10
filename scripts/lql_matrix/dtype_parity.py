#!/usr/bin/env python3
"""
Compare dtype values between paired CLI and LQL extract legs.

This script reads descriptor files from matrix results and checks whether
the known dtype divergence (CLI f16 vs LQL f32) is captured and surfaced.

Design: matches legs by (model_id, level) across the (native, lql) operations.
Reports findings per pair, with no judgment layer — just the observed dtypes.

`dtype` is whatever descriptor.py's `dtype` field declares (index.json's own
self-report), not a byte-level observation of the produced weight files — this
can only ever say "declared dtypes agree/disagree", not "the actual bytes
disagree".

A (model, level) with only one side present (e.g. the lql leg's extract
crashed and produced no descriptor) is not reported here — this mirrors the
prior version's behavior; see the matrix-audit for the open discussion of
whether a missing-side leg should get an explicit row instead of being
dropped.

Usage: dtype_parity.py [<descriptors_glob>] [<output_file>]

Defaults:
  glob:   artifacts/results-*/descriptor-*.json
  output: dtype-parity.json
"""
import json
import sys
import re
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_legs import _model_id  # noqa: E402  (needs the sys.path line above)

_LEG_RE = re.compile(r"descriptor-(.+?)\.json$")


def load_descriptor(path):
    """Load a descriptor JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _glob_all(pattern):
    """Flat + recursive glob, deduped, deterministic order — matches
    tensor_presence.collect()'s discovery so both scripts in the same
    inventory job see the same files regardless of how
    actions/upload-artifact laid out the tree."""
    seen = set()
    paths = []
    for p in sorted(glob.glob(pattern)) + sorted(glob.glob("**/" + pattern, recursive=True)):
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return sorted(paths)


def find_pairs(glob_pattern):
    """Group descriptor files by (model, level) and keep only groups that
    have both a native (CLI) and lql leg present.

    Examples of the filename shape being parsed:
    - descriptor-smol135.native.inference.json -> model=smol135 op=native level=inference
    - descriptor-qwen05.lql.default.json       -> model=qwen05  op=lql    level=default
    """
    groups = {}
    for desc_file in _glob_all(glob_pattern):
        m = _LEG_RE.search(Path(desc_file).name)
        if not m:
            continue
        leg_name = m.group(1)
        parts = leg_name.split(".")
        if len(parts) < 3:
            continue
        model = _model_id(leg_name)
        op = parts[1]
        level = ".".join(parts[2:])
        groups.setdefault((model, level), {})[op] = desc_file

    return {
        f"{model}.{level}": legs
        for (model, level), legs in groups.items()
        if "native" in legs and "lql" in legs
    }


def leg_summary(desc):
    """The fixed per-leg shape written into the output for both sides of a
    pair — one place to extend if a new field is needed."""
    return {
        "dtype": desc.get("dtype"),
        "produced": desc.get("produced"),
        "error": desc.get("error"),
    }


def main():
    args = sys.argv[1:]
    glob_pattern = args[0] if args else "artifacts/results-*/descriptor-*.json"
    out_file = args[1] if len(args) > 1 else "dtype-parity.json"

    pairs = find_pairs(glob_pattern)

    results = []
    for pair_key in sorted(pairs.keys()):
        legs = pairs[pair_key]
        cli_desc = load_descriptor(legs["native"])
        lql_desc = load_descriptor(legs["lql"])
        results.append({
            "pair": pair_key,
            "cli": leg_summary(cli_desc),
            "lql": leg_summary(lql_desc),
        })

    Path(out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")
    # "declared" because dtype comes from index.json's own self-report, not
    # from re-reading the produced weight bytes.
    print(f"dtype-parity (declared dtypes): {len(results)} pairs")
    for r in results:
        cli, lql = r["cli"], r["lql"]
        cli_dt = f"ERROR({cli['error']})" if cli["error"] else cli["dtype"]
        lql_dt = f"ERROR({lql['error']})" if lql["error"] else lql["dtype"]
        print(f"  {r['pair']}: cli={cli_dt} lql={lql_dt}")


if __name__ == "__main__":
    main()
