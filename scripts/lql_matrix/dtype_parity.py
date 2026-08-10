#!/usr/bin/env python3
"""
Compare dtype values between paired CLI and LQL extract legs.

This script reads descriptor files from matrix results and checks whether
the known dtype divergence (CLI f16 vs LQL f32) is captured and surfaced.

Design: matches legs by (model_id, level) across the (native, lql) operations.
Reports findings per pair, with no judgment layer — just the observed dtypes.

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

_LEG_RE = re.compile(r"descriptor-(.+?)\.json$")


def load_descriptor(path):
    """Load a descriptor JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def extract_leg_info(filename):
    """Extract model, op, and level from a descriptor filename.

    Examples:
    - descriptor-smol135.native.inference.json → ("smol135", "native", "inference")
    - descriptor-qwen05.lql.default.json → ("qwen05", "lql", "default")
    """
    m = _LEG_RE.search(filename)
    if not m:
        return None

    name = m.group(1)
    parts = name.split(".")
    if len(parts) >= 3:
        model = parts[0]
        op = parts[1]
        level = ".".join(parts[2:])
        return (model, op, level)
    return None


def find_descriptors(glob_pattern):
    """Find all descriptor JSON files matching the glob."""
    descriptors = {}

    for desc_file in sorted(glob.glob(glob_pattern)):
        path = Path(desc_file)
        info = extract_leg_info(path.name)
        if info:
            key = f"{info[0]}.{info[1]}.{info[2]}"
            descriptors[key] = {
                "path": desc_file,
                "model": info[0],
                "op": info[1],
                "level": info[2],
            }

    return descriptors


def find_pairs(descriptors):
    """Find paired native (CLI) vs lql legs.

    A pair is (model, level) with both .native and .lql variants present.
    """
    pairs = {}

    for key, desc in descriptors.items():
        model = desc["model"]
        level = desc["level"]

        cli_key = f"{model}.native.{level}"
        lql_key = f"{model}.lql.{level}"

        if cli_key in descriptors and lql_key in descriptors:
            pair_key = f"{model}.{level}"
            if pair_key not in pairs:
                pairs[pair_key] = {"native": None, "lql": None}
            pairs[pair_key][desc["op"]] = desc

    return {k: v for k, v in pairs.items() if v["native"] and v["lql"]}


def main():
    args = sys.argv[1:]
    glob_pattern = args[0] if args else "artifacts/results-*/descriptor-*.json"
    out_file = args[1] if len(args) > 1 else "dtype-parity.json"

    # Find all descriptors
    descriptors = find_descriptors(glob_pattern)

    # Find paired legs
    pairs = find_pairs(descriptors)

    # Load and analyze each pair
    results = []

    for pair_key in sorted(pairs.keys()):
        pair = pairs[pair_key]

        cli_desc = load_descriptor(pair["native"]["path"])
        lql_desc = load_descriptor(pair["lql"]["path"])

        cli_dtype = cli_desc.get("dtype")
        lql_dtype = lql_desc.get("dtype")

        result = {
            "pair": pair_key,
            "cli": {
                "dtype": cli_dtype,
                "produced": cli_desc.get("produced"),
                "error": cli_desc.get("error"),
            },
            "lql": {
                "dtype": lql_dtype,
                "produced": lql_desc.get("produced"),
                "error": lql_desc.get("error"),
            },
        }
        results.append(result)

    # Write output
    Path(out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"dtype-parity: {len(results)} pairs")
    for r in results:
        cli_dt = r["cli"]["dtype"]
        lql_dt = r["lql"]["dtype"]
        print(f"  {r['pair']}: cli={cli_dt} lql={lql_dt}")


if __name__ == "__main__":
    main()
