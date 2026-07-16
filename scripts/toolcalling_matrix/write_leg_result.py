#!/usr/bin/env python3
"""Shared leg-result writer for run_patch_chain.sh / run_patch_ensemble.sh /
run_introspect_patch.sh, which previously each hand-rolled their own inline
`python3 -c "json.dump({...})"` tail with the same leg_id/approach_id/outcome/detail
shape plus one script-specific extra field. Factored out so the JSON-serialization
(including safe string escaping, which shell-side triple-quote interpolation did
not actually guarantee) lives in one place.

Usage: write_leg_result.py --leg-id ID --approach-id ID --outcome OUTCOME
                            --detail TEXT [--extra '{"k": v, ...}'] --out PATH
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-id", required=True)
    ap.add_argument("--approach-id", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--detail", required=True)
    ap.add_argument("--extra", default="{}", help="JSON object string merged into the result")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {
        "leg_id": args.leg_id,
        "approach_id": args.approach_id,
        "outcome": args.outcome,
        "detail": args.detail,
    }
    result.update(json.loads(args.extra))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
