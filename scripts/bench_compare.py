#!/usr/bin/env python3
"""
GT10 — Bench JSON comparator (ADR-0012 §Layer 3).

Compares a current bench run against a stored baseline. Fails with exit
code 1 if tok/s dropped by more than tok_per_s_threshold or p99 rose by
more than p99_threshold.

Usage:
    python3 scripts/bench_compare.py \
        --baseline bench/baselines/grid-gemma3-4b-q4k.json \
        --current  /tmp/current.json \
        --tok-per-s-threshold 0.05 \
        --p99-threshold 0.10
"""

import argparse
import json
import sys
from typing import Optional


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_row(results: list, backend_prefix: str) -> Optional[dict]:
    """Return the first row whose backend starts with backend_prefix."""
    for row in results:
        if row.get("backend", "").startswith(backend_prefix):
            return row
    return None


def check_regression(
    baseline_rows: list,
    current_rows: list,
    tok_threshold: float,
    p99_threshold: float,
) -> list[str]:
    """Return a list of regression messages. Empty = pass."""
    failures = []
    for base_row in baseline_rows:
        backend = base_row["backend"]
        cur_row = next(
            (r for r in current_rows if r["backend"] == backend), None
        )
        if cur_row is None:
            print(f"  WARN: backend '{backend}' not in current run — skipping")
            continue

        base_tps = base_row.get("tok_per_s", 0.0)
        cur_tps = cur_row.get("tok_per_s", 0.0)
        if base_tps > 0 and cur_tps > 0:
            drop = (base_tps - cur_tps) / base_tps
            if drop > tok_threshold:
                failures.append(
                    f"  FAIL [{backend}] tok/s dropped {drop*100:.1f}% "
                    f"({base_tps:.1f} → {cur_tps:.1f}; threshold {tok_threshold*100:.0f}%)"
                )
            else:
                print(
                    f"  OK   [{backend}] tok/s: {base_tps:.1f} → {cur_tps:.1f} "
                    f"({'+' if cur_tps >= base_tps else ''}{(cur_tps-base_tps)/base_tps*100:.1f}%)"
                )

        base_p99 = (base_row.get("ms_per_tok") or {}).get("p99", 0.0)
        cur_p99 = (cur_row.get("ms_per_tok") or {}).get("p99", 0.0)
        if base_p99 > 0 and cur_p99 > 0:
            rise = (cur_p99 - base_p99) / base_p99
            if rise > p99_threshold:
                failures.append(
                    f"  FAIL [{backend}] p99 rose {rise*100:.1f}% "
                    f"({base_p99:.1f}ms → {cur_p99:.1f}ms; threshold {p99_threshold*100:.0f}%)"
                )
            else:
                print(
                    f"  OK   [{backend}] p99:    {base_p99:.1f}ms → {cur_p99:.1f}ms "
                    f"({'+' if cur_p99 > base_p99 else ''}{(cur_p99-base_p99)/base_p99*100:.1f}%)"
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare bench JSON runs")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--tok-per-s-threshold", type=float, default=0.05)
    parser.add_argument("--p99-threshold", type=float, default=0.10)
    args = parser.parse_args()

    baseline = load(args.baseline)
    current = load(args.current)

    print(f"\n[bench_compare] Baseline: {args.baseline}")
    print(f"[bench_compare] Current:  {args.current}")
    print(
        f"[bench_compare] Thresholds: tok/s -{args.tok_per_s_threshold*100:.0f}%  "
        f"p99 +{args.p99_threshold*100:.0f}%\n"
    )

    failures = check_regression(
        baseline_rows=baseline.get("results", []),
        current_rows=current.get("results", []),
        tok_threshold=args.tok_per_s_threshold,
        p99_threshold=args.p99_threshold,
    )

    if failures:
        print("\n[bench_compare] REGRESSIONS DETECTED:")
        for f in failures:
            print(f)
        print()
        return 1

    print("\n[bench_compare] All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
