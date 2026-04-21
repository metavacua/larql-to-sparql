#!/usr/bin/env python3
"""
Memory isolation benchmark.

Checks that each shard server holds only the memory proportional to
its owned layers. Reports RSS per shard, expected %, actual %, and
the VSZ/RSS ratio (low ratio confirms demand-paging is working).

Usage:
    python3 scripts/grid/bench_memory.py [--router http://127.0.0.1:9090]
"""
import argparse
import subprocess
import sys
import os

try:
    import psutil
except ImportError:
    sys.exit("pip install psutil")

# Topology matching start.sh
SHARD_MAP = {
    8080: (0, 11),
    8081: (0, 11),
    8082: (0, 11),
    8083: (12, 23),
    8084: (12, 23),
    8085: (12, 23),
    8086: (24, 33),
    8087: (24, 33),
    8088: (24, 33),
    8089: (24, 33),
}
TOTAL_LAYERS = 34


def pid_for_port(port: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"], text=True
        ).strip()
        return int(out.splitlines()[0]) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def rss_mb(pid: int) -> float:
    try:
        p = psutil.Process(pid)
        info = p.memory_info()
        return info.rss / 1024 / 1024
    except psutil.NoSuchProcess:
        return 0.0


def vsz_mb(pid: int) -> float:
    try:
        p = psutil.Process(pid)
        info = p.memory_info()
        return info.vms / 1024 / 1024
    except psutil.NoSuchProcess:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:9090")
    args = parser.parse_args()

    print("=" * 60)
    print(" Memory Benchmark — Layer Isolation")
    print("=" * 60)
    print()

    rows = []
    by_range: dict[tuple, list[float]] = {}

    header = f"{'Port':<6}  {'Layers':<8}  {'RSS(MB)':>8}  {'VSZ(MB)':>8}  {'RSS/VSZ%':>8}  {'Expected%':>9}  Status"
    print(header)
    print("-" * len(header))

    for port in sorted(SHARD_MAP):
        start, end = SHARD_MAP[port]
        n = end - start + 1
        pid = pid_for_port(port)

        if pid is None:
            print(f"{port:<6}  {start}-{end:<6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>9}  NOT RUNNING")
            continue

        rss = rss_mb(pid)
        vsz = vsz_mb(pid)
        rss_vsz_pct = (rss / vsz * 100) if vsz > 0 else 0
        expected_pct = n / TOTAL_LAYERS * 100

        key = (start, end)
        by_range.setdefault(key, []).append(rss)

        print(f"{port:<6}  {start}-{end:<6}  {rss:>8.0f}  {vsz:>8.0f}  {rss_vsz_pct:>8.1f}  {expected_pct:>9.1f}  OK")
        rows.append((port, start, end, n, rss, vsz))

    if not rows:
        print("\nNo shards found. Run ./scripts/grid/start.sh first.")
        return

    print()
    print("--- By layer range (average across replicas) ---")

    # Estimate full-model RSS: largest range shard × (34 / range_size)
    full_model_estimates = []
    for (start, end), rss_list in sorted(by_range.items()):
        n = end - start + 1
        avg_rss = sum(rss_list) / len(rss_list)
        full_est = avg_rss * TOTAL_LAYERS / n
        full_model_estimates.append(full_est)
        expected_pct = n / TOTAL_LAYERS * 100
        print(f"  layers {start}-{end} ({n} layers, {len(rss_list)} replicas): "
              f"avg RSS={avg_rss:.0f}MB  expected≈{expected_pct:.0f}%  "
              f"full-model est≈{full_est:.0f}MB")

    print()
    full_model_est = sum(full_model_estimates) / len(full_model_estimates)
    total_rss = sum(r[4] for r in rows)
    print(f"  Total RSS across {len(rows)} shards: {total_rss:.0f}MB")
    print(f"  Single full-model estimate:          {full_model_est:.0f}MB")
    print(f"  Grid overhead vs 1 full-model:       {total_rss / full_model_est:.1f}×")
    print()
    print("  Note: overhead >1 because replicas share layers.")
    print(f"  With 3× redundancy, expect ~3× RSS for covered layers.")
    print()

    # VSZ/RSS sanity: confirm demand paging
    avg_rss_vsz = sum(r[4] / r[5] * 100 for r in rows if r[5] > 0) / len(rows)
    print(f"--- Demand-paging check ---")
    print(f"  Average RSS/VSZ across all shards: {avg_rss_vsz:.1f}%")
    if avg_rss_vsz < 30:
        print(f"  ✓ Low ratio confirms mmap pages outside owned layers are not resident.")
    else:
        print(f"  ⚠ Higher than expected — may indicate eager page faults.")


if __name__ == "__main__":
    main()
