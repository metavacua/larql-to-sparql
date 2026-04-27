#!/usr/bin/env python3
"""
Latency benchmark: single-layer, batched fan-out, same-shard batches.
Reports mean / p50 / p95 / p99 for each pattern.

Usage:
    python3 scripts/grid/bench_latency.py [--router URL] [--reps N] [--warmup N]
"""
import argparse
import json
import statistics
import time

try:
    import httpx
except ImportError:
    import sys; sys.exit("pip install httpx")

HIDDEN = 2560
RESIDUAL = [0.01] * HIDDEN


def measure(client: httpx.Client, url: str, body: dict, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        client.post(url, json=body)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = client.post(url, json=body)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            times.append(elapsed_ms)
        else:
            print(f"  WARN: status {r.status_code}: {r.text[:80]}")
    return times


def stats(times: list[float]) -> str:
    if not times:
        return "NO DATA"
    s = sorted(times)
    n = len(s)
    mean = statistics.mean(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]
    return f"mean={mean:6.1f}ms  p50={p50:5.1f}ms  p95={p95:5.1f}ms  p99={p99:5.1f}ms  n={n}"


def row(label: str, times: list[float]):
    print(f"  {label:<48}  {stats(times)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:9090")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    url = f"{args.router}/v1/walk-ffn"

    print("=" * 70)
    print(" Latency Benchmark — larql grid")
    print("=" * 70)
    print(f"Router: {args.router}  reps={args.reps}  warmup={args.warmup}")
    print()

    with httpx.Client(timeout=30.0) as client:
        # Quick health check
        try:
            r = client.get(f"{args.router}/v1/health")
            assert r.status_code == 200
        except Exception as e:
            print(f"Router unreachable: {e}")
            return

        single = lambda layer: {"layer": layer, "residual": RESIDUAL}
        batched = lambda layers: {"layers": layers, "residual": RESIDUAL}

        print("--- Single-layer (one shard per request) ---")
        for layer, note in [(0, "range A first"), (5, "range A mid"), (11, "range A last"),
                            (12, "range B first"), (20, "range B mid"),
                            (24, "range C first"), (33, "range C last")]:
            t = measure(client, url, single(layer), args.reps, args.warmup)
            row(f"layer {layer:>2}  ({note})", t)

        print()
        print("--- Batched fan-out (parallel across shards) ---")
        for layers, note in [([5, 20], "2 shards"), ([5, 20, 30], "3 shards"),
                              ([0, 5, 11, 12, 20, 24, 33], "all shards, 7 layers")]:
            t = measure(client, url, batched(layers), args.reps, args.warmup)
            row(f"layers {layers}  ({note})", t)

        print()
        print("--- Same-shard batches (no fan-out, sequential within shard) ---")
        for layers, note in [([0, 1, 2, 3], "4 layers, range A"), ([12, 13, 14, 15], "4 layers, range B"),
                              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "all 12 layers, range A")]:
            t = measure(client, url, batched(layers), args.reps, args.warmup)
            row(f"layers {layers[:4]}{'...' if len(layers) > 4 else ''}  ({note})", t)

        print()
        print("--- Router overhead ---")
        health_times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            client.get(f"{args.router}/v1/health")
            health_times.append((time.perf_counter() - t0) * 1000)
        row("GET /v1/health  (no FFN, baseline)", health_times)

        # latency_ms from response vs wall-clock comparison
        print()
        print("--- FFN compute vs routing overhead ---")
        resp = client.post(url, json=single(20))
        if resp.status_code == 200:
            d = resp.json()
            ffn_ms = d.get("latency_ms", 0)
            t0 = time.perf_counter()
            resp2 = client.post(url, json=single(20))
            wall_ms = (time.perf_counter() - t0) * 1000
            ffn_ms2 = resp2.json().get("latency_ms", 0)
            overhead_ms = wall_ms - ffn_ms2
            print(f"  Wall-clock: {wall_ms:.1f}ms  FFN reported: {ffn_ms2:.1f}ms  "
                  f"Routing overhead: {overhead_ms:.1f}ms")


if __name__ == "__main__":
    main()
