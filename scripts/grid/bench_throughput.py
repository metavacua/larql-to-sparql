#!/usr/bin/env python3
"""
Throughput benchmark: concurrent requests to the grid.
Measures req/s at increasing concurrency levels using asyncio.

Usage:
    python3 scripts/grid/bench_throughput.py [--router URL] [--duration N] [--max-conc N]
"""
import argparse
import asyncio
import statistics
import time

try:
    import httpx
except ImportError:
    import sys; sys.exit("pip install httpx")

HIDDEN = 2560
RESIDUAL = [0.01] * HIDDEN


async def worker(client: httpx.AsyncClient, url: str, body: dict,
                 stop_event: asyncio.Event, results: list):
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=body, timeout=30.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                results.append(elapsed_ms)
        except Exception:
            pass


async def run_level(router: str, body: dict, concurrency: int, duration: float) -> tuple[float, float, float]:
    url = f"{router}/v1/walk-ffn"
    results: list[float] = []
    stop = asyncio.Event()

    async with httpx.AsyncClient() as client:
        # Warmup with 3 serial requests
        for _ in range(3):
            await client.post(url, json=body, timeout=30.0)

        tasks = [asyncio.create_task(worker(client, url, body, stop, results))
                 for _ in range(concurrency)]
        await asyncio.sleep(duration)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    if not results:
        return 0.0, 0.0, 0.0
    rps = len(results) / duration
    s = sorted(results)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]
    return rps, p50, p95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:9090")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds per concurrency level")
    parser.add_argument("--max-conc", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print(" Throughput Benchmark — larql grid")
    print("=" * 70)
    print(f"Router: {args.router}  duration={args.duration}s per level  max_conc={args.max_conc}")
    print()

    patterns = [
        ("Single layer=5  (range A)", {"layer": 5, "residual": RESIDUAL}),
        ("Single layer=20 (range B)", {"layer": 20, "residual": RESIDUAL}),
        ("Fan-out [5,20,30] (3 shards)", {"layers": [5, 20, 30], "residual": RESIDUAL}),
        ("Fan-out [5,20]    (2 shards)", {"layers": [5, 20], "residual": RESIDUAL}),
    ]

    conc_levels = [c for c in [1, 2, 4, 8, 16, 32] if c <= args.max_conc]

    for desc, body in patterns:
        print(f"--- {desc} ---")
        print(f"  {'conc':>5}  {'req/s':>7}  {'p50(ms)':>9}  {'p95(ms)':>9}")
        for conc in conc_levels:
            rps, p50, p95 = asyncio.run(run_level(args.router, body, conc, args.duration))
            print(f"  {conc:>5}  {rps:>7.1f}  {p50:>9.1f}  {p95:>9.1f}")
        print()


if __name__ == "__main__":
    main()
