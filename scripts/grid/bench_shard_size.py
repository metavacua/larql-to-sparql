#!/usr/bin/env python3
"""
Shard-size sweep: find the optimal layers-per-shard for minimum latency
without memory thrashing.

For each shard size N, starts a single server owning layers 0-(N-1),
warms it up, and measures:
  - RSS (resident RAM)
  - First-request latency (cold gate decode)
  - Warm p50 latency (gate cache populated)
  - Sequential walk latency (time to step through all N layers once)
  - VSZ/RSS ratio (confirms demand-paging)

Then prints a composite score: SeqLatency × (TOTAL_LAYERS / N) — the
estimated wall-clock cost of a full inference pass, assuming N-layer shards
are strung together serially. Lower is better.

Usage:
    python3 scripts/grid/bench_shard_size.py [--vindex PATH] [--reps N]
"""
import argparse
import subprocess
import sys
import time
import os
import signal

try:
    import httpx
    import psutil
except ImportError:
    sys.exit("pip install httpx psutil")

HIDDEN = 2560
TOTAL_LAYERS = 34
RESIDUAL = [0.01] * HIDDEN
TEST_PORT = 9001


def wait_ready(port: int, proc: subprocess.Popen, timeout: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/v1/health"
    for _ in range(timeout):
        # Bail early if the server crashed
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


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
        return psutil.Process(pid).memory_info().rss / 1024 / 1024
    except psutil.NoSuchProcess:
        return 0.0


def vsz_mb(pid: int) -> float:
    try:
        return psutil.Process(pid).memory_info().vms / 1024 / 1024
    except psutil.NoSuchProcess:
        return 0.0


def measure_shard(vindex: str, n_layers: int, reps: int) -> dict | None:
    end = n_layers - 1
    logfile = open(f"/tmp/bench_shard_{n_layers}.log", "w")
    proc = subprocess.Popen(
        [
            "./target/release/larql-server", vindex,
            "--ffn-only", "--layers", f"0-{end}",
            "--port", str(TEST_PORT),
        ],
        stdout=logfile, stderr=logfile,
    )

    if not wait_ready(TEST_PORT, proc):
        proc.kill(); proc.wait(); logfile.close()
        # Show last line of log to help diagnose
        try:
            lines = open(f"/tmp/bench_shard_{n_layers}.log").readlines()
            hint = lines[-1].strip() if lines else "no output"
            print(f" FAILED (exit={proc.returncode}, hint: {hint[:80]})")
        except Exception:
            print(" FAILED")
        return None

    time.sleep(1)  # let mmap settle

    url = f"http://127.0.0.1:{TEST_PORT}/v1/walk-ffn"
    mid = n_layers // 2

    with httpx.Client(timeout=60.0) as client:
        # First request (cold gate decode)
        t0 = time.perf_counter()
        r = client.post(url, json={"layer": mid, "residual": RESIDUAL})
        first_ms = (time.perf_counter() - t0) * 1000

        # Warm requests
        warm_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            client.post(url, json={"layer": mid, "residual": RESIDUAL})
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_p50 = sorted(warm_times)[len(warm_times) // 2]

        # Sequential walk through all owned layers (simulates one inference pass)
        t0 = time.perf_counter()
        for layer in range(n_layers):
            client.post(url, json={"layer": layer, "residual": RESIDUAL})
        seq_ms = (time.perf_counter() - t0) * 1000

    pid = pid_for_port(TEST_PORT) or proc.pid
    rss = rss_mb(pid)
    vsz = vsz_mb(pid)

    proc.terminate()
    proc.wait()
    logfile.close()
    time.sleep(2)

    return {
        "n_layers": n_layers,
        "rss_mb": rss,
        "vsz_mb": vsz,
        "rss_vsz_pct": rss / vsz * 100 if vsz > 0 else 0,
        "first_ms": first_ms,
        "warm_p50_ms": warm_p50,
        "seq_ms": seq_ms,
        # Estimated full-pass cost: sequential hops × seq_ms per shard
        "full_pass_est_ms": seq_ms * (TOTAL_LAYERS / n_layers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindex", default="output/gemma3-4b-q4k-v2.vindex")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--sizes", default="1,2,3,4,6,8,12,17,34",
                        help="comma-separated layer counts to test")
    args = parser.parse_args()

    if not os.path.isdir(args.vindex):
        sys.exit(f"vindex not found: {args.vindex}")

    sizes = [int(s) for s in args.sizes.split(",") if 0 < int(s) <= TOTAL_LAYERS]

    print("=" * 90)
    print(" Shard-Size Sweep")
    print("=" * 90)
    print(f"Vindex: {args.vindex}  reps={args.reps}")
    print()
    print(f"{'N layers':>9}  {'RSS(MB)':>8}  {'VSZ/RSS%':>8}  "
          f"{'First(ms)':>10}  {'Warm p50':>9}  {'SeqN(ms)':>9}  "
          f"{'Hops':>5}  {'PassEst(ms)':>11}  Notes")
    print("-" * 90)

    results = []
    for n in sizes:
        print(f"  Testing shard_size={n}...", end="", flush=True)
        r = measure_shard(args.vindex, n, args.reps)
        if r is None:
            print(" TIMEOUT")
            continue
        results.append(r)
        hops = TOTAL_LAYERS / n
        notes = []
        if r["rss_vsz_pct"] < 15:
            notes.append("demand-paged✓")
        if r["warm_p50_ms"] < 20:
            notes.append("fast✓")
        print(f"\r  {n:>9}  {r['rss_mb']:>8.0f}  {r['rss_vsz_pct']:>8.1f}  "
              f"{r['first_ms']:>10.0f}  {r['warm_p50_ms']:>9.1f}  {r['seq_ms']:>9.0f}  "
              f"{hops:>5.1f}  {r['full_pass_est_ms']:>11.0f}  {', '.join(notes)}")

    if not results:
        print("No results.")
        return

    print()
    best = min(results, key=lambda r: r["full_pass_est_ms"])
    print(f"  ► Best full-pass estimate: shard_size={best['n_layers']} "
          f"({TOTAL_LAYERS / best['n_layers']:.1f} hops, "
          f"~{best['full_pass_est_ms']:.0f}ms/token)")
    print()
    print("Columns:")
    print("  N layers    — layers owned by this shard")
    print("  RSS(MB)     — resident RAM after warm requests")
    print("  VSZ/RSS%    — resident/virtual ratio (low = demand-paging working)")
    print("  First(ms)   — first request latency (lazy gate decode, cold)")
    print("  Warm p50    — p50 latency after warmup (gate cache hot)")
    print("  SeqN(ms)    — time to walk all N layers once (1 shard's pass cost)")
    print("  Hops        — total_layers / shard_layers (number of shards needed)")
    print("  PassEst(ms) — SeqN × Hops (estimated serial full-pass cost)")
    print()
    print("  Note: PassEst assumes shards are called serially.")
    print("        With fan-out, parallel shards reduce wall-clock to max(shard_latency).")


if __name__ == "__main__":
    main()
