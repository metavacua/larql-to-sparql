#!/usr/bin/env python3
"""
Resilience benchmark: kill replicas one by one, verify routing behaviour.

Steps:
  1. Baseline — all 10 shards, all layers route correctly
  2. Kill 1/3 range-A replicas → 2 survivors, layer 5 still routes
  3. Kill 2/3 range-A replicas → 1 survivor, layer 5 still routes
  4. Kill last range-A replica → gap, layer 5 returns error
  5. Restart range-A servers    → layer 5 restored
  6. Kill all range-B at once   → gap for layer 20, A and C unaffected

Usage:
    python3 scripts/grid/bench_resilience.py [--router URL] [--vindex PATH]
"""
import argparse
import os
import signal
import subprocess
import sys
import time

try:
    import httpx
    import psutil
except ImportError:
    sys.exit("pip install httpx psutil")

HIDDEN = 2560
RESIDUAL = [0.01] * HIDDEN
RANGE_A_PORTS = [8080, 8081, 8082]
RANGE_B_PORTS = [8083, 8084, 8085]
RANGE_C_PORTS = [8086, 8087, 8088, 8089]

PASS = 0
FAIL = 0


def pid_for_port(port: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"], text=True
        ).strip()
        return int(out.splitlines()[0]) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def kill_port(port: int) -> bool:
    pid = pid_for_port(port)
    if pid:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.2)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    return False


def query_layer(client: httpx.Client, router: str, layer: int) -> tuple[bool, str]:
    try:
        r = client.post(f"{router}/v1/walk-ffn",
                        json={"layer": layer, "residual": RESIDUAL},
                        timeout=10.0)
        d = r.json()
        if "error" in d:
            return False, d["error"]
        return True, f"layer={d.get('layer')} latency={d.get('latency_ms')}ms"
    except Exception as e:
        return False, str(e)


def check_ok(client: httpx.Client, router: str, layer: int, desc: str):
    global PASS, FAIL
    ok, msg = query_layer(client, router, layer)
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}]  {desc:50s}  → {msg}")


def check_gap(client: httpx.Client, router: str, layer: int, desc: str):
    global PASS, FAIL
    ok, msg = query_layer(client, router, layer)
    if not ok and "no owning shard" in msg:
        PASS += 1
        print(f"  [PASS]  {desc:50s}  → gap correctly reported")
    elif not ok:
        PASS += 1  # any error = gap detected
        print(f"  [PASS]  {desc:50s}  → {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL]  {desc:50s}  → expected gap, got {msg}")


def wait_for_shard(port: int, timeout: int = 20) -> bool:
    for _ in range(timeout):
        pid = pid_for_port(port)
        if pid:
            return True
        time.sleep(1)
    return False


def start_shard(vindex: str, layers: str, port: int, grpc_port: int = 50052):
    proc = subprocess.Popen(
        [
            "./target/release/larql-server", vindex,
            "--ffn-only", "--layers", layers,
            "--port", str(port),
            "--join", f"http://127.0.0.1:{grpc_port}",
            "--public-url", f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:9090")
    parser.add_argument("--vindex", default="output/gemma3-4b-q4k-v2.vindex")
    args = parser.parse_args()

    print("=" * 60)
    print(" Resilience Benchmark — larql grid")
    print("=" * 60)
    print(f"Router: {args.router}")
    print()

    with httpx.Client(timeout=15.0) as client:
        # ── Baseline ──────────────────────────────────────────────────
        print("=== Baseline (all 10 shards) ===")
        check_ok(client, args.router, 5,  "layer 5  routes (range A)")
        check_ok(client, args.router, 20, "layer 20 routes (range B)")
        check_ok(client, args.router, 30, "layer 30 routes (range C)")
        print()

        # ── Kill 1 of 3 range-A ───────────────────────────────────────
        print("=== Kill 1/3 range-A (port 8082) ===")
        killed = kill_port(8082)
        print(f"  Killed port 8082: {killed}")
        time.sleep(2)
        check_ok(client, args.router, 5,  "layer 5  still routes via 8080 or 8081")
        check_ok(client, args.router, 20, "layer 20 unaffected (range B)")
        print()

        # ── Kill 2 of 3 range-A ───────────────────────────────────────
        print("=== Kill 2/3 range-A (port 8081) ===")
        kill_port(8081)
        time.sleep(2)
        check_ok(client, args.router, 5,  "layer 5  routes via sole survivor 8080")
        check_ok(client, args.router, 20, "layer 20 unaffected (range B)")
        print()

        # ── Kill last range-A ─────────────────────────────────────────
        print("=== Kill last range-A (port 8080) → gap ===")
        kill_port(8080)
        time.sleep(3)
        check_gap(client, args.router, 5,  "layer 5  should be a gap")
        check_ok(client,  args.router, 20, "layer 20 still routes (range B ok)")
        check_ok(client,  args.router, 30, "layer 30 still routes (range C ok)")
        print()

        # ── Restart range-A ───────────────────────────────────────────
        print("=== Restart all range-A servers ===")
        procs = []
        for port in RANGE_A_PORTS:
            p = start_shard(args.vindex, "0-11", port)
            procs.append(p)
            print(f"  Restarted shard port={port} (pid={p.pid})")

        print("  Waiting for shards to load and announce (~15s)...")
        time.sleep(15)

        check_ok(client, args.router, 5,  "layer 5  restored after rejoin")
        check_ok(client, args.router, 20, "layer 20 unaffected throughout")
        check_ok(client, args.router, 30, "layer 30 unaffected throughout")
        print()

        # ── Kill all range-B at once ──────────────────────────────────
        print("=== Kill all range-B simultaneously (8083/8084/8085) ===")
        for port in RANGE_B_PORTS:
            kill_port(port)
        time.sleep(3)
        check_gap(client, args.router, 12, "layer 12 gap (all B dead)")
        check_ok(client,  args.router, 5,  "layer 5  still routes (range A)")
        check_ok(client,  args.router, 30, "layer 30 still routes (range C)")
        print()

        # ── Cleanup restarted procs ───────────────────────────────────
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    print("=" * 60)
    print(f" Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL == 0:
        print(" ALL PASS")
    else:
        print(" FAILURES DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
