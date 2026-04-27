#!/usr/bin/env python3
"""
Performance diagnostic for larql-router.

Triangulates where latency is spent by comparing:
  - Direct shard access (bypass router)
  - Via router
  - Health-only baseline
  - JSON vs binary wire format (full_output mode)

Usage:
    python3 scripts/grid/bench_profile.py \
        [--router URL] [--shard-a URL] [--shard-b URL] \
        [--reps N] [--warmup N]

    # Auto-discover shards if not specified (tries ports 8080/8081):
    python3 scripts/grid/bench_profile.py --router http://127.0.0.1:9090
"""
import argparse
import json
import struct
import statistics
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

HIDDEN = 2560

# ── Binary wire codec (mirrors crates/larql-inference/src/ffn/remote.rs) ──────

BATCH_MARKER = 0xFFFFFFFF
BINARY_CT    = "application/x-larql-ffn"

def bin_encode_single(layer: int, residual_list: list[float],
                      seq_len: int = 1, full_output: bool = True, top_k: int = 0) -> bytes:
    buf = bytearray()
    buf += struct.pack('<I', layer)
    buf += struct.pack('<I', seq_len)
    buf += struct.pack('<I', 1 if full_output else 0)
    buf += struct.pack('<I', top_k)
    buf += struct.pack(f'<{len(residual_list)}f', *residual_list)
    return bytes(buf)

def bin_decode_single(body: bytes) -> tuple[int, float, int]:
    """Returns (layer, latency_ms, num_output_floats)."""
    if len(body) < 12:
        raise ValueError(f"response too short: {len(body)} bytes")
    layer, seq_len = struct.unpack_from('<II', body, 0)
    latency_ms     = struct.unpack_from('<f',  body, 8)[0]
    num_floats     = (len(body) - 12) // 4
    return layer, latency_ms, num_floats


# ── helpers ───────────────────────────────────────────────────────────────────

def residual(n: int) -> list[float]:
    return [0.01] * n


def post(client: httpx.Client, url: str, body: dict) -> tuple[float, dict | None]:
    t0 = time.perf_counter()
    try:
        r = client.post(url, json=body, timeout=15.0)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            return ms, r.json()
        return ms, None
    except Exception:
        return (time.perf_counter() - t0) * 1000, None


def get(client: httpx.Client, url: str) -> float:
    t0 = time.perf_counter()
    try:
        client.get(url, timeout=5.0)
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000


def warmup(client: httpx.Client, url: str, body: dict, n: int):
    for _ in range(n):
        client.post(url, json=body, timeout=15.0)


def sample(client: httpx.Client, url: str, body: dict, reps: int) -> tuple[list[float], list[float]]:
    """Returns (wall_times, ffn_reported_times)."""
    walls, ffns = [], []
    for _ in range(reps):
        ms, d = post(client, url, body)
        if d is not None:
            walls.append(ms)
            ffn = d.get("latency_ms", 0.0)
            ffns.append(ffn)
    return walls, ffns


def post_binary(client: httpx.Client, url: str, body: bytes) -> tuple[float, bytes | None]:
    t0 = time.perf_counter()
    try:
        r = client.post(url, content=body,
                        headers={"content-type": BINARY_CT}, timeout=15.0)
        ms = (time.perf_counter() - t0) * 1000
        return ms, r.content if r.status_code == 200 else None
    except Exception:
        return (time.perf_counter() - t0) * 1000, None


def sample_binary(client: httpx.Client, url: str, body: bytes, reps: int) -> list[float]:
    walls = []
    for _ in range(reps):
        ms, resp = post_binary(client, url, body)
        if resp is not None:
            walls.append(ms)
    return walls


def p(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[int(len(s) * pct)]


def fmt(vals: list[float]) -> str:
    if not vals:
        return "  n/a"
    return (f"p50={p(vals, 0.50):5.1f}  p95={p(vals, 0.95):5.1f}  "
            f"mean={statistics.mean(vals):5.1f}  n={len(vals)}")


def section(title: str):
    print()
    print(f"── {title} {'─' * (60 - len(title))}")


def reachable(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/v1/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",  default="http://127.0.0.1:9090")
    parser.add_argument("--shard-a", default=None, help="Direct URL for shard A (layers 0-16)")
    parser.add_argument("--shard-b", default=None, help="Direct URL for shard B (layers 17-33)")
    parser.add_argument("--reps",    type=int, default=40)
    parser.add_argument("--warmup",  type=int, default=8)
    args = parser.parse_args()

    # Auto-discover shards
    shard_a = args.shard_a or "http://127.0.0.1:8080"
    shard_b = args.shard_b or "http://127.0.0.1:8081"

    print("=" * 72)
    print(" larql-router Performance Diagnostic")
    print("=" * 72)
    print(f"  Router:  {args.router}")
    print(f"  Shard A: {shard_a}  {'✓' if reachable(shard_a) else '✗ unreachable'}")
    print(f"  Shard B: {shard_b}  {'✓' if reachable(shard_b) else '✗ unreachable'}")
    print(f"  reps={args.reps}  warmup={args.warmup}")

    if not reachable(args.router):
        sys.exit(f"\nRouter not reachable: {args.router}")

    body_a = {"layer": 5,  "residual": residual(HIDDEN)}
    body_b = {"layer": 25, "residual": residual(HIDDEN)}

    with httpx.Client(timeout=20.0) as client:

        # ── 1. Pure HTTP baseline ─────────────────────────────────────────────
        section("1. Pure HTTP baseline (no FFN)")
        health_times = [get(client, f"{args.router}/v1/health") for _ in range(args.reps)]
        print(f"  Health GET (router):              {fmt(health_times)}")
        if reachable(shard_a):
            health_a = [get(client, f"{shard_a}/v1/health") for _ in range(args.reps)]
            print(f"  Health GET (shard A direct):      {fmt(health_a)}")
        print(f"  → Fixed TCP+HTTP overhead: ~{p(health_times, 0.50):.1f}ms per round-trip")

        # ── 2. Direct shard — bypass router ──────────────────────────────────
        section("2. Direct shard access (router bypassed)")
        direct_a_wall, direct_a_ffn = [], []
        direct_b_wall, direct_b_ffn = [], []
        if reachable(shard_a):
            warmup(client, f"{shard_a}/v1/walk-ffn", body_a, args.warmup)
            direct_a_wall, direct_a_ffn = sample(client, f"{shard_a}/v1/walk-ffn", body_a, args.reps)
            print(f"  Shard A wall-clock (layer 5):     {fmt(direct_a_wall)}")
            print(f"  Shard A FFN reported:             {fmt(direct_a_ffn)}")
            if direct_a_wall and direct_a_ffn:
                shard_http_a = p(direct_a_wall, 0.50) - p(direct_a_ffn, 0.50)
                print(f"  Shard A HTTP+JSON overhead:       ~{shard_http_a:.1f}ms  "
                      f"(wall p50 − FFN p50)")
        if reachable(shard_b):
            warmup(client, f"{shard_b}/v1/walk-ffn", body_b, args.warmup)
            direct_b_wall, direct_b_ffn = sample(client, f"{shard_b}/v1/walk-ffn", body_b, args.reps)
            print(f"  Shard B wall-clock (layer 25):    {fmt(direct_b_wall)}")
            print(f"  Shard B FFN reported:             {fmt(direct_b_ffn)}")
            if direct_b_wall and direct_b_ffn:
                shard_http_b = p(direct_b_wall, 0.50) - p(direct_b_ffn, 0.50)
                print(f"  Shard B HTTP+JSON overhead:       ~{shard_http_b:.1f}ms")

        # ── 3. Via router ─────────────────────────────────────────────────────
        section("3. Via router (full stack)")
        warmup(client, f"{args.router}/v1/walk-ffn", body_a, args.warmup)
        warmup(client, f"{args.router}/v1/walk-ffn", body_b, args.warmup)
        router_a_wall, router_a_ffn = sample(client, f"{args.router}/v1/walk-ffn", body_a, args.reps)
        router_b_wall, router_b_ffn = sample(client, f"{args.router}/v1/walk-ffn", body_b, args.reps)
        print(f"  Router→A wall-clock (layer 5):    {fmt(router_a_wall)}")
        print(f"  Router→A FFN reported:            {fmt(router_a_ffn)}")
        print(f"  Router→B wall-clock (layer 25):   {fmt(router_b_wall)}")
        print(f"  Router→B FFN reported:            {fmt(router_b_ffn)}")

        # ── 4. Router overhead decomposition ─────────────────────────────────
        section("4. Router overhead decomposition")
        if direct_a_wall and router_a_wall:
            router_added_a = p(router_a_wall, 0.50) - p(direct_a_wall, 0.50)
            client_http    = p(health_times, 0.50)
            shard_http     = p(direct_a_wall, 0.50) - p(direct_a_ffn, 0.50) if direct_a_ffn else 0
            ffn_compute    = p(direct_a_ffn, 0.50)
            router_internal = p(router_a_wall, 0.50) - p(direct_a_wall, 0.50)
            total           = p(router_a_wall, 0.50)

            print(f"  Layer 5 end-to-end budget (p50):")
            print(f"    FFN compute (shard kernel):     {ffn_compute:5.1f} ms  "
                  f"({ffn_compute/total*100:.0f}%)")
            print(f"    Shard HTTP+JSON overhead:       {shard_http:5.1f} ms  "
                  f"({shard_http/total*100:.0f}%)")
            print(f"    Router added overhead:          {router_internal:5.1f} ms  "
                  f"({router_internal/total*100:.0f}%)")
            print(f"    ─────────────────────────────────────────")
            print(f"    Total (client→router→shard→client): {total:.1f} ms")
        else:
            print("  (need direct shard access for decomposition)")

        # ── 5. Body size sensitivity ──────────────────────────────────────────
        section("5. Body size sensitivity (JSON serialisation cost)")
        sizes = [64, 256, 512, 1024, 2048, HIDDEN]
        print(f"  {'residual len':>12}  {'body ~bytes':>11}  {'router p50':>10}  {'direct p50':>10}  overhead")
        for n in sizes:
            b = {"layer": 5, "residual": residual(n)}
            body_bytes = len(json.dumps(b))
            warmup(client, f"{args.router}/v1/walk-ffn", b, 3)
            r_times, _ = sample(client, f"{args.router}/v1/walk-ffn", b, 20)
            d_times: list[float] = []
            if reachable(shard_a):
                warmup(client, f"{shard_a}/v1/walk-ffn", b, 3)
                d_times, _ = sample(client, f"{shard_a}/v1/walk-ffn", b, 20)
            r_p50 = p(r_times, 0.50)
            d_p50 = p(d_times, 0.50) if d_times else 0
            ovhd  = r_p50 - d_p50 if d_times else 0
            print(f"  {n:>12}  {body_bytes:>11}  {r_p50:>9.1f}ms  {d_p50:>9.1f}ms  {ovhd:+.1f}ms")

        # ── 6. Connection pool effectiveness ──────────────────────────────────
        section("6. Connection warm vs cold")
        cold_times = []
        for _ in range(10):
            with httpx.Client(timeout=15.0) as fresh:   # new client = new TCP connection
                ms, _ = post(fresh, f"{args.router}/v1/walk-ffn", body_a)
                cold_times.append(ms)
        warm_a_sample, _ = sample(client, f"{args.router}/v1/walk-ffn", body_a, 10)
        print(f"  Cold connection (new TCP per req): {fmt(cold_times)}")
        print(f"  Warm connection (pooled):          {fmt(warm_a_sample)}")
        if cold_times and warm_a_sample:
            tcp_cost = p(cold_times, 0.50) - p(warm_a_sample, 0.50)
            print(f"  → TCP handshake cost: ~{tcp_cost:.1f}ms per request")

        # ── 7. Batched walk amortisation ──────────────────────────────────────
        section("7. Batched walk — amortisation per layer")
        print(f"  {'layers':>6}  {'wall p50':>9}  {'per-layer':>10}  {'fan-out?':>9}")
        for layers in [[5], [5, 25], [0,1,2,3], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]:
            b = {"layers": layers, "residual": residual(HIDDEN)}
            warmup(client, f"{args.router}/v1/walk-ffn", b, 3)
            t, _ = sample(client, f"{args.router}/v1/walk-ffn", b, 20)
            w = p(t, 0.50)
            per = w / len(layers)
            fanout = "yes" if len(set(l < 17 for l in layers)) > 1 else "no"
            print(f"  {str(layers):>6}  {w:>8.1f}ms  {per:>9.1f}ms  {fanout:>9}")

        # ── 8. Binary vs JSON — full_output mode ─────────────────────────────
        section("8. Binary vs JSON wire format (full_output=true, hidden=2560)")
        res = residual(HIDDEN)
        json_body_full = {"layer": 5, "residual": res, "full_output": True, "seq_len": 1}
        bin_body_full  = bin_encode_single(5, res)

        if reachable(shard_a):
            # warmup both paths
            for _ in range(args.warmup):
                client.post(f"{shard_a}/v1/walk-ffn", json=json_body_full, timeout=15.0)
            for _ in range(args.warmup):
                post_binary(client, f"{shard_a}/v1/walk-ffn", bin_body_full)

            json_full, _ = sample(client, f"{shard_a}/v1/walk-ffn", json_body_full, args.reps)
            bin_full      = sample_binary(client, f"{shard_a}/v1/walk-ffn", bin_body_full, args.reps)

            print(f"  JSON  (full_output, direct shard A):    {fmt(json_full)}")
            print(f"  Binary (full_output, direct shard A):   {fmt(bin_full)}")
            if json_full and bin_full:
                saving = p(json_full, 0.50) - p(bin_full, 0.50)
                pct    = saving / p(json_full, 0.50) * 100 if p(json_full, 0.50) else 0
                req_bytes_json = len(json.dumps(json_body_full).encode())
                req_bytes_bin  = len(bin_body_full)
                print(f"  → Binary saves: {saving:+.1f}ms / hop  ({pct:.0f}% of JSON round-trip)")
                print(f"  → Request size: JSON={req_bytes_json:,}B  Binary={req_bytes_bin:,}B  "
                      f"({(1 - req_bytes_bin/req_bytes_json)*100:.0f}% smaller)")

            # also via router
            warmup(client, f"{args.router}/v1/walk-ffn", json_body_full, 3)
            for _ in range(3):
                post_binary(client, f"{args.router}/v1/walk-ffn", bin_body_full)
            json_router_full, _ = sample(client, f"{args.router}/v1/walk-ffn", json_body_full, 20)
            bin_router_full      = sample_binary(client, f"{args.router}/v1/walk-ffn", bin_body_full, 20)
            print(f"  JSON  (full_output, via router):        {fmt(json_router_full)}")
            print(f"  Binary (full_output, via router):       {fmt(bin_router_full)}")

        # ── 9. Summary and recommendations ───────────────────────────────────
        section("9. Summary and optimisation targets")
        if direct_a_ffn and router_a_wall:
            ffn   = p(direct_a_ffn, 0.50)
            s_http = p(direct_a_wall, 0.50) - ffn if direct_a_wall else 0
            r_add  = p(router_a_wall, 0.50) - p(direct_a_wall, 0.50) if direct_a_wall else 0
            total  = p(router_a_wall, 0.50)
            print(f"  Per-hop budget breakdown (p50, layer 5):")
            print(f"    {ffn:5.1f} ms  FFN kernel — irreducible compute")
            print(f"    {s_http:5.1f} ms  Shard HTTP+JSON — reducible (byte pass-through, HTTP/2)")
            print(f"    {r_add:5.1f} ms  Router add — reducible (JSON parse, route lookup, re-serialize)")
            print(f"    {'─'*40}")
            print(f"    {total:5.1f} ms  total end-to-end")
            print()
            reducible = s_http + r_add
            print(f"  Reducible overhead: {reducible:.1f} ms / hop  "
                  f"({reducible/total*100:.0f}% of total)")
            print(f"  Irreducible (FFN):  {ffn:.1f} ms / hop  "
                  f"({ffn/total*100:.0f}% of total)")
            print()
            if s_http > 1.0:
                print(f"  HIGH IMPACT: shard HTTP+JSON ({s_http:.1f}ms) — "
                      "byte pass-through in router proxy_to()")
            if r_add > 0.5:
                print(f"  MEDIUM:      router JSON parse+route ({r_add:.1f}ms) — "
                      "raw Bytes extractor + avoid re-serialize")
            print(f"  LOW:         FFN compute ({ffn:.1f}ms) — "
                  "already optimised (GPU/Metal kernels)")


if __name__ == "__main__":
    main()
