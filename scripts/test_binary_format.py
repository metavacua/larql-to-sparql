#!/usr/bin/env python3
"""
Verify the application/x-larql-ffn binary wire format end-to-end.

Tests:
  1. Binary single-layer == JSON single-layer (same output floats)
  2. Binary batch == JSON batch  (same outputs for each layer)
  3. Binary features-only returns HTTP 400
  4. Binary truncated body returns HTTP 400
  5. Binary latency header parses correctly
  6. Router transparently forwards binary (if --router is set)

Usage:
  python3 scripts/test_binary_format.py --server http://127.0.0.1:9183
  python3 scripts/test_binary_format.py --server http://127.0.0.1:9183 --router http://127.0.0.1:9190
"""

import argparse
import struct
import sys
import time
import urllib.request
import urllib.error
import json

BINARY_CT = "application/x-larql-ffn"
BATCH_MARKER = 0xFFFF_FFFF

# ── binary codec ──────────────────────────────────────────────────────────────

def encode_single(layer: int, residual: list[float], seq_len: int = 1,
                  full_output: bool = True, top_k: int = 8092) -> bytes:
    buf = struct.pack("<IIII", layer, seq_len, int(full_output), top_k)
    buf += struct.pack(f"<{len(residual)}f", *residual)
    return buf


def encode_batch(layers: list[int], residual: list[float], seq_len: int = 1,
                 full_output: bool = True, top_k: int = 8092) -> bytes:
    buf = struct.pack("<II", BATCH_MARKER, len(layers))
    buf += struct.pack(f"<{len(layers)}I", *layers)
    buf += struct.pack("<III", seq_len, int(full_output), top_k)
    buf += struct.pack(f"<{len(residual)}f", *residual)
    return buf


def decode_single_response(body: bytes) -> tuple[int, int, float, list[float]]:
    """Returns (layer, seq_len, latency_ms, output_floats)."""
    if len(body) < 12:
        raise ValueError(f"response too short: {len(body)} bytes")
    layer, seq_len = struct.unpack_from("<II", body, 0)
    latency = struct.unpack_from("<f", body, 8)[0]
    n = (len(body) - 12) // 4
    floats = list(struct.unpack_from(f"<{n}f", body, 12))
    return layer, seq_len, latency, floats


def decode_batch_response(body: bytes) -> tuple[float, dict[int, list[float]]]:
    """Returns (latency_ms, {layer: output_floats})."""
    marker, num_results = struct.unpack_from("<II", body, 0)
    if marker != BATCH_MARKER:
        raise ValueError(f"expected BATCH_MARKER, got {marker:#010x}")
    latency = struct.unpack_from("<f", body, 8)[0]
    offset = 12
    results = {}
    for _ in range(num_results):
        layer, seq_len, num_floats = struct.unpack_from("<III", body, offset)
        offset += 12
        floats = list(struct.unpack_from(f"<{num_floats}f", body, offset))
        offset += num_floats * 4
        results[layer] = floats
    return latency, results


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_binary(url: str, body: bytes) -> tuple[int, str, bytes]:
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": BINARY_CT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


# ── test helpers ──────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, bool, str]] = []

def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def floats_close(a: list[float], b: list[float], tol: float = 1e-4) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + 1e-6 * max(abs(x), abs(y)) for x, y in zip(a, b))


# ── tests ─────────────────────────────────────────────────────────────────────

def run_tests(server: str, router: str | None):
    url = f"{server}/v1/walk-ffn"

    # Get hidden_size from stats.
    with urllib.request.urlopen(f"{server}/v1/stats", timeout=10) as r:
        stats = json.loads(r.read())
    hidden = stats["hidden_size"]
    print(f"\nServer: {server}  hidden_size={hidden}")

    # Fixed residual (deterministic: unit vector in dim 0).
    residual = [0.0] * hidden
    residual[0] = 1.0

    test_layer = 5
    batch_layers = [5, 20]

    # ── Test 1: single-layer binary vs JSON output match ──────────────────────
    print("\n── Test 1: single-layer binary == JSON ──")

    status_j, resp_j = post_json(url, {
        "layer": test_layer, "residual": residual, "full_output": True
    })
    check("JSON single-layer 200", status_j == 200, f"status={status_j}")

    body_bin = encode_single(test_layer, residual)
    status_b, ct_b, raw_b = post_binary(url, body_bin)
    check("binary single-layer 200", status_b == 200, f"status={status_b}")
    check("binary response Content-Type", ct_b.startswith(BINARY_CT), f"ct={ct_b!r}")

    if status_j == 200 and status_b == 200:
        layer_b, seq_len_b, latency_b, output_b = decode_single_response(raw_b)
        output_j = resp_j["output"]
        check("binary layer matches", layer_b == test_layer, f"got {layer_b}")
        check("binary seq_len matches", seq_len_b == 1, f"got {seq_len_b}")
        check("output length matches", len(output_b) == len(output_j),
              f"bin={len(output_b)} json={len(output_j)}")
        close = floats_close(output_b, output_j)
        check("output values match JSON", close,
              "" if close else f"first diff at idx {next(i for i,(a,b) in enumerate(zip(output_b,output_j)) if abs(a-b)>1e-3)}: bin={output_b[0]:.6f} json={output_j[0]:.6f}")
        check("latency in response > 0", latency_b > 0, f"latency={latency_b:.2f}ms")
        print(f"   Output size: {len(output_b)} floats  latency: {latency_b:.1f}ms")

    # ── Test 2: batch binary vs JSON ─────────────────────────────────────────
    print("\n── Test 2: batch binary == JSON ──")

    status_jb, resp_jb = post_json(url, {
        "layers": batch_layers, "residual": residual, "full_output": True
    })
    check("JSON batch 200", status_jb == 200, f"status={status_jb}")

    body_bin_batch = encode_batch(batch_layers, residual)
    status_bb, ct_bb, raw_bb = post_binary(url, body_bin_batch)
    check("binary batch 200", status_bb == 200, f"status={status_bb}")
    check("binary batch Content-Type", ct_bb.startswith(BINARY_CT), f"ct={ct_bb!r}")

    if status_jb == 200 and status_bb == 200:
        latency_bb, batch_results = decode_batch_response(raw_bb)
        check("batch has both layers", set(batch_results.keys()) == set(batch_layers),
              f"got {sorted(batch_results.keys())}")

        json_results = {r["layer"]: r["output"] for r in resp_jb["results"]}
        for layer in batch_layers:
            if layer in batch_results and layer in json_results:
                close = floats_close(batch_results[layer], json_results[layer])
                check(f"layer {layer} output matches JSON", close)
        print(f"   Layers: {sorted(batch_results.keys())}  latency: {latency_bb:.1f}ms")

    # ── Test 3: binary features-only rejected ────────────────────────────────
    print("\n── Test 3: binary features-only → 400 ──")

    body_feat = encode_single(test_layer, residual, full_output=False)
    status_f, ct_f, raw_f = post_binary(url, body_feat)
    check("binary features-only → 400", status_f == 400, f"status={status_f}")
    if status_f == 400:
        err = json.loads(raw_f).get("error", "")
        check("error message mentions full_output", "full_output" in err.lower() or "binary" in err.lower(), f"{err!r}")

    # ── Test 4: truncated binary body → 400 ──────────────────────────────────
    print("\n── Test 4: truncated binary body → 400 ──")

    status_t, _, raw_t = post_binary(url, b"\x05\x00\x00\x00")  # layer=5, nothing else
    check("truncated body → 400", status_t == 400, f"status={status_t}")

    # ── Test 5: batch with single shard (same-shard batch) ───────────────────
    print("\n── Test 5: same-shard batch layers 0..5 ──")

    many_layers = list(range(6))  # layers 0-5 (all same shard if server owns all)
    body_many = encode_batch(many_layers, residual)
    status_m, ct_m, raw_m = post_binary(url, body_many)
    check("6-layer binary batch 200", status_m == 200, f"status={status_m}")
    if status_m == 200:
        _, batch_many = decode_batch_response(raw_m)
        check("all 6 layers in response", set(batch_many.keys()) == set(many_layers),
              f"got {sorted(batch_many.keys())}")

    # ── Test 6: size comparison JSON vs binary ────────────────────────────────
    print("\n── Test 6: wire size JSON vs binary ──")

    json_body = json.dumps({
        "layer": test_layer, "residual": residual, "full_output": True
    }).encode()
    bin_body = encode_single(test_layer, residual)
    pct = (1 - len(bin_body) / len(json_body)) * 100
    check(f"binary smaller than JSON ({len(bin_body)}B vs {len(json_body)}B, -{pct:.0f}%)",
          len(bin_body) < len(json_body))

    # ── Test 7: router transparent forwarding (if --router given) ────────────
    if router:
        print(f"\n── Test 7: router transparent forwarding ({router}) ──")
        router_url = f"{router}/v1/walk-ffn"

        # JSON via router
        status_rj, resp_rj = post_json(router_url, {
            "layer": test_layer, "residual": residual, "full_output": True
        })
        check("JSON via router 200", status_rj == 200, f"status={status_rj}")

        # Binary via router (single shard — should be forwarded raw)
        body_rb = encode_single(test_layer, residual)
        status_rb, ct_rb, raw_rb = post_binary(router_url, body_rb)
        check("binary via router 200", status_rb == 200, f"status={status_rb}")
        check("router preserves binary Content-Type", ct_rb.startswith(BINARY_CT), f"ct={ct_rb!r}")

        if status_rj == 200 and status_rb == 200:
            _, _, _, output_rb = decode_single_response(raw_rb)
            output_rj = resp_rj["output"]
            close = floats_close(output_rb, output_rj)
            check("router binary output matches JSON", close)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'─'*50}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("\nFailed:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}  {detail}")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:9183")
    parser.add_argument("--router", default=None,
                        help="Router URL for transparent-forward test")
    args = parser.parse_args()

    ok = run_tests(args.server, args.router)
    sys.exit(0 if ok else 1)
