"""
Benchmark: Residual stream trace capture and storage.

Measures:
  - Trace capture time (full forward pass with decomposition)
  - Storage sizes at each tier
  - Mmap read latency (zero-copy)
  - Boundary store append/read throughput

Usage:
    python examples/demos/trace_bench.py output/gemma3-4b-v2.vindex
"""

import sys
import time
import os
import tempfile
import numpy as np

import larql
from larql._native import TraceStore, BoundaryWriter, BoundaryStore


def bench(label, fn, n=1):
    """Run fn n times, return mean time in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    mean = np.mean(times)
    print(f"  {label}: {mean:.1f}ms" + (f" (n={n})" if n > 1 else ""))
    return result


def main():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex"

    wm = larql.WalkModel(vindex_path)
    hidden = wm.hidden_size
    n_layers = wm.num_layers
    print(f"Model: {n_layers} layers, {hidden}D hidden")

    prompts = [
        "The capital of France is",
        "Albert Einstein was born in",
        "The programming language Python was created by",
        "The largest ocean on Earth is the",
        "The company Apple was founded by",
    ]

    # ── Trace capture ──
    print(f"\n--- Trace capture ---")

    bench("Last-token trace (1 prompt)", lambda: wm.trace(prompts[0]))
    bench("All-position trace (1 prompt)", lambda: wm.trace(prompts[0], positions="all"))

    traces = []
    def capture_all():
        traces.clear()
        for p in prompts:
            traces.append(wm.trace(p, positions="all"))
    bench(f"All-position trace ({len(prompts)} prompts)", capture_all)

    # ── Answer trajectory ──
    print(f"\n--- Answer trajectory ---")
    t = wm.trace("The capital of France is")
    bench("answer_trajectory('Paris')", lambda: t.answer_trajectory("Paris"), n=5)
    bench("top_k(24)", lambda: t.top_k(24), n=5)
    bench("rank_of('Paris', 24)", lambda: t.rank_of("Paris", 24), n=5)
    bench("summary()", lambda: t.summary(), n=3)

    # ── Storage sizes ──
    print(f"\n--- Storage sizes ---")

    chain_bytes = (n_layers + 1) * 3 * hidden * 4
    boundary_bytes = hidden * 4
    tier4_vecs = 1 + 2 * 11  # L22 residual + 11 layers of attn+ffn deltas
    tier4_bytes = tier4_vecs * hidden * 4
    tier4_int8_bytes = tier4_vecs * (8 + hidden)  # 8 bytes range + int8 data

    print(f"  Per-token full chain: {chain_bytes:,} bytes ({chain_bytes/1024:.1f} KB)")
    print(f"  Per-window boundary:  {boundary_bytes:,} bytes ({boundary_bytes/1024:.1f} KB)")
    print(f"  Per-window Tier 4:    {tier4_bytes:,} bytes ({tier4_bytes/1024:.1f} KB)")
    print(f"  Per-window Tier 4 int8: {tier4_int8_bytes:,} bytes ({tier4_int8_bytes/1024:.1f} KB)")

    for n_tokens in [1000, 10_000, 100_000, 370_000, 1_000_000]:
        n_windows = n_tokens // 200
        t1_mb = n_windows * boundary_bytes / 1e6
        t4_mb = n_windows * tier4_bytes / 1e6
        t4i8_mb = n_windows * tier4_int8_bytes / 1e6
        kv_mb = n_tokens * 56000 / 370000  # scale from Apollo 11
        print(f"  {n_tokens:>10,} tokens: Tier1={t1_mb:.0f}MB, Tier4={t4_mb:.0f}MB, "
              f"Tier4-int8={t4i8_mb:.0f}MB, KV≈{kv_mb:.0f}MB")

    # ── Full chain store ──
    print(f"\n--- Full chain store (write/read) ---")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        trace_path = tmp.name

    t_all = wm.trace(prompts[0], positions="all")
    bench("Save trace (6 tokens, all layers)", lambda: t_all.save(trace_path))
    fsize = os.path.getsize(trace_path)
    print(f"  File size: {fsize:,} bytes ({fsize/1e6:.2f} MB)")

    def read_store():
        s = TraceStore(trace_path)
        _ = s.residual(5, 25)  # last token, L24
        return s
    bench("Open + read one vector", read_store, n=5)
    os.unlink(trace_path)

    # ── Boundary store ──
    print(f"\n--- Boundary store (append/read throughput) ---")

    with tempfile.NamedTemporaryFile(suffix=".bndx", delete=False) as tmp:
        bndx_path = tmp.name

    n_boundaries = 100
    vecs = [np.random.randn(hidden).astype(np.float32) for _ in range(n_boundaries)]

    def write_boundaries():
        w = BoundaryWriter(bndx_path, hidden, window_size=200, max_boundaries=n_boundaries)
        for i, v in enumerate(vecs):
            w.append(i * 200, 200, v.tolist())
        w.finish()
    bench(f"Write {n_boundaries} boundaries", write_boundaries)
    fsize = os.path.getsize(bndx_path)
    print(f"  File size: {fsize:,} bytes ({fsize/1024:.1f} KB)")

    def read_boundaries():
        s = BoundaryStore(bndx_path)
        for i in range(n_boundaries):
            _ = s.residual(i)
    bench(f"Read {n_boundaries} boundaries", read_boundaries, n=5)

    def read_single():
        s = BoundaryStore(bndx_path)
        _ = s.residual(50)
    bench("Open + read 1 boundary", read_single, n=10)

    os.unlink(bndx_path)

    # ── Projection: Apollo 11 ──
    print(f"\n--- Apollo 11 projection (370K tokens) ---")
    n_windows = 1850
    print(f"  Windows: {n_windows}")
    print(f"  Tier 1: {n_windows * boundary_bytes / 1e6:.1f} MB ({56000 / (n_windows * boundary_bytes / 1e6):.0f}x vs KV)")
    print(f"  Tier 4: {n_windows * tier4_bytes / 1e6:.1f} MB ({56000 / (n_windows * tier4_bytes / 1e6):.0f}x vs KV)")
    print(f"  Tier 4 int8: {n_windows * tier4_int8_bytes / 1e6:.1f} MB ({56000 / (n_windows * tier4_int8_bytes / 1e6):.0f}x vs KV)")
    print(f"  KV cache: 56,000 MB")


if __name__ == "__main__":
    main()
