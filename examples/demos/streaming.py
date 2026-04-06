"""
Demo: MLX inference modes from vindex.

Compares three paths:
  1. Dense (larql.mlx.load)         — all weights in GPU memory, fast
  2. Streaming (larql.streaming.load) — mmap'd, Metal pages from SSD on demand
  3. Walk FFN (larql.walk_ffn.load)   — FFN in Rust, vindex as knowledge layer

All use full weights. No feature dropping. Correct output.

Usage:
    python examples/demos/streaming.py [path/to/model.vindex]

Requires: mlx, mlx-lm, larql (built with maturin develop --release)
"""

import sys
import os
import time

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "gemma3-4b-v2.vindex"
)


def main():
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        print("Requires: pip install mlx mlx-lm")
        sys.exit(1)

    import larql

    prompts = [
        "The capital of France is",
        "Albert Einstein was a famous",
        "The largest planet in our solar system is",
    ]

    print("=" * 60)
    print("  MLX Inference Modes from Vindex")
    print("=" * 60)

    # ── 1. Dense (all weights in GPU) ──
    print(f"\n{'─' * 60}")
    print("  1. Dense — all weights in GPU memory")
    print(f"{'─' * 60}")

    t0 = time.time()
    dense_model, dense_tok = larql.mlx.load(VINDEX_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    for prompt in prompts:
        t0 = time.time()
        resp = mlx_lm.generate(dense_model, dense_tok, prompt=prompt, max_tokens=20, verbose=False)
        elapsed = time.time() - t0
        print(f"  {prompt}")
        print(f"  → {resp.strip()}")
        print(f"    {elapsed:.2f}s\n")

    # ── 2. Streaming (mmap'd, paged from SSD) ──
    print(f"{'─' * 60}")
    print("  2. Streaming — mmap'd weights, Metal pages from SSD")
    print(f"{'─' * 60}")

    t0 = time.time()
    from larql.streaming import load as load_streaming
    stream_model, stream_tok = load_streaming(VINDEX_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    for prompt in prompts:
        t0 = time.time()
        resp = mlx_lm.generate(stream_model, stream_tok, prompt=prompt, max_tokens=20, verbose=False)
        elapsed = time.time() - t0
        print(f"  {prompt}")
        print(f"  → {resp.strip()}")
        print(f"    {elapsed:.2f}s\n")

    # ── 3. Walk FFN (Rust FFN, vindex knowledge layer) ──
    print(f"{'─' * 60}")
    print("  3. Walk FFN — FFN in Rust, vindex as knowledge layer")
    print(f"{'─' * 60}")

    try:
        t0 = time.time()
        from larql.walk_ffn import load as load_walk
        walk_model, walk_tok = load_walk(VINDEX_PATH)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        for prompt in prompts:
            t0 = time.time()
            resp = mlx_lm.generate(walk_model, walk_tok, prompt=prompt, max_tokens=20, verbose=False)
            elapsed = time.time() - t0
            print(f"  {prompt}")
            print(f"  → {resp.strip()}")
            print(f"    {elapsed:.2f}s\n")
    except Exception as e:
        print(f"  Failed: {e}\n")

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print("  Dense:     fast, needs GPU memory for full model")
    print("  Streaming: slower, runs models that don't fit in GPU (120B on 8GB)")
    print("  Walk FFN:  Rust FFN, editable knowledge (INSERT/DELETE/UPDATE)")
    print()


if __name__ == "__main__":
    main()
