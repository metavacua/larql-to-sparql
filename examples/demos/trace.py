"""
Demo: Residual stream trace — capture, query, decompose, persist.

Shows how to trace a forward pass, track an answer through layers,
inspect the L24-26 phase transition, and use mmap'd stores.

Usage:
    python examples/demos/trace.py [path/to/model.vindex]
"""

import sys
import os
import tempfile

import numpy as np

import larql
from larql._native import TraceStore, BoundaryWriter, BoundaryStore

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "gemma3-4b-v2.vindex"
)


def main():
    # ── 1. Basic trace capture ──
    # Run a decomposed forward pass and record every residual, attn delta, FFN delta.
    print("Loading model and capturing trace...")
    wm = larql.WalkModel(VINDEX_PATH)
    t = wm.trace("The capital of France is")
    print(f"  {t}")

    # ── 2. Answer trajectory ──
    # Track "Paris" through all layers — rank, probability, and per-component contribution.
    print("\n--- Answer Trajectory: 'Paris' ---")
    traj = t.answer_trajectory("Paris")
    print(f"  {'Layer':>5}  {'Rank':>5}  {'Prob':>8}  {'Attn':>8}  {'FFN':>8}")
    for w in traj:
        print(f"  L{w.layer:>3}  {w.rank:>5}  {w.prob:>8.4f}  {w.attn_logit:>+8.1f}  {w.ffn_logit:>+8.1f}")

    # ── 3. Phase transition detail (L24-26) ──
    # The answer crystallises in a sudden 3-layer event.
    print("\n--- Phase Transition (L22-L27) ---")
    for layer in range(22, 28):
        top = t.top_k(layer)
        rank = t.rank_of("Paris", layer)
        leader = top[0] if top else ("?", 0.0)
        print(f"  L{layer}: top-1='{leader[0]}' p={leader[1]:.3f}  "
              f"Paris rank={rank}")

    # ── 4. Attention vs FFN decomposition at key layers ──
    # Show the raw vectors at the two biggest events: L24 (attention) and L25 (FFN).
    print("\n--- Attn vs FFN Decomposition ---")
    for layer in [23, 24, 25, 26]:
        attn = t.attn_delta(layer)
        ffn = t.ffn_delta(layer)
        res = t.residual(layer)
        attn_norm = float(np.linalg.norm(attn))
        ffn_norm = float(np.linalg.norm(ffn))
        res_norm = float(np.linalg.norm(res))
        # Ratio tells us which component dominated this layer
        ratio = attn_norm / (attn_norm + ffn_norm) if (attn_norm + ffn_norm) > 0 else 0
        print(f"  L{layer}: |attn|={attn_norm:>8.1f}  |ffn|={ffn_norm:>8.1f}  "
              f"|res|={res_norm:>8.1f}  attn_share={ratio:.0%}")

    # ── 5. Save to disk and re-open as mmap'd store ──
    # The store is zero-copy: the OS pages in only what you touch.
    print("\n--- Save / mmap Round-Trip ---")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        trace_path = f.name
    t.save(trace_path)
    store = TraceStore(trace_path)
    print(f"  Saved:  {trace_path}")
    print(f"  Store:  {store}")
    # Verify: residual from trace == residual from store
    orig = np.array(t.residual(25))
    # Store indexing: token 0 (only 1 token in last-position trace), layer 26 (0=emb, 1=L0, ..., 26=L25)
    loaded = np.array(store.residual(0, 26))
    cos = float(np.dot(orig, loaded) / (np.linalg.norm(orig) * np.linalg.norm(loaded)))
    print(f"  Cosine(trace L25, store L25) = {cos:.6f}")
    os.unlink(trace_path)

    # ── 6. Boundary store: create, append, read back ──
    # Boundary stores hold one residual per context window (~10 KB each).
    print("\n--- Boundary Store ---")
    with tempfile.NamedTemporaryFile(suffix=".bndx", delete=False) as f:
        bndx_path = f.name
    hidden = len(t.residual(0))
    writer = BoundaryWriter(bndx_path, hidden_size=hidden, window_size=200)
    # Append a few synthetic boundaries using real residuals
    for i, layer in enumerate([22, 25, 33]):
        vec = list(t.residual(layer))
        writer.append(token_offset=i * 200, window_tokens=200, residual=vec)
    writer.finish()
    bstore = BoundaryStore(bndx_path)
    print(f"  Written: {bndx_path}")
    print(f"  Store:   {bstore}")
    # Verify round-trip
    r0 = np.array(bstore.residual(0))
    orig22 = np.array(t.residual(22))
    cos = float(np.dot(orig22, r0) / (np.linalg.norm(orig22) * np.linalg.norm(r0)))
    print(f"  Cosine(boundary 0, trace L22) = {cos:.6f}")
    os.unlink(bndx_path)

    # ── 7. Multi-position trace ──
    # Trace all token positions, not just the last one.
    print("\n--- Multi-Position Trace ---")
    t_all = wm.trace("The capital of France is", positions="all")
    print(f"  {t_all}")
    # Compare "France" (position 4) and final position at the phase-transition layer
    res_france = np.array(t_all.residual(24, position=4))
    res_last = np.array(t_all.residual(24))
    cos = float(np.dot(res_france, res_last)
                / (np.linalg.norm(res_france) * np.linalg.norm(res_last)))
    print(f"  Cosine(France@L24, last@L24) = {cos:.4f}")
    print(f"  Top-1 at last pos L24: {t_all.top_k(24)[0]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
