#!/usr/bin/env python3
"""
Q-rank sweep: SVD rank of W_Q_h @ W_K_h^T across all heads.

Sharper question after GQA ruling: do Q matrices learn low-rank
specialisations at *deep* layers, independent of architecture?

Runs on any HF model. Key outputs:
  - Rank distribution (histogram)
  - Low-rank heads (rank < threshold) at deep layers (layer > 25% depth)
  - Rank vs layer scatter (is low rank concentrated at L0?)
  - Cross-model comparison summary

Usage:
  python3 scripts/qrank_sweep.py --model meta-llama/Llama-2-7b-hf
  python3 scripts/qrank_sweep.py --model meta-llama/Meta-Llama-3-8B
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM

THRESHOLD = 0.1
LOW_RANK_CUTOFF = 20   # heads with rank < this are "specialised"

def effective_rank(S, threshold=THRESHOLD):
    if S.numel() == 0 or S[0] < 1e-12:
        return 0
    cutoff = threshold * S[0].item()
    return int((S > cutoff).sum().item())

def get_layers(model):
    for path in [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = path(model)
            if layers: return layers
        except AttributeError:
            continue
    raise RuntimeError("cannot find layers")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--low-rank-cutoff", type=int, default=LOW_RANK_CUTOFF)
    args = parser.parse_args()

    print(f"=== Q-rank sweep: {args.model} ===\n")
    print("Loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", dtype=torch.bfloat16)
    layers = get_layers(model)

    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_q   = cfg.num_attention_heads
    num_kv  = getattr(cfg, "num_key_value_heads", num_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_q)
    gqa_g   = num_q // num_kv
    num_layers = len(layers)

    print(f"  {num_layers} layers, {num_q} Q-heads, {num_kv} KV-heads")
    print(f"  head_dim={head_dim}, GQA group={gqa_g}  ({'full MHA' if gqa_g == 1 else 'GQA'})\n")

    all_ranks = []   # (layer, head, rank)

    for li, layer in enumerate(layers):
        attn = layer.self_attn
        W_Q = attn.q_proj.weight.detach()
        W_K = attn.k_proj.weight.detach()
        for h in range(num_q):
            q_slice = W_Q[h * head_dim:(h + 1) * head_dim, :]
            kg = h // gqa_g
            k_slice = W_K[kg * head_dim:(kg + 1) * head_dim, :]
            M = (q_slice @ k_slice.T).float()
            _, S, _ = torch.linalg.svd(M, full_matrices=False)
            r = effective_rank(S, args.threshold)
            all_ranks.append((li, h, r))

    ranks = np.array([r for _, _, r in all_ranks])
    total = len(ranks)

    # ── Rank distribution ─────────────────────────────────────────────────
    print("Rank distribution (all heads):")
    buckets = [(1,5),(6,10),(11,20),(21,40),(41,80),(81,130),(131,256)]
    for lo, hi in buckets:
        n = int(((ranks >= lo) & (ranks <= hi)).sum())
        bar = "█" * min(n, 40)
        print(f"  rank {lo:>3}-{hi:<3}  {n:>4} heads  ({100*n/total:4.1f}%)  {bar}")
    print()

    # ── Layer-depth analysis ──────────────────────────────────────────────
    deep_start = num_layers // 4   # "deep" = top 75% of layers
    low_rank_deep = [(li, h, r) for li, h, r in all_ranks
                     if r < args.low_rank_cutoff and li >= deep_start]
    low_rank_early = [(li, h, r) for li, h, r in all_ranks
                      if r < args.low_rank_cutoff and li < deep_start]

    print(f"Low-rank heads (rank < {args.low_rank_cutoff}):")
    print(f"  Layer 0 to {deep_start-1} (early): {len(low_rank_early)}")
    print(f"  Layer {deep_start} to {num_layers-1} (deep):  {len(low_rank_deep)}")

    if low_rank_deep:
        print(f"\n  Deep low-rank heads (the key test):")
        for li, h, r in sorted(low_rank_deep, key=lambda x: x[2]):
            depth_pct = 100 * li / num_layers
            print(f"    L{li:02d}H{h:02d}  rank={r:>3}  (layer depth {depth_pct:.0f}%)")
    else:
        print(f"\n  No deep low-rank heads found. Low-rank is layer-0 effect only.")
    print()

    # ── Rank vs layer scatter (mean rank per layer) ───────────────────────
    print("Mean rank per layer (first 5 and last 5):")
    by_layer = defaultdict(list)
    for li, h, r in all_ranks:
        by_layer[li].append(r)

    show_layers = list(range(min(5, num_layers))) + list(range(max(0, num_layers-5), num_layers))
    for li in sorted(set(show_layers)):
        rs = by_layer[li]
        depth_tag = "(early)" if li < deep_start else "(deep) "
        low = sum(1 for r in rs if r < args.low_rank_cutoff)
        print(f"  L{li:02d} {depth_tag}  mean={np.mean(rs):5.1f}  min={min(rs):3d}  max={max(rs):3d}  "
              f"low-rank({args.low_rank_cutoff}) heads: {low}/{len(rs)}")

    print()

    # ── Summary verdict ───────────────────────────────────────────────────
    pct_low_rank_deep = 100 * len(low_rank_deep) / (num_layers * num_q - deep_start * num_q + 1e-9)
    print("=== Verdict ===\n")
    if len(low_rank_deep) == 0:
        verdict = "C (pure) — low-rank heads only at early layers. Deep Q matrices are full-rank."
    elif len(low_rank_deep) <= 5:
        verdict = "C+ — mostly layer-0 effect, a few deep specialisations."
    elif pct_low_rank_deep > 5:
        verdict = "A — Q specialisation is widespread across depth. General training property."
    else:
        verdict = f"C+ — {len(low_rank_deep)} deep low-rank heads. Both effects present."
    print(f"  {verdict}")
    print(f"  Deep low-rank heads: {len(low_rank_deep)} / {(num_layers - deep_start) * num_q} deep heads")
    print(f"  GQA: {'yes (group=' + str(gqa_g) + ')' if gqa_g > 1 else 'no (full MHA)'}")

if __name__ == "__main__":
    main()
