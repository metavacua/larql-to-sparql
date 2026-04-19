#!/usr/bin/env python3
"""
GQA group rank analysis — Gemma 3 4B.

For each KV-sharing group, compute SVD effective rank of all Q-heads.
Tests whether L19H3's low rank is a GQA-group property or head-specific.

Focus layers: 0, 19 (the two target layers).
Also prints group rank variance across ALL layers to see if GQA groups
are correlated in rank (B hypothesis) or independent (C hypothesis).
"""

import torch
from transformers import AutoModelForCausalLM

MODEL = "google/gemma-3-4b-it"
THRESHOLD = 0.1
FOCUS_LAYERS = [0, 19]

def effective_rank(S, threshold=0.1):
    if S.numel() == 0 or S[0] < 1e-12:
        return 0
    cutoff = threshold * S[0].item()
    return int((S > cutoff).sum().item())

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32)

for path in [
    lambda m: m.model.language_model.layers,
    lambda m: m.model.model.layers,
    lambda m: m.model.layers,
]:
    try:
        layers = path(model)
        if layers: break
    except AttributeError:
        continue

cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
num_q   = cfg.num_attention_heads    # 8
num_kv  = cfg.num_key_value_heads    # 4
head_dim = cfg.head_dim              # 256
gqa_g   = num_q // num_kv            # 2

print(f"  {len(layers)} layers, {num_q} Q-heads, {num_kv} KV-heads, group={gqa_g}\n")

# ── Per-layer, per-group rank table ─────────────────────────────────────────

print("Layer  KV-group  Q-heads  Ranks (each head)  Group-range  Note")
print("─" * 72)

# Collect all ranks for correlation analysis
all_group_ranks = {}  # (layer, kv_group) -> [rank_h0, rank_h1, ...]

for li, layer in enumerate(layers):
    attn = layer.self_attn
    W_Q = attn.q_proj.weight.detach()
    W_K = attn.k_proj.weight.detach()

    for kg in range(num_kv):
        k_slice = W_K[kg * head_dim:(kg + 1) * head_dim, :]
        heads = range(kg * gqa_g, (kg + 1) * gqa_g)
        ranks = []
        for h in heads:
            q_slice = W_Q[h * head_dim:(h + 1) * head_dim, :]
            M = q_slice @ k_slice.T
            _, S, _ = torch.linalg.svd(M, full_matrices=False)
            ranks.append(effective_rank(S, THRESHOLD))
        all_group_ranks[(li, kg)] = ranks

        if li in FOCUS_LAYERS:
            rng = max(ranks) - min(ranks)
            note = ""
            # Flag the target heads
            for h, r in zip(heads, ranks):
                if (li, h) in [(0, 4), (0, 3), (19, 3), (20, 4)]:
                    note += f" ← L{li}H{h}(rank={r})"
            bar = "  ".join(f"H{h}:{r}" for h, r in zip(heads, ranks))
            print(f"L{li:02d}    KV{kg}      {list(heads)}  {bar}  range={rng}{note}")

    if li in FOCUS_LAYERS:
        print()

# ── Within-group rank correlation across all layers ─────────────────────────

print("\n=== Within-group rank correlation (all layers) ===\n")
print("Do Q-heads in the same KV group have correlated ranks?")
print("If B (GQA mechanism): within-group variance << across-group variance.\n")

import numpy as np

within_diffs = []
across_diffs = []

for li in range(len(layers)):
    # Within-group: rank difference between the two heads in each group
    group_means = []
    for kg in range(num_kv):
        r = all_group_ranks[(li, kg)]
        within_diffs.append(abs(r[0] - r[1]))
        group_means.append(np.mean(r))
    # Across-group: variance of group means within a layer
    if len(group_means) > 1:
        across_diffs.append(np.std(group_means))

print(f"  Within-group |rank_h0 - rank_h1| mean = {np.mean(within_diffs):.2f}  std={np.std(within_diffs):.2f}")
print(f"  Across-group std(group_means) per layer  mean = {np.mean(across_diffs):.2f}  std={np.std(across_diffs):.2f}")
print()

ratio = np.mean(across_diffs) / (np.mean(within_diffs) + 1e-6)
if ratio > 3:
    verdict = "ACROSS >> WITHIN → ranks vary MORE between groups than within. GQA groups coherent (supports B)."
elif ratio < 1:
    verdict = "WITHIN ≈ ACROSS → no group coherence. B is unlikely."
else:
    verdict = f"Mixed signal (ratio={ratio:.1f}). Inconclusive."
print(f"  Ratio across/within = {ratio:.1f}  →  {verdict}")

# ── Specifically: L19H3's full KV group ─────────────────────────────────────

print("\n=== L19H3 GQA group detail ===\n")
# L19H3: head=3, kv_group = 3 // 2 = 1 → group heads {2, 3}
li = 19
kg = 3 // gqa_g
print(f"  L19H3 is in KV-group {kg}, sharing K/V with heads {list(range(kg*gqa_g, (kg+1)*gqa_g))}")
attn = layers[li].self_attn
W_Q = attn.q_proj.weight.detach()
W_K = attn.k_proj.weight.detach()
k_slice = W_K[kg * head_dim:(kg + 1) * head_dim, :]
for h in range(kg * gqa_g, (kg + 1) * gqa_g):
    q_slice = W_Q[h * head_dim:(h + 1) * head_dim, :]
    M = q_slice @ k_slice.T
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    r = effective_rank(S, THRESHOLD)
    print(f"    L19H{h}: rank={r}  top-8 σ = {S[:8].tolist()}")

# K matrix rank
_, SK, _ = torch.linalg.svd(k_slice, full_matrices=False)
print(f"\n  K matrix itself (KV-group {kg}, layer 19):")
print(f"    rank(K) = {effective_rank(SK, THRESHOLD)}  top-8 σ = {SK[:8].tolist()}")

# ── L0H4's full KV group ─────────────────────────────────────────────────────

print("\n=== L0H4 GQA group detail ===\n")
li = 0
kg = 4 // gqa_g  # head=4, group = 4//2 = 2 → heads {4,5}
print(f"  L0H4 is in KV-group {kg}, sharing K/V with heads {list(range(kg*gqa_g, (kg+1)*gqa_g))}")
attn = layers[li].self_attn
W_Q = attn.q_proj.weight.detach()
W_K = attn.k_proj.weight.detach()
k_slice = W_K[kg * head_dim:(kg + 1) * head_dim, :]
for h in range(kg * gqa_g, (kg + 1) * gqa_g):
    q_slice = W_Q[h * head_dim:(h + 1) * head_dim, :]
    M = q_slice @ k_slice.T
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    r = effective_rank(S, THRESHOLD)
    print(f"    L0H{h}: rank={r}  top-8 σ = {S[:8].tolist()}")

_, SK, _ = torch.linalg.svd(k_slice, full_matrices=False)
print(f"\n  K matrix itself (KV-group {kg}, layer 0):")
print(f"    rank(K) = {effective_rank(SK, THRESHOLD)}  top-8 σ = {SK[:8].tolist()}")

print("\nDone.")
