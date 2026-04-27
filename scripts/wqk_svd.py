#!/usr/bin/env python3
"""
W_Q × W_K^T SVD on all attention heads — Gemma 3 4B.

For each head, computes SVD of W_Q_h @ W_K_h^T (256×256) and counts
significant singular values (effective rank = template count per head).

Also computes W_OV × W_gate^T coupling: which FFN gate features does
each head's OV circuit write toward most strongly?

Known circuit heads (from prior research):
  L4H6   — entity routing
  L8H0   — relation detection
  L8H5   — relation detection
  L33H7  — output position

Usage:
  python3 scripts/wqk_svd.py [--threshold 0.1] [--top-k-features 10]
"""

import argparse
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

# Known circuit heads from prior research
CIRCUIT_HEADS = {
    (4, 6): "entity-routing",
    (8, 0): "relation-detect",
    (8, 5): "relation-detect",
    (33, 7): "output-position",
}

# Per-research variable head threshold: cosine < 0.94 across entity substitutions
# We use SVD effective rank > 1 as a proxy here (any head that can express > 1 mode)


def effective_rank(singular_values: torch.Tensor, threshold: float) -> int:
    """Count singular values above threshold * max singular value."""
    if singular_values.numel() == 0 or singular_values[0] < 1e-12:
        return 0
    cutoff = threshold * singular_values[0].item()
    return int((singular_values > cutoff).sum().item())


def head_label(layer: int, head: int) -> str:
    key = (layer, head)
    if key in CIRCUIT_HEADS:
        return f"L{layer:02d}H{head}  *** {CIRCUIT_HEADS[key]} ***"
    return f"L{layer:02d}H{head}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Singular value threshold as fraction of max (default 0.1)")
    parser.add_argument("--top-k-features", type=int, default=10,
                        help="Top-K gate features per head in OV coupling (default 10)")
    parser.add_argument("--no-ov", action="store_true",
                        help="Skip W_OV x W_gate^T coupling (faster)")
    args = parser.parse_args()

    print(f"=== W_Q × W_K^T SVD — {args.model} ===\n")
    print(f"  Threshold:     σ > {args.threshold} × σ_max")
    print(f"  Top-K features: {args.top_k_features}\n")

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32
    )

    # Navigate to the transformer layers — try common attribute paths
    for path in [
        lambda m: m.model.language_model.layers,         # Gemma3ForConditionalGeneration
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = path(model)
            if layers:
                break
        except AttributeError:
            continue
    else:
        print("ERROR: cannot find transformer layers", file=sys.stderr)
        sys.exit(1)

    num_layers = len(layers)
    # Detect head dimensions from first layer
    attn0 = layers[0].self_attn
    q_proj = attn0.q_proj.weight  # (num_q * head_dim, hidden)
    k_proj = attn0.k_proj.weight  # (num_kv * head_dim, hidden)
    hidden = q_proj.shape[1]

    # Detect head counts from config
    cfg = model.config
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    num_q_heads = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim = text_cfg.head_dim
    gqa_group = num_q_heads // num_kv_heads
    intermediate_size = text_cfg.intermediate_size

    print(f"  {num_layers} layers, {num_q_heads} Q-heads, {num_kv_heads} KV-heads")
    print(f"  head_dim={head_dim}, hidden={hidden}, intermediate={intermediate_size}")
    print(f"  GQA group size: {gqa_group}\n")

    # ── Per-head SVD ──────────────────────────────────────────────────────

    print(f"{'Head':>8}  {'Rank':>4}  {'σ_max':>8}  {'σ_min_kept':>10}  {'Modes'}")
    print("  " + "─" * 65)

    all_ranks = []                     # (layer, head, rank)
    high_rank_heads = []               # rank >= 4

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        W_Q = attn.q_proj.weight.detach()   # (num_q * head_dim, hidden)
        W_K = attn.k_proj.weight.detach()   # (num_kv * head_dim, hidden)

        for h in range(num_q_heads):
            q_slice = W_Q[h * head_dim:(h + 1) * head_dim, :]   # (head_dim, hidden)
            kv_group = h // gqa_group
            k_slice = W_K[kv_group * head_dim:(kv_group + 1) * head_dim, :]  # (head_dim, hidden)

            # W_Q_h @ W_K_h^T: (head_dim, hidden) @ (hidden, head_dim) = (head_dim, head_dim)
            M = q_slice @ k_slice.T

            # SVD — head_dim × head_dim matrix, fast
            try:
                _, S, _ = torch.linalg.svd(M, full_matrices=False)
            except Exception as e:
                print(f"  SVD failed at L{layer_idx}H{h}: {e}", file=sys.stderr)
                continue

            rank = effective_rank(S, args.threshold)
            sigma_max = S[0].item()
            sigma_kept = S[rank - 1].item() if rank > 0 else 0.0

            label = f"L{layer_idx:02d}H{h}"
            circuit_tag = CIRCUIT_HEADS.get((layer_idx, h), "")
            tag_str = f"  ← {circuit_tag}" if circuit_tag else ""

            print(f"  {label:>8}  {rank:>4}  {sigma_max:>8.1f}  {sigma_kept:>10.2f}  {'█' * min(rank, 32)}{tag_str}")

            all_ranks.append((layer_idx, h, rank, sigma_max))
            if rank >= 4:
                high_rank_heads.append((layer_idx, h, rank, circuit_tag))

    # ── Summary ───────────────────────────────────────────────────────────

    print()
    print("=== Summary ===\n")

    by_rank = defaultdict(list)
    for layer, head, rank, _ in all_ranks:
        by_rank[rank].append((layer, head))

    total_heads = len(all_ranks)
    print(f"Total heads:  {total_heads}  ({num_layers} layers × {num_q_heads} heads)")
    print(f"Threshold:    σ > {args.threshold} × σ_max\n")

    print("Rank distribution:")
    for rank in sorted(by_rank.keys()):
        heads = by_rank[rank]
        pct = 100 * len(heads) / total_heads
        bar = "█" * min(len(heads), 40)
        print(f"  rank={rank:>2}  {len(heads):>3} heads  ({pct:4.1f}%)  {bar}")

    print()
    # Variable heads = rank >= 2 (can express more than one mode)
    variable = [(l, h, r, t) for l, h, r, t in
                [(x[0], x[1], x[2], CIRCUIT_HEADS.get((x[0], x[1]), "")) for x in all_ranks]
                if r >= 2]
    single_mode = [x for x in all_ranks if x[2] <= 1]
    print(f"Single-mode heads (rank ≤ 1):  {len(single_mode)} ({100*len(single_mode)/total_heads:.1f}%)")
    print(f"Variable heads  (rank ≥ 2):    {len(variable)} ({100*len(variable)/total_heads:.1f}%)")

    total_templates = sum(r for _, _, r, _ in
                         [(x[0], x[1], x[2], "") for x in all_ranks])
    print(f"Total template modes:          {total_templates}")
    print(f"Unique template budget:        {total_templates}  (upper bound on distinct attention patterns)")

    print()
    print("High-rank heads (rank ≥ 4):")
    for layer, head, rank, circuit_tag in sorted(high_rank_heads, key=lambda x: -x[2]):
        tag = f"  [{circuit_tag}]" if circuit_tag else ""
        print(f"  L{layer:02d}H{head}  rank={rank}{tag}")

    # ── W_OV × W_gate^T coupling ─────────────────────────────────────────

    if not args.no_ov:
        print()
        print(f"=== W_OV × W_gate^T coupling (top-{args.top_k_features} FFN features per head) ===\n")
        print("  Scores = row norms of (W_gate_l @ W_O_h), measuring how strongly")
        print("  each gate feature is activated by head h's OV circuit writes.\n")

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            ffn = layer.mlp

            W_O = attn.o_proj.weight.detach()   # (hidden, num_q * head_dim)

            # Get gate weight — Gemma uses gate_proj
            if hasattr(ffn, "gate_proj"):
                W_gate = ffn.gate_proj.weight.detach()   # (intermediate, hidden)
            else:
                print(f"  L{layer_idx:02d}: no gate_proj, skipping OV coupling")
                continue

            # W_gate @ W_O: (intermediate, hidden) @ (hidden, num_q * head_dim)
            # = (intermediate, num_q * head_dim)
            coupling_full = W_gate @ W_O   # (intermediate, num_q * head_dim)

            for h in range(num_q_heads):
                # Column slice for head h
                coupling_h = coupling_full[:, h * head_dim:(h + 1) * head_dim]  # (intermediate, head_dim)
                # Row norms = coupling strength per gate feature
                scores = coupling_h.norm(dim=1)   # (intermediate,)
                top_vals, top_idx = scores.topk(args.top_k_features)

                circuit_tag = CIRCUIT_HEADS.get((layer_idx, h), "")
                tag_str = f"  [{circuit_tag}]" if circuit_tag else ""
                label = f"L{layer_idx:02d}H{h}"
                features = ", ".join(f"f{i.item()}({v.item():.1f})" for i, v in zip(top_idx, top_vals))
                print(f"  {label:>8}{tag_str}")
                print(f"    top features: {features}")

    print("\nDone.")


if __name__ == "__main__":
    main()
