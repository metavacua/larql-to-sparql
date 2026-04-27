#!/usr/bin/env python3
"""
residual_clustering.py — Does the L13 residual predict attention head patterns?

Follow-up to syntax_attention_minimal.py. The attention heads separate perfectly
by category, but the gate features don't. This script checks whether the raw
residual vector at the bottleneck (L13) clusters by category.

If it does: residual -> attention pattern is a learnable mapping, no QK needed.
If it doesn't: the routing signal emerges DURING attention, not before it.

Also tests: projecting residual through q_proj to get Q vectors — if Q clusters
by category, the routing is determined before K is even involved.

USAGE:
  python3 experiments/05_syntax_circuit_routing/residual_clustering.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import mlx.core as mx
import mlx.nn as nn


# ---- Same 20 prompts, 4 categories ------------------------------------

PROMPTS = {
    "capital": {
        "trigram": "NOUN->FUNC->NOUN",
        "prompts": [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Brazil is",
            "The capital of Egypt is",
            "The capital of Australia is",
        ],
    },
    "synonym": {
        "trigram": "ADJ->SYN->ADJ",
        "prompts": [
            "Happy means",
            "Sad means",
            "Big means",
            "Fast means",
            "Brave means",
        ],
    },
    "arithmetic": {
        "trigram": "NUM->OP->NUM",
        "prompts": [
            "2 + 3 =",
            "7 - 4 =",
            "5 * 6 =",
            "15 + 27 =",
            "8 * 9 =",
        ],
    },
    "code": {
        "trigram": "^->KW->FUNC",
        "prompts": [
            "def hello():\n    return",
            "def add(a, b):\n    return",
            "for i in range(10):\n    print",
            "if x > 0:\n    return",
            "class Dog:\n    def __init__",
        ],
    },
}


# ---- Model loading (reuse from minimal) --------------------------------

def find_model_parts(model):
    try:
        lm = model.language_model
        inner = lm.model
        if hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
            embed_fn = inner.embed_tokens
            def lm_head(h):
                return h @ embed_fn.weight.T
            return embed_fn, inner.layers, inner.norm, lm_head, True
    except AttributeError:
        pass
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            lm_head_fn = model.lm_head
            def lm_head(h):
                return lm_head_fn(h)
        else:
            def lm_head(h):
                return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale
    raise RuntimeError("Could not detect model structure.")


# ---- Forward pass capturing residuals + Q vectors ----------------------

def forward_capture_residuals_and_q(model, tokenizer, prompt, capture_layers):
    """
    Forward pass capturing:
      - Residual at every layer (last token)
      - Q vector projections at capture_layers (per-head, last token)
      - Attention head max-weight at capture_layers
    """
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    h = embed_fn(input_ids)
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    residuals = {}
    q_vectors = {}     # layer -> [n_heads, head_dim] Q for last token
    attn_heads = {}    # (layer, head) -> max_weight

    for i, layer in enumerate(layers):
        if i in capture_layers:
            sa = layer.self_attn
            B, L, D = h.shape
            h_norm = layer.input_layernorm(h)

            q = sa.q_proj(h_norm)
            k = sa.k_proj(h_norm)
            v = sa.v_proj(h_norm)

            n_heads = sa.n_heads
            n_kv_heads = sa.n_kv_heads
            head_dim = sa.head_dim
            scale = sa.scale

            q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            if hasattr(sa, 'q_norm'):
                q = sa.q_norm(q)
            if hasattr(sa, 'k_norm'):
                k = sa.k_norm(k)

            q = sa.rope(q)
            k = sa.rope(k)

            # Capture Q vectors (last token, all heads)
            q_last = q[0, :, -1, :]  # [n_heads, head_dim]
            mx.eval(q_last)
            q_vectors[i] = np.array(q_last.astype(mx.float32))

            # GQA expand
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)

            weights = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                weights = weights + mask
            weights = mx.softmax(weights, axis=-1)

            weights_np = np.array(weights[0, :, -1, :].astype(mx.float32))
            mx.eval(weights)

            for head_idx in range(n_heads):
                attn_heads[(i, head_idx)] = float(np.max(weights_np[head_idx]))

            attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            attn_out = sa.o_proj(attn_out)

            if hasattr(layer, 'post_attention_layernorm'):
                h = h + layer.post_attention_layernorm(attn_out)
            else:
                h = h + attn_out

            if hasattr(layer, 'pre_feedforward_layernorm'):
                h_ffn = layer.pre_feedforward_layernorm(h)
            else:
                h_ffn = h
            ffn_out = layer.mlp(h_ffn)
            if hasattr(layer, 'post_feedforward_layernorm'):
                h = h + layer.post_feedforward_layernorm(ffn_out)
            else:
                h = h + ffn_out
            mx.eval(h)
        else:
            h = layer(h, mask=mask)
            mx.eval(h)

        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return residuals, q_vectors, attn_heads, pred_tok


# ---- Analysis -----------------------------------------------------------

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def analyze_clustering(all_data, label, extractor):
    """
    Compute within-category vs between-category cosine similarity.

    extractor: function(data_item) -> numpy vector
    """
    cats = list(all_data.keys())
    vectors = {}
    for cat in cats:
        vectors[cat] = [extractor(d) for d in all_data[cat] if extractor(d) is not None]

    # Within-category similarity
    within_sims = []
    for cat in cats:
        vecs = vectors[cat]
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within_sims.append(cosine_sim(vecs[i], vecs[j]))

    # Between-category similarity
    between_sims = []
    for i, cat_a in enumerate(cats):
        for cat_b in cats[i+1:]:
            for va in vectors[cat_a]:
                for vb in vectors[cat_b]:
                    between_sims.append(cosine_sim(va, vb))

    within_avg = np.mean(within_sims) if within_sims else 0
    between_avg = np.mean(between_sims) if between_sims else 0
    gap = within_avg - between_avg

    print(f"\n  {label}:")
    print(f"    Within-category cosine:  {within_avg:.4f}  (n={len(within_sims)})")
    print(f"    Between-category cosine: {between_avg:.4f}  (n={len(between_sims)})")
    print(f"    Gap (within - between):  {gap:+.4f}")

    if gap > 0.05:
        print(f"    -> CLUSTERS (gap > 0.05)")
    elif gap > 0.02:
        print(f"    -> WEAK CLUSTERING")
    else:
        print(f"    -> NO CLUSTERING")

    # Per-pair breakdown
    print(f"    Per-pair between-category:")
    for i, cat_a in enumerate(cats):
        for cat_b in cats[i+1:]:
            pair_sims = []
            for va in vectors[cat_a]:
                for vb in vectors[cat_b]:
                    pair_sims.append(cosine_sim(va, vb))
            avg = np.mean(pair_sims) if pair_sims else 0
            print(f"      {cat_a:12s} vs {cat_b:12s}: {avg:.4f}")

    return {
        "within": float(within_avg),
        "between": float(between_avg),
        "gap": float(gap),
    }


def analyze_head_pattern_as_vector(all_data, n_heads=8, knowledge_layers=range(14, 28)):
    """
    Encode the attention head activity as a binary vector and check clustering.
    Each dimension = one (layer, head) pair, value = max_weight.
    """
    dim = len(knowledge_layers) * n_heads

    def make_head_vector(d):
        vec = np.zeros(dim)
        for (layer, head), maxw in d["attn_heads"].items():
            if layer in knowledge_layers:
                idx = (layer - min(knowledge_layers)) * n_heads + head
                if idx < dim:
                    vec[idx] = maxw
        return vec

    return analyze_clustering(all_data, "Attention head pattern vector", make_head_vector)


def analyze_residual_diff(all_data, layer_a, layer_b):
    """
    Check if the residual DELTA between two layers clusters.
    The delta captures what attention+FFN did at those layers.
    """
    def make_delta(d):
        if layer_a in d["residuals"] and layer_b in d["residuals"]:
            return d["residuals"][layer_b] - d["residuals"][layer_a]
        return None

    return analyze_clustering(
        all_data,
        f"Residual delta L{layer_a}->L{layer_b}",
        make_delta
    )


def print_head_signature_table(all_data, knowledge_layers=range(14, 28), n_heads=8):
    """Print which heads are consistently most active per category."""
    print(f"\n  Head activation signature per category:")
    print(f"  {'':15s}", end="")
    for layer in knowledge_layers:
        for h in range(n_heads):
            print(f"L{layer}H{h} ", end="")
    print()

    for cat, items in all_data.items():
        counts = defaultdict(int)
        for d in items:
            # Top 5 heads by max_weight
            sorted_heads = sorted(d["attn_heads"].items(), key=lambda x: x[1], reverse=True)[:5]
            for (layer, head), maxw in sorted_heads:
                counts[(layer, head)] += 1

        print(f"  {cat:15s}", end="")
        for layer in knowledge_layers:
            for h in range(n_heads):
                c = counts.get((layer, h), 0)
                if c >= 4:
                    print(f"  ## ", end="")
                elif c >= 3:
                    print(f"  #  ", end="")
                elif c >= 1:
                    print(f"  .  ", end="")
                else:
                    print(f"     ", end="")
        print()


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Residual clustering analysis for syntax->circuit routing"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    args = parser.parse_args()

    # Load vindex config for layer bands
    vindex_dir = Path(args.vindex)
    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    bands = config.get("layer_bands", {})
    knowledge_start = bands.get("knowledge", [14, 27])[0]
    knowledge_end = bands.get("knowledge", [14, 27])[1]
    knowledge_range = range(knowledge_start, knowledge_end + 1)
    n_layers = config["num_layers"]

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    print(f"  Model: {args.model}")
    print(f"  Knowledge layers: L{knowledge_start}-L{knowledge_end}")

    total = sum(len(p["prompts"]) for p in PROMPTS.values())
    print(f"\nRunning {total} prompts...\n")

    all_data = {}
    n = 0
    t0 = time.time()

    for cat, info in PROMPTS.items():
        cat_data = []
        for prompt in info["prompts"]:
            residuals, q_vectors, attn_heads, pred = forward_capture_residuals_and_q(
                model, tokenizer, prompt,
                capture_layers=knowledge_range,
            )
            cat_data.append({
                "prompt": prompt,
                "residuals": residuals,
                "q_vectors": q_vectors,
                "attn_heads": attn_heads,
                "prediction": pred,
            })
            n += 1
            print(f"\r  {n}/{total} ({time.time()-t0:.0f}s)", end="", flush=True)

        all_data[cat] = cat_data

    print(f"\n  Done in {time.time()-t0:.0f}s")

    # ---- Analysis ----
    print(f"\n{'='*70}")
    print(f"RESIDUAL & Q-VECTOR CLUSTERING ANALYSIS")
    print(f"{'='*70}")

    # 1. Raw residual at key layers
    for layer in [12, 13, knowledge_start, knowledge_start + 3, knowledge_end - 2, knowledge_end]:
        if layer < n_layers:
            analyze_clustering(
                all_data,
                f"Residual at L{layer}",
                lambda d, l=layer: d["residuals"].get(l),
            )

    # 2. Q vectors at first knowledge layer (per-head)
    first_kl = knowledge_start
    if first_kl in all_data[list(all_data.keys())[0]][0]["q_vectors"]:
        n_heads = all_data[list(all_data.keys())[0]][0]["q_vectors"][first_kl].shape[0]

        # Concatenated Q across all heads
        analyze_clustering(
            all_data,
            f"Q vector (all heads concat) at L{first_kl}",
            lambda d, l=first_kl: d["q_vectors"].get(l, np.zeros(1)).flatten(),
        )

        # Per-head Q
        for h in range(min(n_heads, 8)):
            analyze_clustering(
                all_data,
                f"Q vector head {h} at L{first_kl}",
                lambda d, l=first_kl, hh=h: d["q_vectors"].get(l, np.zeros((8, 256)))[hh],
            )

    # 3. Attention head pattern as vector
    analyze_head_pattern_as_vector(all_data, knowledge_layers=knowledge_range)

    # 4. Residual deltas (what did attention do?)
    analyze_residual_diff(all_data, 12, knowledge_start)
    analyze_residual_diff(all_data, 12, knowledge_end)
    analyze_residual_diff(all_data, knowledge_start, knowledge_end)

    # 5. Head signature table
    print(f"\n{'='*70}")
    print(f"HEAD ACTIVATION SIGNATURES")
    print(f"{'='*70}")
    print_head_signature_table(all_data, knowledge_layers=knowledge_range)

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")
    print(f"""
  If residual at L12/L13 clusters:
    -> The routing decision is made BEFORE attention
    -> A learned projection from residual -> head pattern can replace QK
    -> This is the "routing table" path

  If Q vectors cluster but residual doesn't:
    -> q_proj is doing the routing (linear projection selects heads)
    -> Can precompute q_proj routing without full QK

  If only attention head patterns cluster:
    -> Routing emerges FROM the QK interaction
    -> Can still cache per-template, but need template detection first

  If residual deltas cluster:
    -> Attention writes a category-specific signal into the residual
    -> The "what attention did" is template-determined
""")


if __name__ == "__main__":
    main()
