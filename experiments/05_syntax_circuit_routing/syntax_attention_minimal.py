#!/usr/bin/env python3
"""
syntax_attention_minimal.py — Minimal routing proof-of-concept

20 prompts x 4 categories. For each prompt capture:
  - Top syntax features (L0-12 gate activations from vindex)
  - Top attention heads (L13-26 entropy + max weight)

Print a table. Eyeball it. If categories cluster -> routing exists.

USAGE:
  python3 experiments/05_syntax_circuit_routing/syntax_attention_minimal.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# ---- 20 Prompts, 4 Categories ----------------------------------------

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


# ---- Vindex Loading ---------------------------------------------------

def load_vindex(vindex_dir):
    """Load vindex gates + feature labels for syntax layers."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    n_layers = config["num_layers"]

    # Gate vectors (per-layer offset loading)
    gate_path = vindex_dir / "gate_vectors.bin"
    gate_file_size = gate_path.stat().st_size
    total_elements = sum(li["num_features"] for li in config["layers"]) * hidden_size

    if gate_file_size == total_elements * 2:
        gate_dtype, bpe = np.float16, 2
    else:
        gate_dtype, bpe = np.float32, 4

    gate_raw = np.fromfile(gate_path, dtype=gate_dtype)
    gates = {}
    n_features_per_layer = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // bpe
        chunk = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)
        gates[layer] = chunk.astype(np.float32)
        n_features_per_layer[layer] = nf

    # Feature labels
    labels = {}
    labels_path = vindex_dir / "feature_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    print(f"  Vindex: {n_layers}L, {hidden_size}d, {len(labels)} labels")
    return config, gates, labels, n_features_per_layer


# ---- Model Loading ----------------------------------------------------

def find_model_parts(model):
    """Auto-detect model internals (handles Gemma 3, Llama, etc)."""
    # Gemma 3: model.language_model.model.{embed_tokens, layers, norm}
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

    # Llama/Mistral: model.model.{embed_tokens, layers, norm}
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


def load_model(model_name):
    """Load MLX model."""
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_name)
    print(f"  Model: {model_name}")
    return model, tokenizer


# ---- Forward Pass with Attention Capture ------------------------------

def forward_with_attention(model, tokenizer, prompt, syntax_layers, knowledge_layers):
    """
    Layer-by-layer forward pass capturing:
      - Residuals at every layer (for syntax gate projection)
      - Attention weights at knowledge layers (for head activity)
      - Top prediction from lm_head

    Handles Gemma 3 GQA: 8 Q heads, 4 KV heads (repeats=2),
    head_dim=256, q_norm/k_norm, RoPE.
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
    attn_info = {}  # (layer, head) -> {entropy, max_w, argmax}

    for i, layer in enumerate(layers):
        if i in knowledge_layers:
            # Decomposed forward: capture attention weights manually
            sa = layer.self_attn
            B, L, D = h.shape

            # Pre-attention layernorm
            h_norm = layer.input_layernorm(h)

            # QKV projections
            q = sa.q_proj(h_norm)  # [B, L, n_heads * head_dim]
            k = sa.k_proj(h_norm)  # [B, L, n_kv_heads * head_dim]
            v = sa.v_proj(h_norm)  # [B, L, n_kv_heads * head_dim]

            n_heads = sa.n_heads        # 8
            n_kv_heads = sa.n_kv_heads  # 4
            head_dim = sa.head_dim      # 256
            scale = sa.scale            # 0.0625

            q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # q_norm, k_norm (RMSNorm per head)
            if hasattr(sa, 'q_norm'):
                q = sa.q_norm(q)
            if hasattr(sa, 'k_norm'):
                k = sa.k_norm(k)

            # RoPE
            q = sa.rope(q)
            k = sa.rope(k)

            # GQA: expand KV heads to match Q heads
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = mx.repeat(k, repeats, axis=1)  # [B, n_heads, L, head_dim]
                v = mx.repeat(v, repeats, axis=1)

            # Attention weights: Q @ K^T * scale
            # [B, n_heads, L, L]
            weights = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask
            if mask is not None:
                weights = weights + mask

            weights = mx.softmax(weights, axis=-1)

            # Capture per-head stats for last token
            weights_np = np.array(weights[0, :, -1, :].astype(mx.float32))
            mx.eval(weights)
            for head_idx in range(n_heads):
                w = weights_np[head_idx]  # [L] - what last token attends to
                entropy = -np.sum(w * np.log(w + 1e-10))
                attn_info[(i, head_idx)] = {
                    "entropy": float(entropy),
                    "max_w": float(np.max(w)),
                    "argmax": int(np.argmax(w)),
                }

            # Finish attention: weights @ V -> o_proj -> residual
            attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            attn_out = sa.o_proj(attn_out)

            # Post-attention norm + residual
            if hasattr(layer, 'post_attention_layernorm'):
                h = h + layer.post_attention_layernorm(attn_out)
            else:
                h = h + attn_out

            # FFN
            if hasattr(layer, 'pre_feedforward_layernorm'):
                h_ffn = layer.pre_feedforward_layernorm(h)
            else:
                h_ffn = layer.post_attention_layernorm(h) if hasattr(layer, 'post_attention_layernorm') else h

            ffn_out = layer.mlp(h_ffn)

            if hasattr(layer, 'post_feedforward_layernorm'):
                h = h + layer.post_feedforward_layernorm(ffn_out)
            else:
                h = h + ffn_out

            mx.eval(h)
        else:
            # Standard forward (no attention capture)
            h = layer(h, mask=mask)
            mx.eval(h)

        # Capture residual (last token)
        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return residuals, attn_info, pred_tok


# ---- Capture One Prompt -----------------------------------------------

def capture_one(model, tokenizer, prompt, gates, labels, config,
                syntax_range, knowledge_range, gate_threshold=3.0):
    """Run one prompt, return syntax features + attention head activity."""

    residuals, attn_info, pred_tok = forward_with_attention(
        model, tokenizer, prompt,
        syntax_layers=syntax_range,
        knowledge_layers=knowledge_range,
    )

    # Syntax features (L0-12): gate activations
    syntax_hits = []
    for layer in syntax_range:
        if layer not in residuals or layer not in gates:
            continue
        res = residuals[layer]
        layer_gates = gates[layer]  # [n_features, hidden_dim]
        acts = layer_gates @ res    # [n_features]

        top_idx = np.argsort(np.abs(acts))[-5:][::-1]
        for fi in top_idx:
            val = float(acts[fi])
            if abs(val) < gate_threshold:
                continue
            key = f"L{layer}_F{fi}"
            label_info = labels.get(key, {})
            rel = label_info.get("relation", "-") if isinstance(label_info, dict) else str(label_info)
            syntax_hits.append((layer, int(fi), val, rel))

    syntax_hits.sort(key=lambda x: abs(x[2]), reverse=True)
    syntax_hits = syntax_hits[:10]

    # Attention heads (L13-26): entropy + max weight
    attn_hits = []
    for (layer, head), info in sorted(attn_info.items()):
        attn_hits.append((layer, head, info["entropy"], info["max_w"]))

    attn_hits.sort(key=lambda x: x[3], reverse=True)
    attn_hits = attn_hits[:10]

    return {
        "syntax": syntax_hits,
        "attention": attn_hits,
        "prediction": pred_tok,
    }


# ---- Display -----------------------------------------------------------

def print_results(all_results):
    """Print the eyeball-it table."""

    # Per-prompt detail
    print(f"\n{'='*90}")
    print(f"PER-PROMPT RESULTS")
    print(f"{'='*90}")

    for cat, prompts in all_results.items():
        trigram = PROMPTS[cat]["trigram"]
        print(f"\n-- {cat.upper()} ({trigram}) --")

        for r in prompts:
            prompt_short = r["prompt"][:40].replace("\n", "\\n")
            print(f"\n  \"{prompt_short}\"  -> {r['prediction']}")

            print(f"    Syntax (L0-12):")
            for layer, feat, val, rel in r["syntax"][:5]:
                label = f"[{rel}]" if rel != "-" else ""
                print(f"      L{layer:2d} F{feat:<5d} {val:+7.1f}  {label}")

            print(f"    Attention (L13-26):")
            for layer, head, ent, maxw in r["attention"][:5]:
                print(f"      L{layer:2d} H{head}  entropy={ent:.2f}  max={maxw:.3f}")

    # Cross-category comparison
    print(f"\n{'='*90}")
    print(f"CROSS-CATEGORY COMPARISON")
    print(f"{'='*90}")

    # Syntax feature frequency per category
    print(f"\n  Most common syntax features per category:")
    for cat, prompts in all_results.items():
        feat_counts = defaultdict(int)
        for r in prompts:
            for layer, feat, val, rel in r["syntax"]:
                feat_counts[(layer, feat, rel)] += 1

        top = sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n    {cat}:")
        for (layer, feat, rel), count in top:
            label = f"[{rel}]" if rel != "-" else ""
            print(f"      L{layer}_F{feat} {label:20s} {count}/{len(prompts)} prompts")

    # Attention head frequency per category
    print(f"\n  Most active attention heads per category:")
    for cat, prompts in all_results.items():
        head_counts = defaultdict(int)
        for r in prompts:
            for layer, head, ent, maxw in r["attention"][:5]:
                head_counts[(layer, head)] += 1

        top = sorted(head_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n    {cat}:")
        for (layer, head), count in top:
            print(f"      L{layer}_H{head}  {count}/{len(prompts)} prompts")

    # Overlap analysis
    print(f"\n  Category overlap (shared features appearing in 3+ prompts):")
    cat_features = {}
    for cat, prompts in all_results.items():
        feat_counts = defaultdict(int)
        for r in prompts:
            for layer, feat, val, rel in r["syntax"]:
                feat_counts[(layer, feat)] += 1
        # features in 3+ prompts = "characteristic"
        cat_features[cat] = {k for k, v in feat_counts.items() if v >= 3}

    cats = list(cat_features.keys())
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            shared = cat_features[a] & cat_features[b]
            only_a = cat_features[a] - cat_features[b]
            only_b = cat_features[b] - cat_features[a]
            print(f"\n    {a} vs {b}:")
            print(f"      shared: {len(shared)}  only-{a}: {len(only_a)}  only-{b}: {len(only_b)}")
            if shared:
                for layer, feat in sorted(shared):
                    print(f"        L{layer}_F{feat}")

    # Verdict
    print(f"\n{'='*90}")
    print(f"VERDICT")
    print(f"{'='*90}")

    all_overlaps = []
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            union = cat_features[a] | cat_features[b]
            shared = cat_features[a] & cat_features[b]
            jaccard = len(shared) / len(union) if union else 0
            all_overlaps.append(jaccard)

    avg_overlap = np.mean(all_overlaps) if all_overlaps else 0

    if avg_overlap < 0.2:
        print(f"\n  CATEGORIES HAVE DISTINCT SYNTAX SIGNATURES")
        print(f"    Average Jaccard overlap: {avg_overlap:.3f}")
        print(f"    -> Syntax features likely predict attention circuits")
        print(f"    -> Scale up to full 240-prompt experiment")
    elif avg_overlap < 0.4:
        print(f"\n  ~ PARTIAL SEPARATION")
        print(f"    Average Jaccard overlap: {avg_overlap:.3f}")
        print(f"    -> Some routing signal, worth investigating further")
    else:
        print(f"\n  HIGH OVERLAP -- categories not well-separated")
        print(f"    Average Jaccard overlap: {avg_overlap:.3f}")
        print(f"    -> Syntax features may be too generic for routing")

    print()


# ---- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Minimal syntax->attention routing test (20 prompts)"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--gate-threshold", type=float, default=3.0,
                        help="Min absolute gate activation for syntax features")
    args = parser.parse_args()

    print("Loading vindex...")
    config, gates, labels, n_features = load_vindex(args.vindex)

    # Determine layer bands
    bands = config.get("layer_bands", {})
    syntax_end = bands.get("syntax", [0, 12])[1]
    knowledge_start = bands.get("knowledge", [13, 27])[0]
    knowledge_end = bands.get("knowledge", [13, 27])[1]

    syntax_range = range(0, syntax_end + 1)
    knowledge_range = range(knowledge_start, knowledge_end + 1)

    print(f"  Syntax layers: L0-L{syntax_end}")
    print(f"  Knowledge layers: L{knowledge_start}-L{knowledge_end}")

    print("Loading model...")
    model, tokenizer = load_model(args.model)

    total = sum(len(p["prompts"]) for p in PROMPTS.values())
    print(f"\nRunning {total} prompts across {len(PROMPTS)} categories...\n")

    all_results = {}
    n = 0
    t0 = time.time()

    for cat, info in PROMPTS.items():
        cat_results = []
        for prompt in info["prompts"]:
            try:
                result = capture_one(
                    model, tokenizer, prompt, gates, labels, config,
                    syntax_range, knowledge_range,
                    gate_threshold=args.gate_threshold,
                )
                result["prompt"] = prompt
                cat_results.append(result)
            except Exception as e:
                print(f"  ERROR: {prompt[:30]}... -> {e}")
                import traceback
                traceback.print_exc()

            n += 1
            elapsed = time.time() - t0
            print(f"\r  {n}/{total} ({elapsed:.0f}s)", end="", flush=True)

        all_results[cat] = cat_results

    print(f"\n  Done in {time.time()-t0:.0f}s")

    print_results(all_results)


if __name__ == "__main__":
    main()
