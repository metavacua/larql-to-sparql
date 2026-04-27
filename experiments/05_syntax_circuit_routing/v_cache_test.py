#!/usr/bin/env python3
"""
v_cache_test.py — Is the attention output cacheable per template?

THE QUESTION:
  For prompts within the same template (e.g. "The capital of France is"
  vs "The capital of Japan is"), does `attention_pattern @ V` produce
  similar outputs?

  If cosine > 0.9 → attention output is template-determined → fully cacheable
  If cosine < 0.5 → entity-specific computation is real → need hybrid

This tests the retrieval/computation boundary. FFN = retrieval (proven).
Template routing = retrieval (proven). Is within-template attention also
retrieval, or is it genuine computation?

CAPTURES per prompt:
  - attn_out[layer][head] = softmax(QK/sqrt(d)) @ V  (before o_proj)
  - attn_out_combined[layer] = o_proj(concat(head_outputs))
  - residual delta from attention (what attention writes into the stream)

USAGE:
  python3 experiments/05_syntax_circuit_routing/v_cache_test.py \
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
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Test prompts: same template, different entities --------------------

TEMPLATE_PROMPTS = {
    "capital_of": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Germany is",
        "The capital of India is",
        "The capital of Mexico is",
        "The capital of Australia is",
    ],
    "language_of": [
        "The official language of France is",
        "The official language of Japan is",
        "The official language of Brazil is",
        "The official language of China is",
        "The official language of Germany is",
        "The official language of Russia is",
    ],
    "synonym": [
        "Happy means",
        "Sad means",
        "Big means",
        "Fast means",
        "Brave means",
        "Cold means",
    ],
    "antonym": [
        "The opposite of happy is",
        "The opposite of big is",
        "The opposite of fast is",
        "The opposite of hot is",
        "The opposite of strong is",
        "The opposite of old is",
    ],
    "hypernym": [
        "A dog is a type of",
        "A rose is a type of",
        "A piano is a type of",
        "A hammer is a type of",
        "A sparrow is a type of",
        "A salmon is a type of",
    ],
    "arithmetic": [
        "2 + 3 =",
        "7 - 4 =",
        "5 * 6 =",
        "8 * 9 =",
        "15 + 27 =",
        "100 - 37 =",
    ],
    "birthplace_of": [
        "Einstein was born in",
        "Shakespeare was born in",
        "Mozart was born in",
        "Picasso was born in",
        "Darwin was born in",
        "Newton was born in",
    ],
    "comparison": [
        "An elephant is bigger than a",
        "A cheetah is faster than a",
        "The sun is hotter than the",
        "Gold is heavier than",
        "A diamond is harder than",
        "The Pacific is larger than the",
    ],
}


# ---- Model helpers ------------------------------------------------------

def find_model_parts(model):
    try:
        lm = model.language_model
        inner = lm.model
        if hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
            embed_fn = inner.embed_tokens
            def lm_head(h): return h @ embed_fn.weight.T
            return embed_fn, inner.layers, inner.norm, lm_head, True
    except AttributeError:
        pass
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            f = model.lm_head
            def lm_head(h): return f(h)
        else:
            def lm_head(h): return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale
    raise RuntimeError("Could not detect model structure.")


def forward_capture_v(model, tokenizer, prompt, knowledge_layers):
    """
    Forward pass capturing attention internals at knowledge layers:
      - per_head_out[layer][head]: (weights @ V)[last_token] per head, before o_proj
      - attn_delta[layer]: full attention contribution to residual (after o_proj + post_norm)
      - prediction token
    """
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    h = embed_fn(mx.array([tokens]))
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])
    seq_len = len(tokens)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    per_head_out = {}  # layer -> [n_heads, head_dim]
    attn_delta = {}    # layer -> [hidden_dim]
    attn_combined = {} # layer -> [hidden_dim] (o_proj output, before residual)

    for i, layer in enumerate(layers):
        if i in knowledge_layers:
            sa = layer.self_attn
            B, L, D = h.shape
            h_norm = layer.input_layernorm(h)

            q = sa.q_proj(h_norm)
            k = sa.k_proj(h_norm)
            v = sa.v_proj(h_norm)

            n_h = sa.n_heads
            n_kv = sa.n_kv_heads
            hd = sa.head_dim
            sc = sa.scale

            q = q.reshape(B,L,n_h,hd).transpose(0,2,1,3)
            k = k.reshape(B,L,n_kv,hd).transpose(0,2,1,3)
            v = v.reshape(B,L,n_kv,hd).transpose(0,2,1,3)

            if hasattr(sa,'q_norm'): q = sa.q_norm(q)
            if hasattr(sa,'k_norm'): k = sa.k_norm(k)
            q, k = sa.rope(q), sa.rope(k)

            # GQA expand
            if n_kv < n_h:
                reps = n_h // n_kv
                k = mx.repeat(k, reps, axis=1)
                v = mx.repeat(v, reps, axis=1)

            # Attention weights
            w = mx.softmax((q @ k.transpose(0,1,3,2)) * sc + mask, axis=-1)

            # Per-head: (weights @ V) for last token
            # w: [B, n_heads, L, L], v: [B, n_heads, L, head_dim]
            # wv: [B, n_heads, L, head_dim]
            wv = w @ v  # [B, n_heads, L, head_dim]

            # Capture per-head output at last token position
            wv_last = wv[0, :, -1, :]  # [n_heads, head_dim]
            mx.eval(wv_last)
            per_head_out[i] = np.array(wv_last.astype(mx.float32))

            # Combined through o_proj
            wv_combined = wv.transpose(0,2,1,3).reshape(B, L, -1)
            o_out = sa.o_proj(wv_combined)  # [B, L, hidden_dim]

            # Capture o_proj output at last token (before adding to residual)
            mx.eval(o_out)
            attn_combined[i] = np.array(o_out[0, -1, :].astype(mx.float32))

            # Post-attention norm + residual
            if hasattr(layer, 'post_attention_layernorm'):
                normed_o = layer.post_attention_layernorm(o_out)
                h_pre = h.copy() if hasattr(h, 'copy') else h
                h = h + normed_o
            else:
                h = h + o_out

            mx.eval(h)

            # Attention delta = what attention wrote into the stream
            attn_delta[i] = np.array(h[0, -1, :].astype(mx.float32)) - \
                           np.array(h_pre[0, -1, :].astype(mx.float32)) if 'h_pre' in dir() else attn_combined[i]

            # FFN
            if hasattr(layer, 'pre_feedforward_layernorm'):
                hf = layer.pre_feedforward_layernorm(h)
            else:
                hf = h
            fo = layer.mlp(hf)
            if hasattr(layer, 'post_feedforward_layernorm'):
                h = h + layer.post_feedforward_layernorm(fo)
            else:
                h = h + fo
            mx.eval(h)
        else:
            h = layer(h, mask=mask)
            mx.eval(h)

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0,0,:].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return per_head_out, attn_combined, pred_tok


# ---- Analysis -----------------------------------------------------------

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def analyze_template(template_name, captures, knowledge_layers, n_heads):
    """
    For one template, compute within-template similarity of attention outputs.
    """
    n = len(captures)

    results = {
        "per_head": {},     # (layer, head) -> within-cosine stats
        "combined": {},     # layer -> within-cosine of o_proj output
    }

    # Per-head analysis
    for layer in knowledge_layers:
        for head in range(n_heads):
            vecs = []
            for cap in captures:
                if layer in cap["per_head"]:
                    vecs.append(cap["per_head"][layer][head])

            if len(vecs) < 2:
                continue

            # Pairwise cosine
            sims = []
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    sims.append(cosine(vecs[i], vecs[j]))

            results["per_head"][(layer, head)] = {
                "mean": float(np.mean(sims)),
                "min": float(np.min(sims)),
                "std": float(np.std(sims)),
            }

    # Combined (o_proj) analysis
    for layer in knowledge_layers:
        vecs = []
        for cap in captures:
            if layer in cap["combined"]:
                vecs.append(cap["combined"][layer])

        if len(vecs) < 2:
            continue

        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(cosine(vecs[i], vecs[j]))

        results["combined"][layer] = {
            "mean": float(np.mean(sims)),
            "min": float(np.min(sims)),
            "std": float(np.std(sims)),
        }

    return results


def analyze_between_templates(all_captures, knowledge_layers, n_heads):
    """Compute between-template similarity for reference."""
    templates = list(all_captures.keys())

    between_head = defaultdict(list)
    between_combined = defaultdict(list)

    for i, ta in enumerate(templates):
        for tb in templates[i+1:]:
            for cap_a in all_captures[ta]:
                for cap_b in all_captures[tb]:
                    for layer in knowledge_layers:
                        # Combined
                        if layer in cap_a["combined"] and layer in cap_b["combined"]:
                            s = cosine(cap_a["combined"][layer], cap_b["combined"][layer])
                            between_combined[layer].append(s)

                        # Sample a few heads
                        for head in [0, 3, 7]:
                            if layer in cap_a["per_head"] and layer in cap_b["per_head"]:
                                s = cosine(
                                    cap_a["per_head"][layer][head],
                                    cap_b["per_head"][layer][head]
                                )
                                between_head[(layer, head)].append(s)

    return between_combined, between_head


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V-cache test: is attention output cacheable per template?"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config for layer bands
    with open(Path(args.vindex) / "index.json") as f:
        config = json.load(f)
    bands = config.get("layer_bands", {})
    kn_start = bands.get("knowledge", [14, 27])[0]
    kn_end = bands.get("knowledge", [14, 27])[1]
    knowledge_layers = range(kn_start, kn_end + 1)

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    print(f"  Model: {args.model}")
    print(f"  Knowledge layers: L{kn_start}-L{kn_end}")

    # Detect n_heads
    try:
        n_heads = model.language_model.model.layers[0].self_attn.n_heads
    except:
        n_heads = model.model.layers[0].self_attn.n_heads
    print(f"  Attention heads: {n_heads}")

    # ---- Capture ----
    total = sum(len(v) for v in TEMPLATE_PROMPTS.values())
    print(f"\nCapturing attention outputs ({total} prompts, {len(TEMPLATE_PROMPTS)} templates)...")
    t0 = time.time()
    n = 0

    all_captures = {}
    for template_name, prompts in TEMPLATE_PROMPTS.items():
        caps = []
        for prompt in prompts:
            per_head, combined, pred = forward_capture_v(
                model, tokenizer, prompt, knowledge_layers
            )
            caps.append({
                "prompt": prompt,
                "per_head": per_head,
                "combined": combined,
                "prediction": pred,
            })
            n += 1
            print(f"\r  {n}/{total} ({time.time()-t0:.0f}s) last: {pred}", end="", flush=True)

        all_captures[template_name] = caps

    print(f"\n  Done in {time.time()-t0:.0f}s")

    # ---- Within-template analysis ----
    print(f"\n{'='*70}")
    print(f"WITHIN-TEMPLATE SIMILARITY (attention_pattern @ V)")
    print(f"{'='*70}")

    all_within = {}
    for template_name, caps in all_captures.items():
        result = analyze_template(template_name, caps, knowledge_layers, n_heads)
        all_within[template_name] = result

    # Combined (o_proj output) per template per layer
    print(f"\n  o_proj output similarity (within template):")
    print(f"  {'Template':<15s}", end="")
    sample_layers = [kn_start, kn_start+3, kn_start+7, kn_end-3, kn_end]
    sample_layers = [l for l in sample_layers if l in knowledge_layers]
    for l in sample_layers:
        print(f"  L{l:2d}   ", end="")
    print()

    template_layer_sims = {}
    for template_name in TEMPLATE_PROMPTS:
        result = all_within[template_name]
        print(f"  {template_name:<15s}", end="")
        for l in sample_layers:
            if l in result["combined"]:
                mean = result["combined"][l]["mean"]
                print(f"  {mean:.3f} ", end="")
                template_layer_sims.setdefault(l, []).append(mean)
            else:
                print(f"    -   ", end="")
        print()

    # Average across templates per layer
    print(f"  {'AVERAGE':<15s}", end="")
    for l in sample_layers:
        if l in template_layer_sims:
            avg = np.mean(template_layer_sims[l])
            print(f"  {avg:.3f} ", end="")
    print()

    # Full layer sweep (averages)
    print(f"\n  Full layer sweep (avg within-template o_proj cosine):")
    layer_avgs = {}
    for layer in knowledge_layers:
        sims = []
        for template_name in TEMPLATE_PROMPTS:
            result = all_within[template_name]
            if layer in result["combined"]:
                sims.append(result["combined"][layer]["mean"])
        if sims:
            avg = np.mean(sims)
            layer_avgs[layer] = avg
            bar = "#" * int(avg * 30)
            print(f"    L{layer:2d}: {avg:.4f}  {bar}")

    # Per-head analysis (show most and least similar heads)
    print(f"\n  Per-head similarity (avg across templates):")
    head_avgs = defaultdict(list)
    for template_name in TEMPLATE_PROMPTS:
        result = all_within[template_name]
        for (layer, head), stats in result["per_head"].items():
            head_avgs[(layer, head)].append(stats["mean"])

    head_summary = []
    for (layer, head), sims in head_avgs.items():
        head_summary.append((layer, head, np.mean(sims), np.min(sims)))

    # Most template-determined (highest within-template similarity)
    head_summary.sort(key=lambda x: x[2], reverse=True)
    print(f"\n    Most template-determined (highest within-template cosine):")
    for layer, head, avg, mn in head_summary[:10]:
        print(f"      L{layer:2d} H{head}: mean={avg:.4f}  min={mn:.4f}")

    # Most entity-specific (lowest within-template similarity)
    print(f"\n    Most entity-specific (lowest within-template cosine):")
    for layer, head, avg, mn in head_summary[-10:]:
        print(f"      L{layer:2d} H{head}: mean={avg:.4f}  min={mn:.4f}")

    # ---- Between-template analysis ----
    print(f"\n{'='*70}")
    print(f"BETWEEN-TEMPLATE SIMILARITY (reference)")
    print(f"{'='*70}")

    between_combined, between_head = analyze_between_templates(
        all_captures, knowledge_layers, n_heads
    )

    print(f"\n  Between-template o_proj cosine (different templates):")
    for layer in sample_layers:
        if layer in between_combined:
            avg = np.mean(between_combined[layer])
            print(f"    L{layer:2d}: {avg:.4f}")

    # ---- Gap analysis ----
    print(f"\n{'='*70}")
    print(f"GAP: WITHIN-TEMPLATE vs BETWEEN-TEMPLATE")
    print(f"{'='*70}")

    print(f"\n  Layer  Within    Between   Gap       Verdict")
    print(f"  -----  --------  --------  --------  -------")

    layer_verdicts = {}
    for layer in knowledge_layers:
        within_sims = []
        for template_name in TEMPLATE_PROMPTS:
            result = all_within[template_name]
            if layer in result["combined"]:
                within_sims.append(result["combined"][layer]["mean"])

        within_avg = np.mean(within_sims) if within_sims else 0
        between_avg = np.mean(between_combined.get(layer, [0]))
        gap = within_avg - between_avg

        if within_avg > 0.8:
            verdict = "CACHEABLE"
        elif within_avg > 0.5:
            verdict = "PARTIAL"
        else:
            verdict = "COMPUTED"

        layer_verdicts[layer] = {
            "within": within_avg,
            "between": between_avg,
            "gap": gap,
            "verdict": verdict,
        }

        print(f"  L{layer:2d}   {within_avg:.4f}    {between_avg:.4f}    {gap:+.4f}   {verdict}")

    # ---- Per-template detail ----
    print(f"\n{'='*70}")
    print(f"PER-TEMPLATE DETAIL")
    print(f"{'='*70}")

    for template_name, caps in all_captures.items():
        result = all_within[template_name]

        # Average across knowledge layers
        combined_sims = [v["mean"] for v in result["combined"].values()]
        avg_sim = np.mean(combined_sims) if combined_sims else 0
        min_sim = np.min(combined_sims) if combined_sims else 0

        predictions = [c["prediction"] for c in caps]
        pred_str = ", ".join(predictions[:4]) + ("..." if len(predictions) > 4 else "")

        print(f"\n  {template_name}:")
        print(f"    Avg within-cosine: {avg_sim:.4f}  min: {min_sim:.4f}")
        print(f"    Predictions: {pred_str}")

        # Show which layers are most/least cacheable for this template
        sorted_layers = sorted(result["combined"].items(), key=lambda x: x[1]["mean"])
        worst = sorted_layers[:2]
        best = sorted_layers[-2:]
        print(f"    Least cacheable: {', '.join(f'L{l}({v['mean']:.3f})' for l,v in worst)}")
        print(f"    Most cacheable:  {', '.join(f'L{l}({v['mean']:.3f})' for l,v in best)}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    overall_within = np.mean([v["within"] for v in layer_verdicts.values()])
    overall_between = np.mean([v["between"] for v in layer_verdicts.values()])
    overall_gap = overall_within - overall_between

    n_cacheable = sum(1 for v in layer_verdicts.values() if v["verdict"] == "CACHEABLE")
    n_partial = sum(1 for v in layer_verdicts.values() if v["verdict"] == "PARTIAL")
    n_computed = sum(1 for v in layer_verdicts.values() if v["verdict"] == "COMPUTED")

    print(f"\n  Overall within-template cosine:  {overall_within:.4f}")
    print(f"  Overall between-template cosine: {overall_between:.4f}")
    print(f"  Gap: {overall_gap:+.4f}")
    print(f"\n  Layer verdicts: {n_cacheable} CACHEABLE, {n_partial} PARTIAL, {n_computed} COMPUTED")

    if overall_within > 0.8:
        print(f"\n  ATTENTION IS RETRIEVAL")
        print(f"    Within-template attention outputs are highly similar ({overall_within:.3f})")
        print(f"    V output is template-determined, not entity-specific")
        print(f"    -> Full V-cache is viable. Entire attention elimination possible.")
    elif overall_within > 0.5:
        print(f"\n  ATTENTION IS MIXED RETRIEVAL + COMPUTATION")
        print(f"    Template determines most of the attention output ({overall_within:.3f})")
        print(f"    But entity-specific variation is significant")
        print(f"    -> Template V-cache + entity correction term")
    else:
        print(f"\n  ATTENTION IS COMPUTATION")
        print(f"    Attention outputs vary significantly within templates ({overall_within:.3f})")
        print(f"    Entity-specific information dominates")
        print(f"    -> V-cache not viable. Attention must be computed per-input.")
        print(f"    -> Routing table still eliminates K+QK (62% coverage)")
        print(f"       but V computation remains")

    print()

    # ---- Save ----
    save_data = {
        "layer_verdicts": {str(k): v for k, v in layer_verdicts.items()},
        "overall_within": overall_within,
        "overall_between": overall_between,
        "overall_gap": overall_gap,
        "n_cacheable": n_cacheable,
        "n_partial": n_partial,
        "n_computed": n_computed,
    }
    with open(output_dir / "v_cache_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {output_dir / 'v_cache_results.json'}")


if __name__ == "__main__":
    main()
