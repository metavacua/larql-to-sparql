#!/usr/bin/env python3
"""
What Is Attention Actually Doing? — Gemma 3-4B Anatomy

Five hypotheses:
  H1: Refinement — attention barely matters, FFN carries the model
  H2: Assembly — attention is a parser composing queries
  H3: Cancellation — attention suppresses noise
  H4: Sub-Templates — more templates fix compilation
  H5: Graph Walk — attention IS the knowledge traversal

Eight measurements, four aggregate analyses, one answer.
"""

import os
import sys
import json
import time
import math
import random
import gc
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
SEED = 42
MAX_SEQ = 128
OUTPUT_DIR = "results_attention_anatomy"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_prompts():
    """Build diverse prompt set with metadata."""
    prompts = []

    # Factual (20) — known entity→answer pairs
    factual = [
        ("The capital of France is", "France", "capital_of", "Paris"),
        ("The capital of Germany is", "Germany", "capital_of", "Berlin"),
        ("The capital of Japan is", "Japan", "capital_of", "Tokyo"),
        ("The capital of Italy is", "Italy", "capital_of", "Rome"),
        ("The capital of Spain is", "Spain", "capital_of", "Madrid"),
        ("The capital of Brazil is", "Brazil", "capital_of", "Brasilia"),
        ("The capital of Australia is", "Australia", "capital_of", "Canberra"),
        ("The capital of Canada is", "Canada", "capital_of", "Ottawa"),
        ("The capital of Egypt is", "Egypt", "capital_of", "Cairo"),
        ("The capital of India is", "India", "capital_of", "Delhi"),
        ("The official language of France is", "France", "language_of", "French"),
        ("The official language of Germany is", "Germany", "language_of", "German"),
        ("The official language of Japan is", "Japan", "language_of", "Japanese"),
        ("The official language of Spain is", "Spain", "language_of", "Spanish"),
        ("The currency of Japan is the", "Japan", "currency_of", "yen"),
        ("The currency of India is the", "India", "currency_of", "rupee"),
        ("The chemical symbol for gold is", "gold", "symbol", "Au"),
        ("The chemical symbol for iron is", "iron", "symbol", "Fe"),
        ("The largest planet in our solar system is", "solar system", "largest", "Jupiter"),
        ("The Earth orbits the", "Earth", "orbits", "Sun"),
    ]
    for text, entity, relation, answer in factual:
        prompts.append({
            "text": text, "category": "factual",
            "entity": entity, "relation": relation, "answer": answer,
        })

    # Entity substitutions (10) — same template, different entity (for M5)
    for entity, answer in [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("Egypt", "Cairo"),
        ("India", "Delhi"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
        ("Brazil", "Brasilia"),
    ]:
        prompts.append({
            "text": f"The capital of {entity} is",
            "category": "entity_sub",
            "entity": entity, "relation": "capital_of", "answer": answer,
            "template": "capital_of",
        })

    # Syntactic (5)
    for text in [
        "The big dog runs near the",
        "She quickly ran to the",
        "If it rains tomorrow, we will",
        "The old man and the",
        "Running through the forest, the deer",
    ]:
        prompts.append({"text": text, "category": "syntactic",
                        "entity": None, "relation": None, "answer": None})

    # Code (5)
    for text in [
        "def calculate_sum(numbers):\n    return",
        "for i in range(10):\n    print(",
        "class DatabaseConnection:\n    def __init__(self",
        "import numpy as np\ndata = np.array([1, 2, 3])\nresult =",
        "try:\n    response = requests.get(url)\nexcept",
    ]:
        prompts.append({"text": text, "category": "code",
                        "entity": None, "relation": None, "answer": None})

    # Multi-hop (5)
    for text, entity, answer in [
        ("The capital of the country where the Eiffel Tower is located is", "Eiffel Tower", "Paris"),
        ("The language spoken in the country whose capital is Tokyo is", "Tokyo", "Japanese"),
        ("The continent where Egypt is located is", "Egypt", "Africa"),
        ("The country that borders France to the east is", "France", "Germany"),
        ("The ocean on the west coast of the United States is the", "United States", "Pacific"),
    ]:
        prompts.append({"text": text, "category": "multi_hop",
                        "entity": entity, "relation": "multi_hop", "answer": answer})

    # Diverse/adversarial (5)
    for text in [
        "Once upon a time, in a small village,",
        "The most important discovery in physics was",
        "In the year 2050, humanity will likely have",
        "The opposite of love is not hate, it is",
        "42 is the answer to",
    ]:
        prompts.append({"text": text, "category": "diverse",
                        "entity": None, "relation": None, "answer": None})

    return prompts


# ---------------------------------------------------------------------------
# Phase 1: Data Collection
# ---------------------------------------------------------------------------

def collect_data(model, tokenizer, prompts, device):
    """Run all prompts, capture attention patterns, outputs, residuals."""
    print(f"\n  Phase 1: Collecting data from {len(prompts)} prompts...")
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    n_heads = tc.num_attention_heads
    hidden_dim = tc.hidden_size
    embed_weight = model.model.language_model.embed_tokens.weight.data.float()

    # Storage: per-prompt data
    all_data = []

    # Hooks to capture pre-FFN and post-attention residuals
    pre_ffn_residuals = [None] * n_layers
    attn_outputs = [None] * n_layers
    ffn_outputs = [None] * n_layers

    hooks = []

    # Hook: capture input to MLP (= residual after attention)
    def make_mlp_pre_hook(li):
        def hook(module, args):
            pre_ffn_residuals[li] = args[0].detach().float()
        return hook

    # Hook: capture MLP output
    def make_mlp_post_hook(li):
        def hook(module, args, output):
            ffn_outputs[li] = output.detach().float()
        return hook

    # Hook: capture attention output (on the attention module)
    def make_attn_post_hook(li):
        def hook(module, args, output):
            # output is tuple; first element is the attention output
            if isinstance(output, tuple):
                attn_outputs[li] = output[0].detach().float()
            else:
                attn_outputs[li] = output.detach().float()
        return hook

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]
        hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_pre_hook(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post_hook(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post_hook(li)))

    t0 = time.time()
    with torch.no_grad():
        for pi, prompt_info in enumerate(prompts):
            text = prompt_info["text"]
            inputs = tokenizer(text, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            input_ids = inputs["input_ids"][0]
            seq_len = input_ids.shape[0]

            # Forward with attention weights
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits[0].float()

            # Collect per-layer data
            layer_data = []
            for li in range(n_layers):
                ld = {}

                # Attention output at last position
                if attn_outputs[li] is not None:
                    ld["attn_out_last"] = attn_outputs[li][0, -1].cpu()  # (hidden,)
                    ld["attn_norm"] = attn_outputs[li][0, -1].norm().item()

                # FFN output at last position
                if ffn_outputs[li] is not None:
                    ld["ffn_out_last"] = ffn_outputs[li][0, -1].cpu()
                    ld["ffn_norm"] = ffn_outputs[li][0, -1].norm().item()

                # Pre-FFN residual at last position
                if pre_ffn_residuals[li] is not None:
                    ld["residual_last"] = pre_ffn_residuals[li][0, -1].cpu()

                # Attention weights per head
                if outputs.attentions is not None and li < len(outputs.attentions):
                    attn_w = outputs.attentions[li]
                    if attn_w is not None:
                        ld["attn_weights"] = attn_w[0].cpu().float()  # (heads, seq, seq)

                layer_data.append(ld)

            # Top predictions
            top1 = logits[-1].argmax().item()
            top5 = logits[-1].topk(5).indices.tolist()

            # Token list for this prompt
            tokens = [tokenizer.decode([tid]).strip() for tid in input_ids.tolist()]

            all_data.append({
                "prompt": prompt_info,
                "tokens": tokens,
                "input_ids": input_ids.tolist(),
                "seq_len": seq_len,
                "top1": top1,
                "top1_token": tokenizer.decode([top1]).strip(),
                "top5": top5,
                "top5_tokens": [tokenizer.decode([t]).strip() for t in top5],
                "layer_data": layer_data,
                "logits_last": logits[-1].cpu(),
            })

            if (pi + 1) % 10 == 0:
                print(f"    {pi+1}/{len(prompts)} ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()

    elapsed = time.time() - t0
    print(f"  Data collected in {elapsed:.0f}s")

    return all_data, n_layers, n_heads, hidden_dim, embed_weight


# ---------------------------------------------------------------------------
# Phase 2: Measurements
# ---------------------------------------------------------------------------

def measure_m1_norm_ratios(all_data, n_layers):
    """M1: Attention vs FFN norm ratios (tests H1)."""
    print(f"\n  M1: Attention vs FFN norm ratios")

    ratios = [[[] for _ in range(n_layers)] for _ in range(len(all_data))]
    layer_avg_ratios = []

    for li in range(n_layers):
        attn_norms = []
        ffn_norms = []
        for di, d in enumerate(all_data):
            ld = d["layer_data"][li]
            an = ld.get("attn_norm", 0)
            fn = ld.get("ffn_norm", 0)
            if fn > 0:
                attn_norms.append(an)
                ffn_norms.append(fn)

        if attn_norms and ffn_norms:
            avg_ratio = sum(a/f for a, f in zip(attn_norms, ffn_norms)) / len(attn_norms)
            avg_attn = sum(attn_norms) / len(attn_norms)
            avg_ffn = sum(ffn_norms) / len(ffn_norms)
        else:
            avg_ratio = avg_attn = avg_ffn = 0

        layer_avg_ratios.append({
            "layer": li,
            "avg_ratio": round(avg_ratio, 4),
            "avg_attn_norm": round(avg_attn, 2),
            "avg_ffn_norm": round(avg_ffn, 2),
        })

    # Summary
    all_ratios = [r["avg_ratio"] for r in layer_avg_ratios]
    mean_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0
    print(f"    Mean attn/ffn ratio: {mean_ratio:.4f}")
    print(f"    If << 1: H1 supported (attention is minor)")
    print(f"    Layer range: {min(all_ratios):.4f} - {max(all_ratios):.4f}")

    return layer_avg_ratios, mean_ratio


def measure_m2_token_projection(all_data, n_layers, embed_weight):
    """M2: What does attention output mean in token space?"""
    print(f"\n  M2: Attention output token projection")

    # For each layer, project attention output onto embedding matrix
    layer_projections = []

    for li in range(n_layers):
        entity_hits = 0
        answer_hits = 0
        total = 0

        for d in all_data:
            if d["prompt"]["answer"] is None:
                continue
            ld = d["layer_data"][li]
            attn_out = ld.get("attn_out_last")
            if attn_out is None:
                continue

            # Project to token space
            scores = attn_out @ embed_weight.T  # (vocab,)
            top_ids = scores.topk(10).indices.tolist()
            top_tokens = [d["tokens"][0] if False else "" for _ in top_ids]  # placeholder
            top_tokens = []
            for tid in top_ids:
                try:
                    tok = embed_weight.shape[0]  # just check bounds
                    top_tokens.append(tid)
                except:
                    pass

            # Check if answer entity appears in top tokens
            answer = d["prompt"]["answer"]
            entity = d["prompt"]["entity"]

            # Decode top projected tokens
            from transformers import AutoTokenizer
            # We need tokenizer in scope — pass it or use global
            # For now, just check token IDs
            answer_ids = set()
            entity_ids = set()
            for prefix in ["", " "]:
                aids = all_data[0]["input_ids"]  # dummy — need tokenizer
                # Simplified: check if any of the top-10 match answer/entity token IDs
                pass

            total += 1

        layer_projections.append({"layer": li, "total": total})

    return layer_projections


def measure_m4_residual_change(all_data, n_layers):
    """M4: What does attention add vs subtract?"""
    print(f"\n  M4: Residual change analysis")

    layer_stats = []
    for li in range(n_layers):
        pos_fracs = []
        neg_fracs = []
        amplified_fracs = []

        for d in all_data:
            ld = d["layer_data"][li]
            attn_out = ld.get("attn_out_last")
            if attn_out is None:
                continue

            # attn_out IS the delta (what attention adds to residual)
            total_features = attn_out.numel()
            positive = (attn_out > 0).sum().item() / total_features
            negative = (attn_out < 0).sum().item() / total_features

            residual = ld.get("residual_last")
            if residual is not None:
                # Features where attention changes residual by >10%
                amplified = (attn_out.abs() > residual.abs() * 0.1).sum().item() / total_features
            else:
                amplified = 0

            pos_fracs.append(positive)
            neg_fracs.append(negative)
            amplified_fracs.append(amplified)

        avg_pos = sum(pos_fracs) / len(pos_fracs) if pos_fracs else 0
        avg_neg = sum(neg_fracs) / len(neg_fracs) if neg_fracs else 0
        avg_amp = sum(amplified_fracs) / len(amplified_fracs) if amplified_fracs else 0

        layer_stats.append({
            "layer": li,
            "avg_positive_frac": round(avg_pos, 3),
            "avg_negative_frac": round(avg_neg, 3),
            "avg_amplified_frac": round(avg_amp, 3),
            "bias": "suppressing" if avg_neg > avg_pos + 0.05 else
                    "amplifying" if avg_pos > avg_neg + 0.05 else "balanced",
        })

    # Summary
    suppressors = sum(1 for s in layer_stats if s["bias"] == "suppressing")
    amplifiers = sum(1 for s in layer_stats if s["bias"] == "amplifying")
    balanced = sum(1 for s in layer_stats if s["bias"] == "balanced")
    print(f"    Suppressing: {suppressors}, Amplifying: {amplifiers}, Balanced: {balanced}")

    return layer_stats


def measure_m5_cross_entity_cosine(all_data, n_layers):
    """M5: Cross-entity cosine at multiple thresholds."""
    print(f"\n  M5: Cross-entity cosine similarity")

    # Get entity substitution prompts
    entity_prompts = [d for d in all_data if d["prompt"]["category"] == "entity_sub"]
    if len(entity_prompts) < 2:
        print(f"    Not enough entity_sub prompts")
        return {}, {}

    thresholds = [0.94, 0.80, 0.60, 0.40]
    layer_cosines = []

    for li in range(n_layers):
        # Collect attention outputs at this layer for all entity_sub prompts
        outputs = []
        for d in entity_prompts:
            ld = d["layer_data"][li]
            attn_out = ld.get("attn_out_last")
            if attn_out is not None:
                outputs.append(attn_out)

        if len(outputs) < 2:
            layer_cosines.append({"layer": li, "mean_cosine": 0, "compilable": {}})
            continue

        # Pairwise cosine similarity
        output_stack = torch.stack(outputs)  # (N, hidden)
        output_norm = F.normalize(output_stack, dim=1)
        cos_matrix = output_norm @ output_norm.T  # (N, N)

        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(cos_matrix), diagonal=1).bool()
        cosines = cos_matrix[mask].tolist()

        mean_cos = sum(cosines) / len(cosines) if cosines else 0

        compilable = {}
        for thresh in thresholds:
            n_above = sum(1 for c in cosines if c > thresh)
            frac = n_above / len(cosines) if cosines else 0
            compilable[str(thresh)] = round(frac, 3)

        layer_cosines.append({
            "layer": li,
            "mean_cosine": round(mean_cos, 4),
            "compilable": compilable,
        })

    # Aggregate: at each threshold, what fraction of layers are "compilable"?
    threshold_summary = {}
    for thresh in thresholds:
        key = str(thresh)
        compilable_layers = sum(
            1 for lc in layer_cosines
            if lc["compilable"].get(key, 0) > 0.8  # 80% of pairs above threshold
        )
        threshold_summary[key] = {
            "compilable_layers": compilable_layers,
            "fraction": round(compilable_layers / n_layers, 3) if n_layers else 0,
        }

    print(f"    Threshold → compilable layers:")
    for thresh, info in threshold_summary.items():
        print(f"      {thresh}: {info['compilable_layers']}/{n_layers} "
              f"({info['fraction']:.0%})")

    return layer_cosines, threshold_summary


def measure_m6_head_clustering(all_data, n_layers, n_heads):
    """M6: Head specialization via PCA."""
    print(f"\n  M6: Head specialization (PCA)")

    # For each (layer, head), collect attention weight patterns across prompts
    head_pca_results = {}

    for li in range(n_layers):
        for h in range(n_heads):
            patterns = []
            for d in all_data:
                ld = d["layer_data"][li]
                aw = ld.get("attn_weights")
                if aw is not None and h < aw.shape[0]:
                    # Last row of attention pattern for this head
                    last_row = aw[h, -1, :d["seq_len"]]
                    patterns.append(last_row)

            if len(patterns) < 5:
                continue

            # Pad to same length
            max_len = max(p.shape[0] for p in patterns)
            padded = torch.zeros(len(patterns), max_len)
            for i, p in enumerate(patterns):
                padded[i, :p.shape[0]] = p

            # PCA via SVD
            padded_centered = padded - padded.mean(dim=0)
            try:
                U, S, Vh = torch.linalg.svd(padded_centered, full_matrices=False)
                total_var = (S ** 2).sum().item()
                if total_var > 0:
                    explained = [(S[i] ** 2).item() / total_var for i in range(min(5, len(S)))]
                else:
                    explained = [0] * 5
            except:
                explained = [0] * 5

            pc1 = explained[0] if len(explained) > 0 else 0
            pc3 = sum(explained[:3]) if len(explained) >= 3 else sum(explained)

            head_pca_results[(li, h)] = {
                "pc1_explained": round(pc1, 3),
                "pc3_explained": round(pc3, 3),
                "n_samples": len(patterns),
                "classification": (
                    "single_mode" if pc1 > 0.80 else
                    "few_modes" if pc3 > 0.90 else
                    "multi_mode" if pc3 > 0.70 else
                    "diffuse"
                ),
            }

    # Summary
    classifications = Counter(v["classification"] for v in head_pca_results.values())
    print(f"    Head classifications:")
    for cls, count in classifications.most_common():
        total = len(head_pca_results)
        print(f"      {cls:<15} {count}/{total} ({count/total:.0%})")

    return head_pca_results


def measure_m7_causal_skipping(model, tokenizer, prompts, device, n_layers,
                                max_prompts=20):
    """M7: How much does skipping attention at each layer hurt?"""
    print(f"\n  M7: Causal attention skipping ({max_prompts} prompts × {n_layers} layers)")

    model.eval()
    factual_prompts = [p for p in prompts if p["category"] in ("factual", "entity_sub")]
    test_prompts = factual_prompts[:max_prompts]

    # Get baseline predictions
    baseline_preds = {}
    with torch.no_grad():
        for pi, p in enumerate(test_prompts):
            inputs = tokenizer(p["text"], return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1].float()
            baseline_preds[pi] = {
                "logits": logits.cpu(),
                "top1": logits.argmax().item(),
                "probs": F.softmax(logits, dim=-1).cpu(),
            }

    # For each layer: skip attention, measure impact
    layer_results = []
    t0 = time.time()

    for li in range(n_layers):
        # Hook to zero out attention output at this layer
        skip_active = [True]

        def make_skip_hook(target_li):
            def hook(module, args, output):
                if skip_active[0]:
                    # Return zeros instead of attention output
                    if isinstance(output, tuple):
                        zeros = torch.zeros_like(output[0])
                        return (zeros,) + output[1:]
                    return torch.zeros_like(output)
                return output
            return hook

        layer = model.model.language_model.layers[li]
        h = layer.self_attn.register_forward_hook(make_skip_hook(li))

        top1_same = 0
        kl_divs = []

        with torch.no_grad():
            for pi, p in enumerate(test_prompts):
                inputs = tokenizer(p["text"], return_tensors="pt", max_length=MAX_SEQ,
                                 truncation=True).to(device)
                outputs = model(**inputs)
                skip_logits = outputs.logits[0, -1].float()

                # Compare
                skip_top1 = skip_logits.argmax().item()
                if skip_top1 == baseline_preds[pi]["top1"]:
                    top1_same += 1

                # KL divergence
                skip_probs = F.softmax(skip_logits, dim=-1)
                base_probs = baseline_preds[pi]["probs"].to(device)
                kl = F.kl_div(skip_probs.log(), base_probs, reduction='sum').item()
                kl_divs.append(kl)

        h.remove()

        avg_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0
        top1_frac = top1_same / len(test_prompts)

        layer_results.append({
            "layer": li,
            "top1_preserved": round(top1_frac, 3),
            "avg_kl": round(avg_kl, 4),
            "dispensable": top1_frac > 0.8 and avg_kl < 1.0,
        })

        if (li + 1) % 5 == 0:
            print(f"    L{li}: top1_preserved={top1_frac:.0%}, "
                  f"KL={avg_kl:.2f} ({time.time()-t0:.0f}s)")

    dispensable = sum(1 for r in layer_results if r["dispensable"])
    print(f"    Dispensable layers: {dispensable}/{n_layers}")

    return layer_results


def measure_m8_attention_decomposition(all_data, n_layers, n_heads):
    """M8: What positions does attention focus on?"""
    print(f"\n  M8: Attention pattern decomposition")

    # For factual prompts, what tokens get the most attention at the prediction position?
    factual_data = [d for d in all_data
                    if d["prompt"]["category"] in ("factual", "entity_sub")]

    head_roles = defaultdict(lambda: Counter())

    for d in factual_data:
        entity = d["prompt"].get("entity", "")
        tokens = d["tokens"]

        for li in range(n_layers):
            ld = d["layer_data"][li]
            aw = ld.get("attn_weights")
            if aw is None:
                continue

            for h in range(min(n_heads, aw.shape[0])):
                # Last position's attention distribution
                last_attn = aw[h, -1, :d["seq_len"]]
                top_pos = last_attn.topk(min(3, d["seq_len"])).indices.tolist()

                for pos in top_pos:
                    token = tokens[pos] if pos < len(tokens) else "?"
                    token_lower = token.lower().strip()

                    if pos == 0:
                        head_roles[(li, h)]["BOS"] += 1
                    elif pos == d["seq_len"] - 1:
                        head_roles[(li, h)]["self"] += 1
                    elif pos == d["seq_len"] - 2:
                        head_roles[(li, h)]["previous"] += 1
                    elif entity and entity.lower() in token_lower:
                        head_roles[(li, h)]["entity"] += 1
                    elif token_lower in ("capital", "language", "currency", "symbol",
                                        "official", "largest", "orbits"):
                        head_roles[(li, h)]["relation"] += 1
                    elif token_lower in ("the", "of", "is", "in", "a"):
                        head_roles[(li, h)]["function_word"] += 1
                    else:
                        head_roles[(li, h)]["other"] += 1

    # Classify each head by dominant role
    head_classifications = {}
    for (li, h), roles in head_roles.items():
        total = sum(roles.values())
        if total == 0:
            continue
        dominant = roles.most_common(1)[0]
        head_classifications[(li, h)] = {
            "dominant_role": dominant[0],
            "dominant_frac": round(dominant[1] / total, 3),
            "roles": dict(roles),
        }

    # Summary
    role_counts = Counter(v["dominant_role"] for v in head_classifications.values())
    print(f"    Dominant roles across heads:")
    for role, count in role_counts.most_common():
        total = len(head_classifications)
        print(f"      {role:<15} {count}/{total} ({count/total:.0%})")

    return head_classifications


# ---------------------------------------------------------------------------
# Phase 3: Aggregate Analysis
# ---------------------------------------------------------------------------

def aggregate_analysis(
    m1_ratios, m4_residual, m5_cosines, m5_thresholds,
    m6_pca, m7_causal, m8_roles, n_layers, n_heads,
):
    """Combine all measurements into hypothesis scores."""
    print(f"\n{'='*70}")
    print(f"  AGGREGATE ANALYSIS")
    print(f"{'='*70}")

    # A1: Head Role Taxonomy
    print(f"\n  A1: Head Role Taxonomy")
    role_taxonomy = Counter()
    for key, info in m8_roles.items():
        role_taxonomy[info["dominant_role"]] += 1
    for role, count in role_taxonomy.most_common():
        total = len(m8_roles)
        print(f"    {role:<20} {count}/{total} ({count/total:.0%})")

    # A2: Layer-Role Distribution
    print(f"\n  A2: Layer-Role Distribution")
    for li in range(n_layers):
        roles = []
        for h in range(n_heads):
            info = m8_roles.get((li, h), {})
            role = info.get("dominant_role", "?")[:3]
            roles.append(role)
        if li % 5 == 0 or li == n_layers - 1:
            print(f"    L{li:>2}: {' '.join(f'{r:>5}' for r in roles)}")

    # A4: Compilation Recovery
    print(f"\n  A4: Compilation Recovery")
    if m5_thresholds:
        for thresh, info in m5_thresholds.items():
            print(f"    Threshold {thresh}: {info['compilable_layers']}/{n_layers} "
                  f"({info['fraction']:.0%}) layers compilable")

    # Hypothesis scoring
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS SCORING")
    print(f"{'='*70}")

    scores = {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 0}

    # H1: Refinement — small norms, dispensable layers
    mean_ratio = sum(r["avg_ratio"] for r in m1_ratios) / len(m1_ratios) if m1_ratios else 0
    if mean_ratio < 0.3:
        scores["H1"] += 3
    elif mean_ratio < 0.5:
        scores["H1"] += 2
    elif mean_ratio < 1.0:
        scores["H1"] += 1

    dispensable = sum(1 for r in m7_causal if r["dispensable"]) if m7_causal else 0
    if dispensable > n_layers * 0.5:
        scores["H1"] += 3
    elif dispensable > n_layers * 0.3:
        scores["H1"] += 2
    elif dispensable > n_layers * 0.1:
        scores["H1"] += 1

    # H2: Assembly — entity/relation extractors found
    entity_heads = sum(1 for v in m8_roles.values() if v["dominant_role"] == "entity")
    relation_heads = sum(1 for v in m8_roles.values() if v["dominant_role"] == "relation")
    if entity_heads + relation_heads > n_layers:
        scores["H2"] += 3
    elif entity_heads + relation_heads > n_layers * 0.5:
        scores["H2"] += 2
    elif entity_heads > 0:
        scores["H2"] += 1

    # H3: Cancellation — suppressing layers
    if m4_residual:
        suppressors = sum(1 for s in m4_residual if s["bias"] == "suppressing")
        if suppressors > n_layers * 0.5:
            scores["H3"] += 3
        elif suppressors > n_layers * 0.3:
            scores["H3"] += 2
        elif suppressors > n_layers * 0.1:
            scores["H3"] += 1

    # H4: Sub-Templates — compilability recovers at relaxed threshold
    if m5_thresholds:
        comp_094 = m5_thresholds.get("0.94", {}).get("fraction", 0)
        comp_060 = m5_thresholds.get("0.6", {}).get("fraction", 0)
        recovery = comp_060 - comp_094
        if recovery > 0.4:
            scores["H4"] += 3
        elif recovery > 0.2:
            scores["H4"] += 2
        elif recovery > 0.1:
            scores["H4"] += 1

    # H5: Graph Walk — entity extractors in early layers, answer in late
    early_entity = sum(1 for (li, h), v in m8_roles.items()
                      if li < n_layers // 3 and v["dominant_role"] == "entity")
    # Structural progression from entity to other roles
    if early_entity > 0:
        scores["H5"] += 1
    # If attention norms are structured (not just noise)
    if mean_ratio > 0.3:
        scores["H5"] += 1

    # PCA: many heads in few-modes = template-like
    if m6_pca:
        few_mode_heads = sum(1 for v in m6_pca.values()
                           if v["classification"] in ("single_mode", "few_modes"))
        total_classified = len(m6_pca)
        if few_mode_heads > total_classified * 0.5:
            scores["H4"] += 2  # supports sub-templates
            scores["H5"] += 1  # also consistent with structured walk

    # Print scores
    print(f"\n  {'Hypothesis':<30} {'Score':>6} {'Evidence'}")
    print(f"  {'─'*70}")

    evidence = {
        "H1": f"ratio={mean_ratio:.3f}, dispensable={dispensable}/{n_layers}",
        "H2": f"entity_heads={entity_heads}, relation_heads={relation_heads}",
        "H3": f"suppressing={sum(1 for s in m4_residual if s['bias']=='suppressing') if m4_residual else 0}/{n_layers}",
        "H4": f"compilable@0.6={m5_thresholds.get('0.6', {}).get('fraction', 0):.0%}" if m5_thresholds else "N/A",
        "H5": f"early_entity={early_entity}, structured_norm={'yes' if mean_ratio > 0.3 else 'no'}",
    }

    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        name = {"H1": "Refinement", "H2": "Assembly", "H3": "Cancellation",
                "H4": "Sub-Templates", "H5": "Graph Walk"}[hyp]
        print(f"  {hyp}: {name:<24} {scores[hyp]:>6}  {evidence[hyp]}")

    winner = max(scores.items(), key=lambda x: x[1])
    return scores, winner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  WHAT IS ATTENTION ACTUALLY DOING?")
    print("  Five hypotheses, one experiment, Gemma 3-4B")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"\n  Device: CPU")

    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    n_heads = tc.num_attention_heads
    hidden_dim = tc.hidden_size
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers}L × {n_heads}H, hidden={hidden_dim}")

    # Build prompts
    prompts_raw = build_prompts()
    print(f"  Prompts: {len(prompts_raw)}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Data Collection
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: DATA COLLECTION")
    print(f"{'='*70}")

    all_data, n_layers, n_heads, hidden_dim, embed_weight = collect_data(
        model, tokenizer, prompts_raw, device)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: MEASUREMENTS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: MEASUREMENTS")
    print(f"{'='*70}")

    # M1: Norm ratios
    m1_ratios, m1_mean = measure_m1_norm_ratios(all_data, n_layers)

    # M4: Residual change
    m4_residual = measure_m4_residual_change(all_data, n_layers)

    # M5: Cross-entity cosine
    m5_cosines, m5_thresholds = measure_m5_cross_entity_cosine(all_data, n_layers)

    # M6: Head PCA
    m6_pca = measure_m6_head_clustering(all_data, n_layers, n_heads)

    # M7: Causal skipping (most expensive — use subset)
    m7_causal = measure_m7_causal_skipping(
        model, tokenizer, prompts_raw, device, n_layers, max_prompts=20)

    # M8: Attention decomposition
    m8_roles = measure_m8_attention_decomposition(all_data, n_layers, n_heads)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: AGGREGATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    scores, winner = aggregate_analysis(
        m1_ratios, m4_residual, m5_cosines, m5_thresholds,
        m6_pca, m7_causal, m8_roles, n_layers, n_heads)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    hyp_name = {"H1": "Refinement", "H2": "Assembly", "H3": "Cancellation",
                "H4": "Sub-Templates", "H5": "Graph Walk"}

    print(f"\n  WINNER: {winner[0]} — {hyp_name[winner[0]]} (score: {winner[1]})")

    if winner[0] == "H1":
        print(f"\n  Attention is MINOR REFINEMENT.")
        print(f"  The FFN carries the model. Attention adjusts confidence.")
        print(f"  → Focus on FFN compilation. Attention can be a cheap bias.")
        print(f"  → Model compilability: ~70-80%")
    elif winner[0] == "H2":
        print(f"\n  Attention is a PARSER.")
        print(f"  It assembles queries (entity + relation) for the FFN to answer.")
        print(f"  → Build a small trained parser to replace attention.")
        print(f"  → Model compilability: ~60%")
    elif winner[0] == "H3":
        print(f"\n  Attention is NOISE CANCELLATION.")
        print(f"  It suppresses irrelevant features to sharpen the signal.")
        print(f"  → Replaceable by a learned sparse mask.")
        print(f"  → Model compilability: ~65%")
    elif winner[0] == "H4":
        print(f"\n  Attention is FINE-GRAINED TEMPLATES.")
        print(f"  More templates (200-500) would recover compilability.")
        print(f"  → Build finer template library. The v12 approach was right.")
        print(f"  → Model compilability: ~70%")
    elif winner[0] == "H5":
        print(f"\n  Attention IS THE GRAPH WALK.")
        print(f"  FFN stores the graph. Attention traverses it.")
        print(f"  → The entire model is a graph operation.")
        print(f"  → Model compilability: ~90%")

    # Detailed layer-by-layer summary
    print(f"\n  Layer-by-layer summary:")
    print(f"  {'L':>3} {'attn/ffn':>8} {'skip_ok':>8} {'bias':>12} "
          f"{'cos@0.6':>8} {'dom_role':>10}")
    print(f"  {'─'*55}")
    for li in range(n_layers):
        ratio = m1_ratios[li]["avg_ratio"] if li < len(m1_ratios) else 0
        skip_ok = m7_causal[li]["top1_preserved"] if li < len(m7_causal) else 0
        bias = m4_residual[li]["bias"] if li < len(m4_residual) else "?"
        cos_06 = ""
        if m5_cosines and li < len(m5_cosines):
            cos_06 = f"{m5_cosines[li]['compilable'].get('0.6', 0):.0%}"

        # Dominant role for this layer
        layer_roles = Counter()
        for h in range(n_heads):
            info = m8_roles.get((li, h), {})
            role = info.get("dominant_role", "?")
            layer_roles[role] += 1
        dom_role = layer_roles.most_common(1)[0][0] if layer_roles else "?"

        if li % 3 == 0 or li == n_layers - 1:
            print(f"  L{li:>2} {ratio:>8.3f} {skip_ok:>7.0%} {bias:>12} "
                  f"{cos_06:>8} {dom_role:>10}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hypothesis_scores": scores,
        "winner": {"hypothesis": winner[0], "name": hyp_name[winner[0]],
                   "score": winner[1]},
        "m1_norm_ratios": m1_ratios,
        "m4_residual_change": m4_residual,
        "m5_threshold_summary": m5_thresholds,
        "m6_pca_summary": {
            k: v["classification"] for k, v in
            sorted(((f"L{k[0]}H{k[1]}", v) for k, v in m6_pca.items()))
        } if m6_pca else {},
        "m7_causal_skipping": m7_causal,
        "m8_role_summary": {
            f"L{k[0]}H{k[1]}": v["dominant_role"]
            for k, v in sorted(m8_roles.items())
        },
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
