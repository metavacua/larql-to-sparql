#!/usr/bin/env python3
"""
BOS Is The Context Register

Track how BOS evolves across layers, whether it routes FFN activation,
and whether trigram types become separable — connecting GPT-OSS, attention
anatomy, and FFN replacement into one framework.

Phase 1: Track BOS across layers (100 prompts × 34 layers)
Phase 2: BOS similarity (same-template vs same-entity vs cross-task)
Phase 3: BOS → FFN routing correlation
Phase 4: Essential layer analysis (what makes L0-5, L24 special)
Phase 5: Trigram type separability in BOS space
Phase 6: Write/read pattern (attention writes BOS, FFN reads BOS)
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
SEED = 42
MAX_SEQ = 64  # short prompts, BOS tracking doesn't need long sequences
OUTPUT_DIR = "results_bos_register"

# ---------------------------------------------------------------------------
# Prompts — structured for hypothesis testing
# ---------------------------------------------------------------------------

PROMPTS = {
    # Same template, different entities (P3: BOS should be similar)
    "capital": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Australia is",
        "The capital of Germany is",
        "The capital of India is",
        "The capital of Canada is",
    ],
    "language": [
        "The official language of France is",
        "The official language of Japan is",
        "The official language of Spain is",
        "The official language of Brazil is",
    ],
    "currency": [
        "The currency of Japan is the",
        "The currency of India is the",
        "The currency of Russia is the",
        "The currency of Mexico is the",
    ],
    # Same entity, different templates (P3: moderate similarity)
    "france_relations": [
        "The capital of France is",
        "The president of France is",
        "The currency of France is the",
        "The population of France is",
        "France is located in",
    ],
    # Creative (different task type)
    "creative": [
        "Once upon a time in a land far away,",
        "The detective opened the door and saw",
        "She picked up the old violin and began to",
        "In a world where time runs backwards,",
        "The last message on earth read:",
    ],
    # Reasoning
    "reasoning": [
        "If all birds can fly and penguins are birds, then",
        "Given that x + y = 10 and x = 3, then y equals",
        "The probability of two independent events both occurring is",
        "If the sequence is 2, 4, 8, 16, the next number is",
    ],
    # Code
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.",
        "class Node:\n    def __init__(self, value):\n        self.",
        "for i in range(len(arr)):\n    for j in range(i + 1, len(arr)):\n        if",
    ],
    # Conversational
    "conversational": [
        "I've been thinking about changing my career because",
        "What do you think about the idea of",
        "The problem with that approach is",
        "That reminds me of something interesting about",
    ],
    # Instructions
    "instructional": [
        "To make scrambled eggs, first",
        "The most effective way to learn a language is",
        "When debugging a program, the first step should be",
        "To prepare for a marathon, you should start by",
    ],
}


# ---------------------------------------------------------------------------
# Phase 1: Track BOS across layers
# ---------------------------------------------------------------------------

def track_bos(model, tokenizer, device, n_layers):
    """Extract BOS state at every layer for every prompt."""
    print(f"\n  Phase 1: Tracking BOS across {n_layers} layers...")
    model.eval()

    # Hooks: capture residual before attention and before FFN at BOS position
    bos_pre_attn = [None] * n_layers
    bos_post_attn = [None] * n_layers
    bos_post_ffn = [None] * n_layers
    gate_activations = [None] * n_layers  # FFN gate activation pattern

    hooks = []

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]

        # Pre-attention: capture input to self_attn (= residual before attention)
        def make_pre_attn(idx):
            def hook(module, args, kwargs):
                hs = kwargs.get('hidden_states', args[0] if args else None)
                if hs is not None:
                    bos_pre_attn[idx] = hs[0, 0].detach().float().cpu()  # batch=0, pos=0
            return hook

        # Post-attention: capture output of self_attn
        def make_post_attn(idx):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                # After attention residual add, BOS state is in the hidden_states
                # We'll compute it from pre + attn_output
                bos_post_attn[idx] = out[0, 0].detach().float().cpu()
            return hook

        # Pre-FFN: capture gate activations
        def make_gate_hook(idx):
            def hook(module, args):
                if args:
                    inp = args[0] if isinstance(args, tuple) else args
                    gate_activations[idx] = inp[0, 0].detach().float().cpu()  # BOS position
            return hook

        # Post-FFN: capture MLP output at BOS
        def make_post_ffn(idx):
            def hook(module, args, output):
                bos_post_ffn[idx] = output[0, 0].detach().float().cpu()
            return hook

        hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_attn(li), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(make_post_attn(li)))
        hooks.append(layer.mlp.register_forward_pre_hook(make_gate_hook(li)))
        hooks.append(layer.mlp.register_forward_hook(make_post_ffn(li)))

    results = {}
    t0 = time.time()
    total_prompts = sum(len(v) for v in PROMPTS.values())
    done = 0

    with torch.no_grad():
        for category, prompt_list in PROMPTS.items():
            for prompt in prompt_list:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                                 truncation=True).to(device)

                _ = model(**inputs)

                trajectory = []
                for li in range(n_layers):
                    t = {}
                    if bos_pre_attn[li] is not None:
                        t["pre_attn"] = bos_pre_attn[li]
                    if bos_post_attn[li] is not None:
                        t["post_attn"] = bos_post_attn[li]
                    if bos_post_ffn[li] is not None:
                        t["post_ffn"] = bos_post_ffn[li]
                    if gate_activations[li] is not None:
                        t["gate_input"] = gate_activations[li]

                    # Derived: attention contribution to BOS
                    if "pre_attn" in t and "post_attn" in t:
                        t["attn_delta"] = t["post_attn"] - t["pre_attn"]
                        t["attn_norm"] = t["attn_delta"].norm().item()

                    # Derived: FFN contribution to BOS
                    if "post_ffn" in t:
                        t["ffn_norm"] = t["post_ffn"].norm().item()

                    trajectory.append(t)

                results[(category, prompt)] = trajectory
                done += 1
                if done % 10 == 0:
                    print(f"    {done}/{total_prompts} ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()

    print(f"  Collected {len(results)} trajectories in {time.time()-t0:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Phase 2: BOS similarity analysis
# ---------------------------------------------------------------------------

def bos_similarity(results, n_layers):
    """Measure BOS cosine similarity at each layer for different prompt groupings."""
    print(f"\n  Phase 2: BOS similarity analysis...")

    def get_bos(cat, prompt, layer, field="post_attn"):
        t = results.get((cat, prompt), [])
        if layer < len(t) and field in t[layer]:
            return t[layer][field]
        return None

    # Collect BOS states by group
    groups = {
        "same_template": [],  # capital-of prompts
        "same_entity": [],    # france-relation prompts
        "cross_task": [],     # factual vs creative vs code
    }

    layer_analysis = []

    for li in range(n_layers):
        # Same template (capital of X): should be HIGH
        cap_bos = []
        for prompt in PROMPTS["capital"]:
            b = get_bos("capital", prompt, li)
            if b is not None:
                cap_bos.append(b)

        same_template_cos = []
        if len(cap_bos) >= 2:
            stack = torch.stack(cap_bos)
            norm = F.normalize(stack, dim=1)
            cos_mat = norm @ norm.T
            mask = torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()
            same_template_cos = cos_mat[mask].tolist()

        # Same entity (France X): should be MODERATE
        france_bos = []
        for prompt in PROMPTS["france_relations"]:
            b = get_bos("france_relations", prompt, li)
            if b is not None:
                france_bos.append(b)

        same_entity_cos = []
        if len(france_bos) >= 2:
            stack = torch.stack(france_bos)
            norm = F.normalize(stack, dim=1)
            cos_mat = norm @ norm.T
            mask = torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()
            same_entity_cos = cos_mat[mask].tolist()

        # Cross-task: factual vs creative vs code vs reasoning
        task_means = {}
        for cat in ["capital", "creative", "code", "reasoning"]:
            bos_list = []
            for prompt in PROMPTS.get(cat, []):
                b = get_bos(cat, prompt, li)
                if b is not None:
                    bos_list.append(b)
            if bos_list:
                task_means[cat] = torch.stack(bos_list).mean(dim=0)

        cross_task_cos = []
        cats = sorted(task_means.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                cos = F.cosine_similarity(
                    task_means[cats[i]].unsqueeze(0),
                    task_means[cats[j]].unsqueeze(0)
                ).item()
                cross_task_cos.append(cos)

        st = sum(same_template_cos) / len(same_template_cos) if same_template_cos else 0
        se = sum(same_entity_cos) / len(same_entity_cos) if same_entity_cos else 0
        ct = sum(cross_task_cos) / len(cross_task_cos) if cross_task_cos else 0

        layer_analysis.append({
            "layer": li,
            "same_template": round(st, 4),
            "same_entity": round(se, 4),
            "cross_task": round(ct, 4),
        })

    # Print
    print(f"\n  {'Layer':>5} {'Same-Tmpl':>10} {'Same-Ent':>10} {'Cross-Task':>10} {'Gradient':>10}")
    print(f"  {'─'*48}")
    for la in layer_analysis:
        gradient = la["same_template"] - la["cross_task"]
        if la["layer"] % 3 == 0 or la["layer"] == n_layers - 1:
            print(f"  L{la['layer']:>3} {la['same_template']:>10.4f} "
                  f"{la['same_entity']:>10.4f} {la['cross_task']:>10.4f} "
                  f"{gradient:>10.4f}")

    return layer_analysis


# ---------------------------------------------------------------------------
# Phase 3: BOS → FFN routing correlation
# ---------------------------------------------------------------------------

def bos_ffn_correlation(results, n_layers):
    """Does BOS content predict FFN gate activation?"""
    print(f"\n  Phase 3: BOS → FFN routing correlation...")

    layer_correlations = []

    for li in range(n_layers):
        bos_states = []
        gate_inputs = []

        for key, traj in results.items():
            if li < len(traj):
                bos = traj[li].get("post_attn")
                gate = traj[li].get("gate_input")
                if bos is not None and gate is not None:
                    bos_states.append(bos)
                    gate_inputs.append(gate)

        if len(bos_states) < 5:
            layer_correlations.append({"layer": li, "correlation": 0})
            continue

        # Pairwise cosine: do similar BOS → similar FFN input?
        B = torch.stack(bos_states)
        G = torch.stack(gate_inputs)

        B_norm = F.normalize(B, dim=1)
        G_norm = F.normalize(G, dim=1)

        bos_cos = (B_norm @ B_norm.T).flatten()
        gate_cos = (G_norm @ G_norm.T).flatten()

        # Pearson correlation
        bos_m = bos_cos - bos_cos.mean()
        gate_m = gate_cos - gate_cos.mean()
        num = (bos_m * gate_m).sum()
        den = (bos_m.norm() * gate_m.norm()) + 1e-10
        corr = (num / den).item()

        layer_correlations.append({
            "layer": li,
            "correlation": round(corr, 4),
        })

    print(f"\n  {'Layer':>5} {'BOS→FFN corr':>14}")
    print(f"  {'─'*22}")
    for lc in layer_correlations:
        marker = " ← HIGH" if lc["correlation"] > 0.7 else ""
        if lc["layer"] % 3 == 0 or lc["layer"] == n_layers - 1:
            print(f"  L{lc['layer']:>3} {lc['correlation']:>14.4f}{marker}")

    return layer_correlations


# ---------------------------------------------------------------------------
# Phase 4: Essential layer analysis
# ---------------------------------------------------------------------------

def essential_layer_analysis(results, n_layers):
    """What makes the 7 essential layers (L0-5, L24) special?"""
    print(f"\n  Phase 4: Essential layer analysis...")

    essential = {0, 1, 2, 3, 4, 5, 24}

    for li in sorted(essential):
        # Attention write norm at BOS
        attn_norms = []
        for key, traj in results.items():
            if li < len(traj) and "attn_norm" in traj[li]:
                attn_norms.append(traj[li]["attn_norm"])

        avg_norm = sum(attn_norms) / len(attn_norms) if attn_norms else 0

        # Category-dependence: do different task types write different things?
        cat_means = {}
        for (cat, prompt), traj in results.items():
            if li < len(traj) and "attn_delta" in traj[li]:
                base_cat = cat.split("_")[0] if "_" in cat else cat
                if base_cat not in cat_means:
                    cat_means[base_cat] = []
                cat_means[base_cat].append(traj[li]["attn_delta"])

        # Between-category cosine
        cats = sorted(cat_means.keys())
        cross_cos = []
        within_cos = []
        for ci, ca in enumerate(cats):
            mean_a = torch.stack(cat_means[ca]).mean(0)
            for cj, cb in enumerate(cats):
                if cj <= ci:
                    continue
                mean_b = torch.stack(cat_means[cb]).mean(0)
                cos = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()
                cross_cos.append(cos)

            # Within-category variance
            if len(cat_means[ca]) >= 2:
                stack = torch.stack(cat_means[ca])
                norm = F.normalize(stack, dim=1)
                wcos = (norm @ norm.T)
                mask = torch.triu(torch.ones_like(wcos), diagonal=1).bool()
                within_cos.extend(wcos[mask].tolist())

        avg_cross = sum(cross_cos) / len(cross_cos) if cross_cos else 0
        avg_within = sum(within_cos) / len(within_cos) if within_cos else 0

        print(f"  L{li}: attn_norm={avg_norm:.3f}, "
              f"within_cat_cos={avg_within:.3f}, cross_cat_cos={avg_cross:.3f}, "
              f"separability={avg_within - avg_cross:.3f}")


# ---------------------------------------------------------------------------
# Phase 5: Trigram type separability
# ---------------------------------------------------------------------------

def trigram_separability(results, n_layers):
    """When do structural pattern types become separable in BOS space?"""
    print(f"\n  Phase 5: Trigram type separability...")

    # Group by structural pattern
    task_groups = {
        "factual": ["capital", "language", "currency"],
        "creative": ["creative"],
        "reasoning": ["reasoning"],
        "code": ["code"],
        "instruction": ["instructional"],
        "conversation": ["conversational"],
    }

    layer_sep = []

    for li in range(n_layers):
        group_centroids = {}
        for group_name, categories in task_groups.items():
            bos_list = []
            for cat in categories:
                for prompt in PROMPTS.get(cat, []):
                    traj = results.get((cat, prompt), [])
                    if li < len(traj) and "post_attn" in traj[li]:
                        bos_list.append(traj[li]["post_attn"])
            if bos_list:
                group_centroids[group_name] = torch.stack(bos_list).mean(0)

        if len(group_centroids) < 2:
            layer_sep.append({"layer": li, "separability": 0})
            continue

        # Within-group vs between-group cosine
        groups = sorted(group_centroids.keys())
        between = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                cos = F.cosine_similarity(
                    group_centroids[groups[i]].unsqueeze(0),
                    group_centroids[groups[j]].unsqueeze(0),
                ).item()
                between.append(cos)

        avg_between = sum(between) / len(between) if between else 0
        separability = 1.0 - avg_between  # higher = more separated

        layer_sep.append({
            "layer": li,
            "between_cos": round(avg_between, 4),
            "separability": round(separability, 4),
        })

    print(f"\n  {'Layer':>5} {'Between-cos':>12} {'Separability':>12}")
    print(f"  {'─'*32}")
    for ls in layer_sep:
        marker = " ← SEPARATED" if ls["separability"] > 0.3 else ""
        if ls["layer"] % 3 == 0 or ls["layer"] == n_layers - 1:
            print(f"  L{ls['layer']:>3} {ls.get('between_cos', 0):>12.4f} "
                  f"{ls['separability']:>12.4f}{marker}")

    return layer_sep


# ---------------------------------------------------------------------------
# Phase 6: Write/read pattern
# ---------------------------------------------------------------------------

def write_read_pattern(results, n_layers):
    """Attention writes to BOS, FFN reads from BOS."""
    print(f"\n  Phase 6: Write/read pattern...")

    layer_wr = []
    for li in range(n_layers):
        write_norms = []
        read_correlations = []

        for key, traj in results.items():
            if li < len(traj):
                # WRITE: attention contribution norm at BOS
                if "attn_norm" in traj[li]:
                    write_norms.append(traj[li]["attn_norm"])

                # READ: does FFN output correlate with BOS state?
                bos = traj[li].get("post_attn")
                ffn = traj[li].get("post_ffn")
                if bos is not None and ffn is not None:
                    cos = F.cosine_similarity(bos.unsqueeze(0), ffn.unsqueeze(0)).item()
                    read_correlations.append(cos)

        avg_write = sum(write_norms) / len(write_norms) if write_norms else 0
        avg_read = sum(read_correlations) / len(read_correlations) if read_correlations else 0

        layer_wr.append({
            "layer": li,
            "write_norm": round(avg_write, 3),
            "read_cos": round(avg_read, 4),
        })

    print(f"\n  {'Layer':>5} {'Write(attn)':>12} {'Read(FFN)':>12}")
    print(f"  {'─'*32}")
    for lw in layer_wr:
        if lw["layer"] % 3 == 0 or lw["layer"] == n_layers - 1:
            print(f"  L{lw['layer']:>3} {lw['write_norm']:>12.3f} {lw['read_cos']:>12.4f}")

    return layer_wr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  BOS IS THE CONTEXT REGISTER")
    print("  Tracking the message bus between attention and FFN")
    print("=" * 70)

    # CPU float32 — MPS float16 produces NaN at L6+ due to precision issues
    device = torch.device("cpu")
    print(f"\n  Device: CPU (float32, ~0.5s per forward pass)")

    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers} layers")
    total_prompts = sum(len(v) for v in PROMPTS.values())
    print(f"  Prompts: {total_prompts} across {len(PROMPTS)} categories")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Track BOS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: TRACK BOS ACROSS LAYERS")
    print(f"{'='*70}")

    results = track_bos(model, tokenizer, device, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: BOS similarity
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: BOS SIMILARITY ANALYSIS")
    print(f"{'='*70}")

    similarity = bos_similarity(results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: BOS → FFN correlation
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: BOS → FFN ROUTING CORRELATION")
    print(f"{'='*70}")

    correlations = bos_ffn_correlation(results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Essential layers
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 4: ESSENTIAL LAYER ANALYSIS")
    print(f"{'='*70}")

    essential_layer_analysis(results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5: Trigram separability
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 5: TRIGRAM TYPE SEPARABILITY")
    print(f"{'='*70}")

    trigram_sep = trigram_separability(results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 6: Write/read
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 6: WRITE/READ PATTERN")
    print(f"{'='*70}")

    write_read = write_read_pattern(results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT: IS BOS THE CONTEXT REGISTER?")
    print(f"{'='*70}")

    # Check predictions
    evidence_for = 0
    evidence_against = 0

    # P1: BOS changes monotonically
    # Check if BOS content at L0 vs L33 is very different
    l0_l33_cos = []
    for key, traj in results.items():
        if len(traj) > 33 and "post_attn" in traj[0] and "post_attn" in traj[33]:
            cos = F.cosine_similarity(
                traj[0]["post_attn"].unsqueeze(0),
                traj[33]["post_attn"].unsqueeze(0)
            ).item()
            l0_l33_cos.append(cos)
    avg_l0_l33 = sum(l0_l33_cos) / len(l0_l33_cos) if l0_l33_cos else 0
    print(f"\n  P1: BOS evolves across layers")
    print(f"    L0↔L33 cosine: {avg_l0_l33:.4f}")
    if avg_l0_l33 < 0.5:
        print(f"    ✓ BOS changes substantially (cos < 0.5)")
        evidence_for += 1
    else:
        print(f"    ~ BOS stays somewhat similar")

    # P2: Same-template > same-entity > cross-task
    mid_layer = n_layers // 2
    if mid_layer < len(similarity):
        st = similarity[mid_layer]["same_template"]
        se = similarity[mid_layer]["same_entity"]
        ct = similarity[mid_layer]["cross_task"]
        print(f"\n  P2: Similarity gradient at L{mid_layer}")
        print(f"    Same-template: {st:.4f}")
        print(f"    Same-entity: {se:.4f}")
        print(f"    Cross-task: {ct:.4f}")
        if st > se > ct:
            print(f"    ✓ Correct ordering: template > entity > task")
            evidence_for += 1
        elif st > ct:
            print(f"    ~ Partial: template > task, but entity not in between")
            evidence_for += 0.5
        else:
            print(f"    ✗ Ordering not as predicted")
            evidence_against += 1

    # P3: BOS→FFN correlation high at knowledge layers
    know_layers = range(6, 20)
    know_corrs = [c["correlation"] for c in correlations
                  if c["layer"] in know_layers and c["correlation"] != 0]
    avg_know = sum(know_corrs) / len(know_corrs) if know_corrs else 0
    other_corrs = [c["correlation"] for c in correlations
                   if c["layer"] not in know_layers and c["correlation"] != 0]
    avg_other = sum(other_corrs) / len(other_corrs) if other_corrs else 0
    print(f"\n  P3: BOS→FFN correlation")
    print(f"    Knowledge layers (L6-19): {avg_know:.4f}")
    print(f"    Other layers: {avg_other:.4f}")
    if avg_know > 0.5:
        print(f"    ✓ High BOS→FFN correlation at knowledge layers")
        evidence_for += 1
    elif avg_know > avg_other:
        print(f"    ~ Higher at knowledge layers but not strong")
        evidence_for += 0.5
    else:
        print(f"    ✗ BOS doesn't predict FFN activation")
        evidence_against += 1

    # P5: Trigram types become separable
    sep_transition = None
    for ts in trigram_sep:
        if ts["separability"] > 0.2 and sep_transition is None:
            sep_transition = ts["layer"]
    print(f"\n  P5: Trigram separability")
    if sep_transition is not None:
        print(f"    Types become separable at L{sep_transition}")
        if 3 <= sep_transition <= 8:
            print(f"    ✓ Matches L5-6 transition (position→context specialist)")
            evidence_for += 1
        else:
            print(f"    ~ Separable but not at expected transition")
            evidence_for += 0.5
    else:
        print(f"    ✗ Types never clearly separate in BOS space")
        evidence_against += 1

    # Final score
    print(f"\n  Evidence for BOS-as-register: {evidence_for}")
    print(f"  Evidence against: {evidence_against}")

    if evidence_for >= 3:
        print(f"\n  ✓ BOS IS THE CONTEXT REGISTER")
        print(f"    Attention writes context to BOS. FFN reads BOS to route.")
        print(f"    The model is a query compiler (attention) + database (FFN).")
    elif evidence_for >= 2:
        print(f"\n  ~ BOS IS PARTIALLY A CONTEXT REGISTER")
        print(f"    Some routing goes through BOS, but other mechanisms also present.")
    else:
        print(f"\n  ✗ BOS IS NOT THE PRIMARY ROUTING MECHANISM")
        print(f"    Attention is doing something else we haven't characterized.")

    # Save results
    save_data = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "total_prompts": total_prompts,
        "similarity": similarity,
        "correlations": correlations,
        "trigram_separability": trigram_sep,
        "write_read": write_read,
        "bos_l0_l33_cosine": avg_l0_l33,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
