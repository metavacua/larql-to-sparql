#!/usr/bin/env python3
"""
The Last Token Is The Query Buffer

Trace the database query lifecycle in the residual stream:
  Attention builds the query → FFN executes it → answer replaces query

Phase 1: Token-space readout at every layer
Phase 2: Attention vs FFN attribution (who contributed the answer?)
Phase 3: Entity signal tracking (hourglass in token space)
Phase 4: Task type entropy profiles
Phase 5: Complete query trace for showcase examples
"""

import os
import sys
import json
import time
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
MAX_SEQ = 64
OUTPUT_DIR = "results_query_lifecycle"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FACTUAL = [
    ("The capital of France is", "France", "Paris"),
    ("The capital of Japan is", "Japan", "Tokyo"),
    ("The capital of Germany is", "Germany", "Berlin"),
    ("The capital of Italy is", "Italy", "Rome"),
    ("The capital of Egypt is", "Egypt", "Cairo"),
    ("The official language of France is", "France", "French"),
    ("The official language of Japan is", "Japan", "Japanese"),
    ("The chemical symbol for gold is", "gold", "Au"),
    ("The Earth orbits the", "Earth", "Sun"),
    ("The currency of Japan is the", "Japan", "yen"),
]

CREATIVE = [
    "Once upon a time, there was a",
    "The detective opened the door and saw",
    "She picked up the old violin and began to",
]

CODE = [
    ("def fibonacci(n):\n    if n <= 1:\n        return", "n"),
    ("for i in range(", "len"),
    ("import pandas as", "pd"),
]

REASONING = [
    ("If all cats are mammals and Whiskers is a cat, then Whiskers is a", "mammal"),
]


# ---------------------------------------------------------------------------
# Core: hook-based layer-by-layer readout
# ---------------------------------------------------------------------------

def run_with_readout(model, tokenizer, input_ids, device, n_layers, hidden_dim):
    """
    Run forward pass, capturing the prediction position's state
    before/after attention and before/after FFN at every layer.
    Returns per-layer readout dicts.
    """
    embed_weight = model.model.language_model.embed_tokens.weight.data.float()
    last_pos = input_ids.shape[1] - 1

    # Storage
    readouts = []
    state = {"residual_pre_attn": None, "residual_post_attn": None}

    hooks = []
    layer_data = [None] * n_layers

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]

        def make_pre_attn(idx):
            def hook(module, args, kwargs):
                hs = kwargs.get('hidden_states')
                if hs is not None:
                    layer_data[idx] = {"pre_attn": hs[0, last_pos].detach().float().cpu()}
            return hook

        def make_post_attn(idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                d = layer_data[idx] or {}
                d["post_attn"] = out[0, last_pos].detach().float().cpu()
                layer_data[idx] = d
            return hook

        def make_pre_ffn(idx):
            def hook(module, args):
                if args:
                    inp = args[0] if isinstance(args, tuple) else args
                    d = layer_data[idx] or {}
                    d["pre_ffn"] = inp[0, last_pos].detach().float().cpu()
                    layer_data[idx] = d
            return hook

        def make_post_ffn(idx):
            def hook(module, args, output):
                d = layer_data[idx] or {}
                d["post_ffn"] = output[0, last_pos].detach().float().cpu()
                layer_data[idx] = d
            return hook

        hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_attn(li), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(make_post_attn(li)))
        hooks.append(layer.mlp.register_forward_pre_hook(make_pre_ffn(li)))
        hooks.append(layer.mlp.register_forward_hook(make_post_ffn(li)))

    with torch.no_grad():
        outputs = model(input_ids)

    for h in hooks:
        h.remove()

    # Process into readouts
    for li in range(n_layers):
        d = layer_data[li] or {}
        r = {"layer": li}

        for key in ["pre_attn", "post_attn", "pre_ffn", "post_ffn"]:
            if key in d:
                vec = d[key]
                # Project to token space
                logits = vec @ embed_weight.T
                top_vals, top_ids = logits.topk(10)
                r[f"{key}_top10"] = [
                    (tokenizer.decode([tid.item()]).strip(), val.item())
                    for val, tid in zip(top_vals, top_ids)
                ]
                r[f"{key}_vec"] = vec

        # Attention delta in token space
        if "pre_attn" in d and "post_attn" in d:
            attn_delta = d["post_attn"] - d["pre_attn"]
            attn_logits = attn_delta @ embed_weight.T
            top_vals, top_ids = attn_logits.topk(5)
            r["attn_adds"] = [
                (tokenizer.decode([tid.item()]).strip(), val.item())
                for val, tid in zip(top_vals, top_ids)
            ]
            r["attn_norm"] = attn_delta.norm().item()

        # FFN delta in token space
        if "pre_ffn" in d and "post_ffn" in d:
            ffn_delta = d["post_ffn"] - d["pre_ffn"]
            ffn_logits = ffn_delta @ embed_weight.T
            top_vals, top_ids = ffn_logits.topk(5)
            r["ffn_adds"] = [
                (tokenizer.decode([tid.item()]).strip(), val.item())
                for val, tid in zip(top_vals, top_ids)
            ]
            r["ffn_norm"] = ffn_delta.norm().item()

        readouts.append(r)

    return readouts


def find_answer_token_ids(tokenizer, answer):
    """Get possible token IDs for the answer."""
    ids = set()
    for prefix in ["", " ", "\n"]:
        encoded = tokenizer.encode(prefix + answer, add_special_tokens=False)
        ids.update(encoded[:2])
    return ids


def answer_rank(readout, embed_weight, answer_ids, field="post_ffn"):
    """What rank is the answer token at this layer?"""
    vec = readout.get(f"{field}_vec")
    if vec is None or not answer_ids:
        return 999999
    logits = vec @ embed_weight.T
    best_rank = 999999
    for aid in answer_ids:
        rank = (logits > logits[aid]).sum().item() + 1
        best_rank = min(best_rank, rank)
    return best_rank


def answer_score(readout, embed_weight, answer_ids, field="post_ffn"):
    """Raw logit score for the answer token."""
    vec = readout.get(f"{field}_vec")
    if vec is None or not answer_ids:
        return -999
    logits = vec @ embed_weight.T
    return max(logits[aid].item() for aid in answer_ids)


# ---------------------------------------------------------------------------
# Phase 1: Token-space readout
# ---------------------------------------------------------------------------

def phase1_readout(model, tokenizer, device, n_layers, embed_weight):
    print(f"\n{'='*70}")
    print(f"  PHASE 1: TOKEN-SPACE READOUT AT EVERY LAYER")
    print(f"{'='*70}")

    for prompt, entity, expected in FACTUAL[:5]:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        readouts = run_with_readout(model, tokenizer, inputs["input_ids"],
                                   device, n_layers, embed_weight.shape[1])
        answer_ids = find_answer_token_ids(tokenizer, expected)

        print(f"\n  Query: '{prompt}' → expecting '{expected}'")
        print(f"  {'L':>3} {'After Attn top-3':>35} {'After FFN top-3':>35} {'Answer':>8}")
        print(f"  {'─'*85}")

        for r in readouts:
            li = r["layer"]
            attn_top = r.get("post_attn_top10", [])[:3]
            ffn_top = r.get("post_ffn_top10", [])[:3]
            rank = answer_rank(r, embed_weight, answer_ids)

            attn_str = ", ".join(f"{t}" for t, s in attn_top)
            ffn_str = ", ".join(f"{t}" for t, s in ffn_top)
            rank_str = f"#{rank}" if rank < 100 else f"#{rank}"
            marker = " ← TOP" if rank == 1 else (" ← top5" if rank <= 5 else "")

            if li % 3 == 0 or li == n_layers - 1 or rank <= 5:
                print(f"  L{li:>2} {attn_str:>35} {ffn_str:>35} {rank_str:>8}{marker}")

    return readouts


# ---------------------------------------------------------------------------
# Phase 2: Attribution
# ---------------------------------------------------------------------------

def phase2_attribution(model, tokenizer, device, n_layers, embed_weight):
    print(f"\n{'='*70}")
    print(f"  PHASE 2: ATTENTION vs FFN ATTRIBUTION")
    print(f"{'='*70}")

    attn_pcts = []
    ffn_pcts = []
    answer_layers = []

    for prompt, entity, expected in FACTUAL:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        readouts = run_with_readout(model, tokenizer, inputs["input_ids"],
                                   device, n_layers, embed_weight.shape[1])
        answer_ids = find_answer_token_ids(tokenizer, expected)

        # Find layer where answer first enters top-10
        first_top10 = None
        for r in readouts:
            rank = answer_rank(r, embed_weight, answer_ids)
            if rank <= 10 and first_top10 is None:
                first_top10 = r["layer"]
                break

        if first_top10 is None:
            continue

        r = readouts[first_top10]
        # Attribution: score increase from attention vs FFN
        pre_score = answer_score(r, embed_weight, answer_ids, "pre_attn")
        post_attn_score = answer_score(r, embed_weight, answer_ids, "post_attn")
        post_ffn_score = answer_score(r, embed_weight, answer_ids, "post_ffn")

        attn_contrib = post_attn_score - pre_score
        ffn_contrib = post_ffn_score - post_attn_score
        total = abs(attn_contrib) + abs(ffn_contrib)

        if total > 0:
            attn_pct = attn_contrib / total * 100
            ffn_pct = ffn_contrib / total * 100
        else:
            attn_pct = ffn_pct = 50

        attn_pcts.append(attn_pct)
        ffn_pcts.append(ffn_pct)
        answer_layers.append(first_top10)

        print(f"  '{prompt[:45]}' → '{expected}'")
        print(f"    Answer enters top-10 at L{first_top10}")
        print(f"    Attention: {attn_contrib:+.2f} ({attn_pct:.0f}%)  "
              f"FFN: {ffn_contrib:+.2f} ({ffn_pct:.0f}%)")

    if attn_pcts:
        avg_attn = sum(attn_pcts) / len(attn_pcts)
        avg_ffn = sum(ffn_pcts) / len(ffn_pcts)
        avg_layer = sum(answer_layers) / len(answer_layers)
        print(f"\n  AVERAGE ATTRIBUTION:")
        print(f"    Attention: {avg_attn:.0f}%")
        print(f"    FFN: {avg_ffn:.0f}%")
        print(f"    Answer appears at: L{avg_layer:.1f} (average)")
        return avg_attn, avg_ffn, avg_layer
    return 0, 0, 0


# ---------------------------------------------------------------------------
# Phase 3: Entity signal tracking
# ---------------------------------------------------------------------------

def phase3_entity_signal(model, tokenizer, device, n_layers, embed_weight):
    print(f"\n{'='*70}")
    print(f"  PHASE 3: ENTITY SIGNAL TRACKING (THE HOURGLASS)")
    print(f"{'='*70}")

    for prompt, entity, expected in FACTUAL[:5]:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        readouts = run_with_readout(model, tokenizer, inputs["input_ids"],
                                   device, n_layers, embed_weight.shape[1])

        entity_ids = find_answer_token_ids(tokenizer, entity)
        answer_ids = find_answer_token_ids(tokenizer, expected)

        print(f"\n  '{prompt}' → {expected}")
        print(f"  {'L':>3} {'Entity(':>8}{entity[:6]:>6}{')'} {'Answer(':>8}{expected[:6]:>6}{')'} {'Phase'}")
        print(f"  {'─'*60}")

        for r in readouts:
            li = r["layer"]
            e_score = answer_score(r, embed_weight, entity_ids, "post_ffn")
            a_score = answer_score(r, embed_weight, answer_ids, "post_ffn")

            # Phase classification
            if a_score > e_score and a_score > 2:
                phase = "ANSWER DOMINATES"
            elif e_score > 2:
                phase = "entity present"
            elif a_score > 0:
                phase = "answer emerging"
            else:
                phase = "building query"

            e_bar = "█" * max(0, min(20, int(e_score)))
            a_bar = "█" * max(0, min(20, int(a_score)))

            if li % 3 == 0 or li == n_layers - 1 or phase in ("entity present", "ANSWER DOMINATES"):
                print(f"  L{li:>2} {e_score:>+8.2f} {e_bar:<20s} "
                      f"{a_score:>+8.2f} {a_bar:<20s} {phase}")


# ---------------------------------------------------------------------------
# Phase 4: Entropy profiles by task type
# ---------------------------------------------------------------------------

def phase4_entropy(model, tokenizer, device, n_layers, embed_weight):
    print(f"\n{'='*70}")
    print(f"  PHASE 4: ENTROPY PROFILES BY TASK TYPE")
    print(f"{'='*70}")

    task_entropies = defaultdict(list)  # task_type → list of entropy curves

    all_prompts = {
        "factual": [(p, e) for p, _, e in FACTUAL[:5]],
        "creative": [(p, None) for p in CREATIVE],
        "code": CODE[:3],
        "reasoning": REASONING,
    }

    for task_type, prompts in all_prompts.items():
        for item in prompts:
            prompt = item[0] if isinstance(item, tuple) else item
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            readouts = run_with_readout(model, tokenizer, inputs["input_ids"],
                                       device, n_layers, embed_weight.shape[1])

            entropies = []
            for r in readouts:
                vec = r.get("post_ffn_vec")
                if vec is not None:
                    logits = vec @ embed_weight.T
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum().item()
                    entropies.append(entropy)
                else:
                    entropies.append(0)

            task_entropies[task_type].append(entropies)

    # Average per task type
    print(f"\n  {'Layer':>5}", end="")
    for task in ["factual", "creative", "code", "reasoning"]:
        print(f" {task:>12}", end="")
    print()
    print(f"  {'─'*55}")

    for li in range(0, n_layers, 3):
        print(f"  L{li:>3}", end="")
        for task in ["factual", "creative", "code", "reasoning"]:
            curves = task_entropies.get(task, [])
            if curves:
                avg = sum(c[li] for c in curves if li < len(c)) / len(curves)
                bar = "█" * min(10, int(avg / 1.0))
                print(f" {avg:>6.1f}{bar:>6s}", end="")
            else:
                print(f" {'?':>12}", end="")
        print()

    return dict(task_entropies)


# ---------------------------------------------------------------------------
# Phase 5: Complete query trace
# ---------------------------------------------------------------------------

def phase5_complete_trace(model, tokenizer, device, n_layers, embed_weight):
    print(f"\n{'='*70}")
    print(f"  PHASE 5: COMPLETE QUERY TRACE")
    print(f"{'='*70}")

    for prompt, entity, expected in FACTUAL[:3]:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        readouts = run_with_readout(model, tokenizer, inputs["input_ids"],
                                   device, n_layers, embed_weight.shape[1])
        answer_ids = find_answer_token_ids(tokenizer, expected)

        print(f"\n  Query: '{prompt}' → '{expected}'")
        print(f"  {'L':>3} {'Attn adds':>30} {'FFN adds':>30} {'Rank':>6}")
        print(f"  {'─'*72}")

        for r in readouts:
            li = r["layer"]
            attn_adds = r.get("attn_adds", [])[:3]
            ffn_adds = r.get("ffn_adds", [])[:3]
            rank = answer_rank(r, embed_weight, answer_ids)

            attn_str = ", ".join(f"{t}({s:+.1f})" for t, s in attn_adds) if attn_adds else "—"
            ffn_str = ", ".join(f"{t}({s:+.1f})" for t, s in ffn_adds) if ffn_adds else "—"

            rank_str = f"#{rank}"
            marker = ""
            if rank == 1:
                marker = " ← #1"
            elif rank <= 5:
                marker = " ← top5"

            # Show every layer — this is the showcase
            print(f"  L{li:>2} {attn_str:>30} {ffn_str:>30} {rank_str:>6}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  THE LAST TOKEN IS THE QUERY BUFFER")
    print("  Tracing the database query lifecycle")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"\n  Device: CPU (float32)")

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
    hidden_dim = tc.hidden_size
    embed_weight = model.model.language_model.embed_tokens.weight.data.float()
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers}L, hidden={hidden_dim}")

    # Run all phases
    t_total = time.time()

    phase1_readout(model, tokenizer, device, n_layers, embed_weight)
    avg_attn, avg_ffn, avg_layer = phase2_attribution(
        model, tokenizer, device, n_layers, embed_weight)
    phase3_entity_signal(model, tokenizer, device, n_layers, embed_weight)
    task_entropies = phase4_entropy(model, tokenizer, device, n_layers, embed_weight)
    phase5_complete_trace(model, tokenizer, device, n_layers, embed_weight)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT: IS THE TRANSFORMER A DATABASE CLIENT?")
    print(f"{'='*70}")

    print(f"\n  Attribution: Attention {avg_attn:.0f}% vs FFN {avg_ffn:.0f}%")
    print(f"  Answer appears at: L{avg_layer:.1f} (average)")

    if avg_ffn > 60:
        print(f"\n  ✓ THE FFN RETRIEVES THE ANSWER")
        print(f"    Attention builds the query ({avg_attn:.0f}%).")
        print(f"    FFN executes it ({avg_ffn:.0f}%).")
        print(f"    The transformer is a database client.")
    elif avg_ffn > 40:
        print(f"\n  ~ MIXED: both contribute")
        print(f"    Attention: {avg_attn:.0f}%, FFN: {avg_ffn:.0f}%")
        print(f"    Both systems contribute to the answer.")
    else:
        print(f"\n  ✗ ATTENTION RETRIEVES THE ANSWER")
        print(f"    Attention: {avg_attn:.0f}%, FFN: {avg_ffn:.0f}%")
        print(f"    The answer comes from attention, not FFN.")

    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    results = {
        "model": MODEL_NAME,
        "attribution": {"attention_pct": avg_attn, "ffn_pct": avg_ffn,
                        "avg_answer_layer": avg_layer},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
