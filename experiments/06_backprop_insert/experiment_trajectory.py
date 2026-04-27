#!/usr/bin/env python3
"""
The Forward Pass Is Predictable

Phase 1: Capture actual trajectory (residual at each layer)
Phase 2: Predict trajectory using derived attention + real FFN
Phase 3: Diagnose drift (attention vs FFN sensitivity)
Phase 4: Analytical shortcut (skip all 34 layers)
Phase 5: Correction frequency (how often do you need real attention?)
"""

import os
import sys
import json
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-3-4b-pt"
MAX_SEQ = 64
OUTPUT_DIR = "results_trajectory"

FACTUAL = [
    ("The capital of France is", "France", "Paris"),
    ("The capital of Japan is", "Japan", "Tokyo"),
    ("The capital of Germany is", "Germany", "Berlin"),
    ("The capital of Italy is", "Italy", "Rome"),
    ("The capital of Egypt is", "Egypt", "Cairo"),
    ("The capital of India is", "India", "Delhi"),
    ("The official language of France is", "France", "French"),
    ("The official language of Japan is", "Japan", "Japanese"),
    ("The chemical symbol for gold is", "gold", "Au"),
    ("The Earth orbits the", "Earth", "Sun"),
]


def get_answer_ids(tokenizer, answer):
    ids = set()
    for prefix in ["", " "]:
        encoded = tokenizer.encode(prefix + answer, add_special_tokens=False)
        ids.update(encoded[:2])
    return ids


def answer_rank(logits, answer_ids):
    if not answer_ids:
        return 999999
    return min((logits > logits[aid]).sum().item() + 1 for aid in answer_ids)


# ---------------------------------------------------------------------------
# Phase 1+2: Actual vs predicted trajectory
# ---------------------------------------------------------------------------

def trajectory_comparison(model, tokenizer, device, n_layers, mean_attn):
    """
    Run actual forward pass and predicted forward pass side by side.
    Predicted: mean attention output + real FFN (on predicted residual).
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 2: TRAJECTORY PREDICTION")
    print(f"{'='*70}")

    embed_weight = model.model.language_model.embed_tokens.weight.data.float()
    hidden = model.config.text_config.hidden_size

    all_cosines = []  # per-prompt list of per-layer cosines

    for prompt, entity, expected in FACTUAL:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        input_ids = inputs["input_ids"]
        last_pos = input_ids.shape[1] - 1
        answer_ids = get_answer_ids(tokenizer, expected)

        # --- ACTUAL forward pass ---
        actual_residuals = []
        actual_hooks = []
        captured_actual = [None] * n_layers

        for li in range(n_layers):
            def make_hook(idx):
                def hook(module, args, output):
                    captured_actual[idx] = output[0, last_pos].detach().float().cpu()
                return hook
            actual_hooks.append(
                model.model.language_model.layers[li].mlp.register_forward_hook(make_hook(li)))

        with torch.no_grad():
            actual_out = model(**inputs)

        for h in actual_hooks:
            h.remove()

        actual_residuals = [r for r in captured_actual if r is not None]

        # --- PREDICTED forward pass ---
        # Use mean attention + real FFN on predicted residual
        predicted_hooks = []
        captured_predicted = [None] * n_layers
        active = [True]

        for li in range(n_layers):
            layer = model.model.language_model.layers[li]
            mean_out = mean_attn[li].to(device)

            def make_attn_hook(idx, mean_vec):
                def hook(module, args, output):
                    if not active[0]:
                        return output
                    out = output[0] if isinstance(output, tuple) else output
                    B, S, D = out.shape
                    replacement = mean_vec.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
                    if isinstance(output, tuple):
                        return (replacement.to(output[0].dtype),) + output[1:]
                    return replacement.to(output.dtype)
                return hook

            def make_ffn_hook(idx):
                def hook(module, args, output):
                    if active[0]:
                        captured_predicted[idx] = output[0, last_pos].detach().float().cpu()
                return hook

            predicted_hooks.append(layer.self_attn.register_forward_hook(
                make_attn_hook(li, mean_out)))
            predicted_hooks.append(layer.mlp.register_forward_hook(
                make_ffn_hook(li)))

        active[0] = True
        with torch.no_grad():
            pred_out = model(**inputs)
        active[0] = False

        for h in predicted_hooks:
            h.remove()

        predicted_residuals = [r for r in captured_predicted if r is not None]

        # --- Compare ---
        cosines = []
        print(f"\n  '{prompt}' → {expected}")
        print(f"  {'L':>3} {'Cosine':>8} {'Actual rank':>12} {'Pred rank':>12}")
        print(f"  {'─'*40}")

        for li in range(min(len(actual_residuals), len(predicted_residuals))):
            cos = F.cosine_similarity(
                actual_residuals[li].unsqueeze(0),
                predicted_residuals[li].unsqueeze(0)
            ).item()
            cosines.append(cos)

            a_logits = actual_residuals[li] @ embed_weight.T
            p_logits = predicted_residuals[li] @ embed_weight.T
            a_rank = answer_rank(a_logits, answer_ids)
            p_rank = answer_rank(p_logits, answer_ids)

            marker = ""
            if cos > 0.95:
                marker = " ✓"
            elif cos > 0.80:
                marker = " ~"
            else:
                marker = " ✗"

            if li % 3 == 0 or li == n_layers - 1 or li < 3:
                print(f"  L{li:>2} {cos:>8.4f}{marker} {a_rank:>12} {p_rank:>12}")

        all_cosines.append(cosines)

    return all_cosines


# ---------------------------------------------------------------------------
# Phase 3: Drift diagnosis
# ---------------------------------------------------------------------------

def diagnose_drift(model, tokenizer, device, n_layers, mean_attn):
    """Is drift from attention prediction or FFN sensitivity?"""
    print(f"\n{'='*70}")
    print(f"  PHASE 3: DRIFT DIAGNOSIS")
    print(f"{'='*70}")

    hidden = model.config.text_config.hidden_size

    for prompt, entity, expected in FACTUAL[:3]:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        last_pos = inputs["input_ids"].shape[1] - 1

        # Capture actual residuals at each layer (before and after FFN)
        actual_pre_ffn = [None] * n_layers
        actual_post_ffn = [None] * n_layers
        actual_attn_out = [None] * n_layers

        hooks = []
        for li in range(n_layers):
            layer = model.model.language_model.layers[li]

            def make_attn_hook(idx):
                def hook(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    actual_attn_out[idx] = out[0, last_pos].detach().float().cpu()
                return hook

            def make_pre_ffn(idx):
                def hook(module, args):
                    if args:
                        inp = args[0] if isinstance(args, tuple) else args
                        actual_pre_ffn[idx] = inp[0, last_pos].detach().float().cpu()
                return hook

            def make_post_ffn(idx):
                def hook(module, args, output):
                    actual_post_ffn[idx] = output[0, last_pos].detach().float().cpu()
                return hook

            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
            hooks.append(layer.mlp.register_forward_pre_hook(make_pre_ffn(li)))
            hooks.append(layer.mlp.register_forward_hook(make_post_ffn(li)))

        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        # Compare: mean attn output vs actual attn output
        print(f"\n  '{prompt}' → {expected}")
        print(f"  {'L':>3} {'Attn agreement':>15} {'FFN norm':>10}")
        print(f"  {'─'*30}")

        for li in range(n_layers):
            if actual_attn_out[li] is None:
                continue

            # Attention agreement: mean vs actual
            mean_out = mean_attn[li]
            attn_cos = F.cosine_similarity(
                mean_out.unsqueeze(0),
                actual_attn_out[li].unsqueeze(0)
            ).item()

            ffn_norm = actual_post_ffn[li].norm().item() if actual_post_ffn[li] is not None else 0

            if li % 3 == 0 or li == n_layers - 1:
                print(f"  L{li:>2} {attn_cos:>15.4f} {ffn_norm:>10.1f}")


# ---------------------------------------------------------------------------
# Phase 4: Analytical shortcut
# ---------------------------------------------------------------------------

def analytical_shortcut(model, tokenizer, device, mean_attn, n_layers):
    """Skip all 34 layers: embedding + scaffold + answer direction."""
    print(f"\n{'='*70}")
    print(f"  PHASE 4: ANALYTICAL SHORTCUT")
    print(f"{'='*70}")

    embed_weight = model.model.language_model.embed_tokens.weight.data.float()
    hidden = model.config.text_config.hidden_size

    # Scaffold = sum of all mean attention outputs
    scaffold = torch.zeros(hidden)
    for li in range(n_layers):
        scaffold += mean_attn[li]

    print(f"  Scaffold norm: {scaffold.norm():.1f}")

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for prompt, entity, expected in FACTUAL:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        last_pos = inputs["input_ids"].shape[1] - 1

        # Initial embedding at prediction position
        with torch.no_grad():
            emb = model.model.language_model.embed_tokens(inputs["input_ids"])
            emb = emb * math.sqrt(hidden)
        initial = emb[0, last_pos].float().cpu()

        # Answer direction: the answer's embedding
        answer_ids = get_answer_ids(tokenizer, expected)
        if answer_ids:
            answer_emb = embed_weight[list(answer_ids)[0]]
        else:
            answer_emb = torch.zeros(hidden)

        # Entity direction
        entity_ids = get_answer_ids(tokenizer, entity)
        if entity_ids:
            entity_emb = embed_weight[list(entity_ids)[0]]
        else:
            entity_emb = torch.zeros(hidden)

        # Try different scale factors
        best_rank = 999999
        for answer_scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            for entity_scale in [0.0, 0.5, 1.0, 2.0]:
                shortcut = initial + scaffold + answer_emb * answer_scale + entity_emb * entity_scale
                logits = shortcut @ embed_weight.T
                rank = answer_rank(logits, answer_ids)
                if rank < best_rank:
                    best_rank = rank
                    best_scales = (answer_scale, entity_scale)

        # Actual top-5
        with torch.no_grad():
            actual_logits = model(**inputs).logits[0, -1].float()
        actual_rank = answer_rank(actual_logits, answer_ids)

        in_top5 = best_rank <= 5
        in_top1 = best_rank == 1
        if in_top1:
            correct_top1 += 1
        if in_top5:
            correct_top5 += 1
        total += 1

        print(f"  '{prompt[:40]}' → {expected}: "
              f"shortcut=#{best_rank} (scales={best_scales}), "
              f"actual=#{actual_rank}")

    print(f"\n  Shortcut accuracy: top-1={correct_top1}/{total} ({correct_top1/total:.0%}), "
          f"top-5={correct_top5}/{total} ({correct_top5/total:.0%})")

    return correct_top1, correct_top5, total


# ---------------------------------------------------------------------------
# Phase 5: Correction frequency
# ---------------------------------------------------------------------------

def correction_frequency(model, tokenizer, device, n_layers, mean_attn):
    """How often do you need real attention to maintain accuracy?"""
    print(f"\n{'='*70}")
    print(f"  PHASE 5: CORRECTION FREQUENCY")
    print(f"{'='*70}")

    from experiment_layer_sweep import evaluate_config

    configs = [
        ("Every layer (baseline)", set(range(n_layers))),
        ("Every other", set(range(0, n_layers, 2))),
        ("Every 3rd", set(range(0, n_layers, 3))),
        ("Every 4th", set(range(0, n_layers, 4))),
        ("Every 6th", set(range(0, n_layers, 6))),
        ("Every 8th", set(range(0, n_layers, 8))),
        ("Only L0, L17, L33", {0, 17, 33}),
        ("Only L0, L11, L22, L33", {0, 11, 22, 33}),
    ]

    print(f"\n  {'Config':<35} {'Real':>5} {'Top-1':>7} {'Top-5':>7}")
    print(f"  {'─'*58}")

    for name, layers in configs:
        r = evaluate_config(model, tokenizer, mean_attn, layers, device, n_layers)
        print(f"  {name:<35} {len(layers):>5} "
              f"{r['top1_pct']:>6.0%} {r['top5_pct']:>6.0%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  THE FORWARD PASS IS PREDICTABLE")
    print("  Trajectory prediction from input parse")
    print("=" * 70)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    print(f"  Loaded: {n_layers} layers")

    # Extract mean attention outputs
    print(f"\n  Extracting mean attention outputs...")
    from experiment_layer_sweep import extract_mean_attn
    mean_attn = extract_mean_attn(model, tokenizer, device, n_layers)

    # Phase 2: Trajectory
    all_cosines = trajectory_comparison(model, tokenizer, device, n_layers, mean_attn)

    # Summary of trajectory prediction
    print(f"\n  Trajectory prediction summary:")
    print(f"  {'Layer':>5} {'Mean cosine':>12} {'Min cosine':>12}")
    print(f"  {'─'*32}")
    for li in range(n_layers):
        layer_cosines = [c[li] for c in all_cosines if li < len(c)]
        if layer_cosines:
            mean_c = sum(layer_cosines) / len(layer_cosines)
            min_c = min(layer_cosines)
            if li % 3 == 0 or li == n_layers - 1:
                print(f"  L{li:>3} {mean_c:>12.4f} {min_c:>12.4f}")

    # Phase 3: Drift diagnosis
    diagnose_drift(model, tokenizer, device, n_layers, mean_attn)

    # Phase 4: Analytical shortcut
    s_t1, s_t5, s_total = analytical_shortcut(model, tokenizer, device, mean_attn, n_layers)

    # Phase 5: Correction frequency
    try:
        correction_frequency(model, tokenizer, device, n_layers, mean_attn)
    except ImportError:
        print(f"\n  Phase 5 skipped (needs experiment_layer_sweep.py)")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    # Average cosine at key layers
    mid_cosines = [c[n_layers//2] for c in all_cosines if n_layers//2 < len(c)]
    late_cosines = [c[-1] for c in all_cosines if len(c) > 0]
    avg_mid = sum(mid_cosines) / len(mid_cosines) if mid_cosines else 0
    avg_late = sum(late_cosines) / len(late_cosines) if late_cosines else 0

    print(f"\n  Trajectory cosine at L{n_layers//2}: {avg_mid:.4f}")
    print(f"  Trajectory cosine at L{n_layers-1}: {avg_late:.4f}")
    print(f"  Analytical shortcut: top-5 = {s_t5}/{s_total} ({s_t5/s_total:.0%})")

    if avg_late > 0.90:
        print(f"\n  ✓ TRAJECTORY IS PREDICTABLE")
        print(f"    Mean attention + real FFN tracks the actual forward pass.")
    elif avg_late > 0.70:
        print(f"\n  ~ TRAJECTORY PARTIALLY PREDICTABLE")
        print(f"    Drifts but maintains rough direction.")
    else:
        print(f"\n  ✗ TRAJECTORY DIVERGES")
        print(f"    Predicted and actual paths diverge significantly.")

    if s_t5 / s_total > 0.4:
        print(f"\n  ✓ ANALYTICAL SHORTCUT WORKS ({s_t5/s_total:.0%} top-5)")
        print(f"    The model IS approximately: embed + scaffold + answer.")
    elif s_t5 / s_total > 0.1:
        print(f"\n  ~ SHORTCUT PARTIALLY WORKS ({s_t5/s_total:.0%} top-5)")
    else:
        print(f"\n  ✗ SHORTCUT FAILS ({s_t5/s_total:.0%} top-5)")
        print(f"    The 34-layer iteration is essential, not collapsible.")

    # Save
    save_data = {
        "trajectory_cosines_summary": {
            "mid_layer": avg_mid,
            "last_layer": avg_late,
        },
        "shortcut": {"top1": s_t1, "top5": s_t5, "total": s_total},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
