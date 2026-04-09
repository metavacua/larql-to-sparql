#!/usr/bin/env python3
"""
Layer Sweep: Find the minimum real attention layers for 95%+ accuracy.

Reuses the derived attention infrastructure. Tests which combination of
real attention layers closes the gap from 83% to 100% top-5.
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

MODEL_NAME = "google/gemma-3-4b-pt"
MAX_SEQ = 64
OUTPUT_DIR = "results_layer_sweep"

TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Egypt is", "Cairo"),
    ("The capital of India is", "Delhi"),
    ("The official language of France is", "French"),
    ("The official language of Japan is", "Japanese"),
    ("The chemical symbol for gold is", "Au"),
    ("The Earth orbits the", "Sun"),
    ("The currency of Japan is the", "yen"),
    ("The currency of India is the", "rupee"),
    ("Once upon a time, there was a", None),
    ("The big dog runs near the", None),
    ("def fibonacci(n):\n    if n <= 1:\n        return", "n"),
    ("import pandas as", "pd"),
    ("To make scrambled eggs, first", None),
    ("I think the best approach would be to", None),
    ("If all cats are mammals, then", None),
    ("The most important discovery in physics was", None),
]


def extract_mean_attn(model, tokenizer, device, n_layers):
    """Measure mean attention output per layer from reference prompts."""
    ref_prompts = [
        "The capital of France is", "The capital of Japan is",
        "The official language of Germany is", "The currency of India is the",
        "Once upon a time, there was a", "def fibonacci(n):\n    if n <= 1:",
        "To make scrambled eggs, first", "I think the best approach is",
        "The detective opened the door and", "The big dog runs near the",
        "import pandas as", "If all cats are mammals, then",
    ]

    attn_outputs = [[] for _ in range(n_layers)]
    hooks = []

    for li in range(n_layers):
        def make_hook(idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                attn_outputs[idx].append(out[0, -1].detach().float().cpu())
            return hook
        hooks.append(model.model.language_model.layers[li].self_attn.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for prompt in ref_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            _ = model(**inputs)

    for h in hooks:
        h.remove()

    mean_attn = {}
    for li in range(n_layers):
        if attn_outputs[li]:
            mean_attn[li] = torch.stack(attn_outputs[li]).mean(0).cpu()
        else:
            mean_attn[li] = torch.zeros(model.config.text_config.hidden_size)

    return mean_attn


def evaluate_config(model, tokenizer, mean_attn, real_layers, device, n_layers):
    """Evaluate: real attention at real_layers, mean constants elsewhere."""

    active = [True]
    captured_pre = {}

    hooks = []
    for li in range(n_layers):
        layer = model.model.language_model.layers[li]

        if li in real_layers:
            # Keep real attention — no hook needed (or passthrough)
            continue

        mean_out = mean_attn[li].to(device)

        def make_hook(idx, mean_vec):
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

        hooks.append(layer.self_attn.register_forward_hook(make_hook(li, mean_out)))

    # Evaluate
    top1_correct = 0
    top5_correct = 0
    agree_top1 = 0
    total_factual = 0
    total_all = 0

    active[0] = True
    with torch.no_grad():
        for prompt, expected in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)

            # Need baseline for agreement
            # Run without hooks first (but hooks are already installed...)
            # Instead: compute derived result, compare post-hoc

            outputs = model(**inputs)
            logits = outputs.logits[0, -1].float()

            top1 = logits.argmax().item()
            top5 = set(logits.topk(5).indices.tolist())
            total_all += 1

            if expected:
                target_ids = set()
                for prefix in ["", " "]:
                    ids = tokenizer.encode(prefix + expected, add_special_tokens=False)
                    target_ids.update(ids[:2])
                if target_ids:
                    if top1 in target_ids:
                        top1_correct += 1
                    if target_ids & top5:
                        top5_correct += 1
                    total_factual += 1

    active[0] = False
    for h in hooks:
        h.remove()

    return {
        "top1": top1_correct,
        "top5": top5_correct,
        "total_factual": total_factual,
        "total_all": total_all,
        "top1_pct": top1_correct / total_factual if total_factual else 0,
        "top5_pct": top5_correct / total_factual if total_factual else 0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  LAYER SWEEP: Find minimum real attention for 95%+ accuracy")
    print("=" * 70)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.text_config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers")

    # Extract mean attention outputs
    print(f"\n  Extracting mean attention outputs...")
    mean_attn = extract_mean_attn(model, tokenizer, device, n_layers)

    # Baseline (all real)
    print(f"\n  Baseline (all real attention)...")
    baseline = evaluate_config(model, tokenizer, mean_attn, set(range(n_layers)),
                              device, n_layers)
    print(f"  Baseline: top-1={baseline['top1']}/{baseline['total_factual']} "
          f"({baseline['top1_pct']:.0%}), "
          f"top-5={baseline['top5']}/{baseline['total_factual']} "
          f"({baseline['top5_pct']:.0%})")

    # Configurations to test
    configs = [
        ("A: L0-5, L24 (current)", {0,1,2,3,4,5,24}),
        ("B: L0-5, L12, L24", {0,1,2,3,4,5,12,24}),
        ("C: L0-5, L17, L24", {0,1,2,3,4,5,17,24}),
        ("D: L0-8, L24", {0,1,2,3,4,5,6,7,8,24}),
        ("E: L0-5, L21, L24", {0,1,2,3,4,5,21,24}),
        ("F: L0-5, L12-19, L24", {0,1,2,3,4,5,12,13,14,15,16,17,18,19,24}),
        ("G: L0-5, L11, L17, L23, L24", {0,1,2,3,4,5,11,17,23,24}),
        ("H: L0-5, L11, L17, L23, L24, L29", {0,1,2,3,4,5,11,17,23,24,29}),
        ("I: All full-attn layers", {5,11,17,23,29}),
        ("J: L0-8", {0,1,2,3,4,5,6,7,8}),
        ("K: L0-5, L24, L33", {0,1,2,3,4,5,24,33}),
        ("L: L0-5, L12, L17, L24", {0,1,2,3,4,5,12,17,24}),
        ("M: Every 3rd layer", set(range(0, n_layers, 3))),
        ("N: Every other layer", set(range(0, n_layers, 2))),
        ("O: L0-11, L24", set(range(12)) | {24}),
        ("P: L0-5, L9-12, L24", {0,1,2,3,4,5,9,10,11,12,24}),
    ]

    print(f"\n  {'Config':<45} {'Layers':>6} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'─'*70}")
    print(f"  {'Baseline (all 34 real)':<45} {34:>6} "
          f"{baseline['top1_pct']:>7.0%} {baseline['top5_pct']:>7.0%}")

    results = []
    t0 = time.time()

    for name, layers in configs:
        r = evaluate_config(model, tokenizer, mean_attn, layers, device, n_layers)
        n_real = len(layers)
        n_derived = n_layers - n_real

        marker = ""
        if r["top5_pct"] >= 0.95:
            marker = " ← 95%+"
        elif r["top5_pct"] >= 0.90:
            marker = " ← 90%+"

        print(f"  {name:<45} {n_real:>6} "
              f"{r['top1_pct']:>7.0%} {r['top5_pct']:>7.0%}{marker}")

        results.append({
            "name": name, "layers": sorted(layers), "n_real": n_real,
            "top1_pct": r["top1_pct"], "top5_pct": r["top5_pct"],
            **r,
        })

    elapsed = time.time() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s")

    # Find optimal
    best_95 = [r for r in results if r["top5_pct"] >= 0.95]
    if best_95:
        optimal = min(best_95, key=lambda r: r["n_real"])
        print(f"\n  OPTIMAL (fewest real layers at ≥95% top-5):")
        print(f"    {optimal['name']}: {optimal['n_real']} real layers, "
              f"top-5={optimal['top5_pct']:.0%}")
        print(f"    → {n_layers - optimal['n_real']}/{n_layers} layers "
              f"({(n_layers - optimal['n_real'])/n_layers:.0%}) replaceable")
    else:
        best = max(results, key=lambda r: r["top5_pct"])
        print(f"\n  BEST (no config reached 95%):")
        print(f"    {best['name']}: top-5={best['top5_pct']:.0%}")

    # Save
    save_data = {
        "baseline": baseline,
        "configs": results,
        "optimal_95": best_95[0] if best_95 else None,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
