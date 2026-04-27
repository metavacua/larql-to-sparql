#!/usr/bin/env python3
"""
Where The Magic Actually Lives

Experiment A: Track the PREDICTION POSITION (last token) across all layers.
Does it carry the context register that BOS didn't?

Experiment B: Analyse the 46 non-BOS heads. What do they contribute?
Selective ablation: which head types are critical?

BOS was a fixed scaffold (confirmed). The question: where does the
input-specific routing signal live?
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict, Counter
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
MAX_SEQ = 64
OUTPUT_DIR = "results_prediction_position"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    "capital_of": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Australia is",
        "The capital of Germany is",
        "The capital of India is",
        "The capital of Canada is",
    ],
    "language_of": [
        "The official language of France is",
        "The official language of Japan is",
        "The official language of Spain is",
        "The official language of Brazil is",
    ],
    "france_queries": [
        "The capital of France is",
        "The president of France is",
        "The currency of France is the",
        "The population of France is",
        "France is located in",
    ],
    "creative": [
        "Once upon a time in a kingdom beneath the sea,",
        "The detective opened the door and saw",
        "She picked up the old violin and began to",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.",
        "for i in range(len(arr)):\n    if arr[i] >",
    ],
    "reasoning": [
        "If all birds can fly and penguins are birds, then",
        "Given that x plus y equals 10 and x equals 3, then y equals",
        "The probability of rolling two sixes is",
    ],
    "conversational": [
        "I have been thinking about changing careers because",
        "What do you think is the best way to",
        "That reminds me of something interesting about",
    ],
    "instructional": [
        "To make scrambled eggs, first",
        "The most effective way to learn a language is",
        "When debugging a program, start by",
    ],
}

# Head classifications from anatomy experiment
ANATOMY_PATH = "results_attention_anatomy/results.json"


# ---------------------------------------------------------------------------
# Experiment A: Prediction Position Tracking
# ---------------------------------------------------------------------------

def experiment_a(model, tokenizer, device, n_layers, hidden_dim):
    """Track the prediction position (last token) across all layers."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT A: PREDICTION POSITION IS THE CONTEXT REGISTER?")
    print(f"{'='*70}")

    # Hooks to capture residual at last position
    pred_pre_attn = [None] * n_layers
    pred_post_attn = [None] * n_layers
    pred_post_ffn = [None] * n_layers
    gate_input_last = [None] * n_layers

    hooks = []

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]

        def make_pre_attn(idx):
            def hook(module, args, kwargs):
                hs = kwargs.get('hidden_states')
                if hs is not None:
                    pred_pre_attn[idx] = hs[0, -1].detach().float().cpu()
            return hook

        def make_post_attn(idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                pred_post_attn[idx] = out[0, -1].detach().float().cpu()
            return hook

        def make_mlp_pre(idx):
            def hook(module, args):
                if args:
                    inp = args[0] if isinstance(args, tuple) else args
                    gate_input_last[idx] = inp[0, -1].detach().float().cpu()
            return hook

        def make_mlp_post(idx):
            def hook(module, args, output):
                pred_post_ffn[idx] = output[0, -1].detach().float().cpu()
            return hook

        hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_attn(li), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(make_post_attn(li)))
        hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post(li)))

    # Collect trajectories
    print(f"\n  Collecting prediction position trajectories...")
    results = {}
    t0 = time.time()
    done = 0
    total = sum(len(v) for v in PROMPTS.values())

    with torch.no_grad():
        for category, prompt_list in PROMPTS.items():
            for prompt in prompt_list:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                                 truncation=True).to(device)
                _ = model(**inputs)

                trajectory = []
                for li in range(n_layers):
                    t = {}
                    for name, arr in [("pre_attn", pred_pre_attn),
                                      ("post_attn", pred_post_attn),
                                      ("post_ffn", pred_post_ffn),
                                      ("gate_input", gate_input_last)]:
                        if arr[li] is not None:
                            t[name] = arr[li]
                    if "pre_attn" in t and "post_attn" in t:
                        t["attn_delta"] = t["post_attn"] - t["pre_attn"]
                        t["attn_norm"] = t["attn_delta"].norm().item()
                    if "post_ffn" in t:
                        t["ffn_norm"] = t["post_ffn"].norm().item()
                    trajectory.append(t)

                results[(category, prompt)] = trajectory
                done += 1
                if done % 10 == 0:
                    print(f"    {done}/{total} ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()

    print(f"  Collected {len(results)} trajectories in {time.time()-t0:.0f}s")

    # --- Similarity Analysis ---
    print(f"\n  Similarity analysis (prediction position):")
    print(f"  {'Layer':>5} {'Same-Tmpl':>10} {'Same-Ent':>10} {'Cross-Task':>10} {'Gradient':>10}")
    print(f"  {'─'*48}")

    similarity = []
    for li in range(n_layers):
        def get_vec(cat, prompt):
            traj = results.get((cat, prompt), [])
            return traj[li].get("post_attn") if li < len(traj) else None

        # Same template (capital_of): should be HIGH
        cap_vecs = [get_vec("capital_of", p) for p in PROMPTS["capital_of"]]
        cap_vecs = [v for v in cap_vecs if v is not None]
        st_cos = pairwise_mean_cosine(cap_vecs)

        # Same entity (france_queries): should be MODERATE
        fr_vecs = [get_vec("france_queries", p) for p in PROMPTS["france_queries"]]
        fr_vecs = [v for v in fr_vecs if v is not None]
        se_cos = pairwise_mean_cosine(fr_vecs)

        # Cross-task: factual mean vs creative mean vs code mean
        task_means = {}
        for cat in ["capital_of", "creative", "code", "reasoning"]:
            vecs = [get_vec(cat, p) for p in PROMPTS.get(cat, [])]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                task_means[cat] = torch.stack(vecs).mean(0)

        ct_cos_list = []
        cats = sorted(task_means.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                cos = F.cosine_similarity(
                    task_means[cats[i]].unsqueeze(0),
                    task_means[cats[j]].unsqueeze(0)
                ).item()
                ct_cos_list.append(cos)
        ct_cos = sum(ct_cos_list) / len(ct_cos_list) if ct_cos_list else 0

        gradient = st_cos - ct_cos
        similarity.append({
            "layer": li,
            "same_template": round(st_cos, 4),
            "same_entity": round(se_cos, 4),
            "cross_task": round(ct_cos, 4),
            "gradient": round(gradient, 4),
        })

        if li % 3 == 0 or li == n_layers - 1:
            print(f"  L{li:>3} {st_cos:>10.4f} {se_cos:>10.4f} "
                  f"{ct_cos:>10.4f} {gradient:>10.4f}")

    # --- Divergence tracking ---
    print(f"\n  Divergence tracking (France vs Japan, France vs Code):")
    france = results.get(("capital_of", "The capital of France is"), [])
    japan = results.get(("capital_of", "The capital of Japan is"), [])
    code_prompt = PROMPTS["code"][0]
    code = results.get(("code", code_prompt), [])

    print(f"  {'Layer':>5} {'France↔Japan':>13} {'France↔Code':>13}")
    print(f"  {'─'*34}")
    for li in range(n_layers):
        fj = 0
        fc = 0
        if li < len(france) and li < len(japan):
            fv = france[li].get("post_attn")
            jv = japan[li].get("post_attn")
            if fv is not None and jv is not None:
                fj = F.cosine_similarity(fv.unsqueeze(0), jv.unsqueeze(0)).item()
        if li < len(france) and li < len(code):
            fv = france[li].get("post_attn")
            cv = code[li].get("post_attn")
            if fv is not None and cv is not None:
                fc = F.cosine_similarity(fv.unsqueeze(0), cv.unsqueeze(0)).item()

        if li % 3 == 0 or li == n_layers - 1:
            e_mark = " ← entity diverges" if fj < 0.90 and fj > 0 else ""
            t_mark = " ← task diverges" if fc < 0.50 and fc > 0 else ""
            print(f"  L{li:>3} {fj:>13.4f} {fc:>13.4f}{e_mark}{t_mark}")

    # --- Pred→FFN correlation ---
    print(f"\n  Prediction position → FFN correlation:")
    pred_ffn_corr = []
    for li in range(n_layers):
        pred_states = []
        gate_states = []
        for key, traj in results.items():
            if li < len(traj):
                p = traj[li].get("post_attn")
                g = traj[li].get("gate_input")
                if p is not None and g is not None:
                    pred_states.append(p)
                    gate_states.append(g)

        if len(pred_states) < 5:
            pred_ffn_corr.append({"layer": li, "correlation": 0})
            continue

        P = torch.stack(pred_states)
        G = torch.stack(gate_states)
        Pn = F.normalize(P, dim=1)
        Gn = F.normalize(G, dim=1)
        p_cos = (Pn @ Pn.T).flatten()
        g_cos = (Gn @ Gn.T).flatten()

        pm = p_cos - p_cos.mean()
        gm = g_cos - g_cos.mean()
        corr = ((pm * gm).sum() / (pm.norm() * gm.norm() + 1e-10)).item()

        pred_ffn_corr.append({"layer": li, "correlation": round(corr, 4)})

        if li % 5 == 0 or li == n_layers - 1:
            marker = " ← HIGH" if corr > 0.7 else ""
            print(f"    L{li}: {corr:.4f}{marker}")

    return results, similarity, pred_ffn_corr


def pairwise_mean_cosine(vecs):
    """Mean pairwise cosine similarity."""
    if len(vecs) < 2:
        return 0
    stack = torch.stack(vecs)
    norm = F.normalize(stack, dim=1)
    cos_mat = norm @ norm.T
    mask = torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()
    vals = cos_mat[mask]
    return vals.mean().item() if vals.numel() > 0 else 0


# ---------------------------------------------------------------------------
# Experiment B: Non-BOS Head Ablation
# ---------------------------------------------------------------------------

def experiment_b(model, tokenizer, device, n_layers, n_heads):
    """Selective ablation of head types."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT B: SELECTIVE HEAD ABLATION")
    print(f"{'='*70}")

    # Load head classifications
    if os.path.exists(ANATOMY_PATH):
        with open(ANATOMY_PATH) as f:
            anatomy = json.load(f)
        head_roles = anatomy.get("m8_role_summary", {})
    else:
        print(f"  No anatomy results — skipping ablation")
        return {}

    # Group heads by type
    heads_by_type = defaultdict(list)
    for key, role in head_roles.items():
        li = int(key.split("H")[0][1:])
        hi = int(key.split("H")[1])
        heads_by_type[role].append((li, hi))

    print(f"\n  Head types:")
    for role, heads in sorted(heads_by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {role}: {len(heads)} heads")

    # Factual test prompts
    factual = [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Germany is", "Berlin"),
        ("The official language of France is", "French"),
        ("The official language of Japan is", "Japanese"),
        ("The chemical symbol for gold is", "Au"),
        ("The Earth orbits the", "Sun"),
        ("The currency of Japan is the", "yen"),
    ]

    def measure_accuracy(active_hook):
        """Measure top-1 and top-5 accuracy on factual prompts."""
        correct_1 = 0
        correct_5 = 0
        total = 0

        with torch.no_grad():
            for prompt, answer in factual:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                                 truncation=True).to(device)
                outputs = model(**inputs)
                logits = outputs.logits[0, -1].float()

                target_ids = set()
                for prefix in ["", " "]:
                    ids = tokenizer.encode(prefix + answer, add_special_tokens=False)
                    target_ids.update(ids[:2])

                if not target_ids:
                    continue

                top1 = logits.argmax().item()
                top5 = set(logits.topk(5).indices.tolist())

                if top1 in target_ids:
                    correct_1 += 1
                if target_ids & top5:
                    correct_5 += 1
                total += 1

        return correct_1 / total if total else 0, correct_5 / total if total else 0

    # Baseline
    print(f"\n  Baseline accuracy...")
    base_t1, base_t5 = measure_accuracy(None)
    print(f"  Baseline: top-1={base_t1:.0%}, top-5={base_t5:.0%}")

    # Ablation: zero out each head type
    ablation_results = {}

    for remove_type in ["BOS", "previous", "function_word", "self", "relation"]:
        heads = heads_by_type.get(remove_type, [])
        if not heads:
            continue

        # Hook to zero out specific heads
        zero_active = [True]

        def make_ablation_hook(target_heads, layer_idx, n_h, hd):
            heads_in_layer = [h for (l, h) in target_heads if l == layer_idx]
            if not heads_in_layer:
                return None

            def hook(module, args, output):
                if not zero_active[0]:
                    return output
                out = output[0] if isinstance(output, tuple) else output
                # Zero out the specified heads' contribution
                # Attention output is already projected through o_proj,
                # so we can't easily zero individual heads.
                # Instead: zero the entire attention output at this layer
                # if ANY target head is here (approximate but tests the idea)
                if heads_in_layer:
                    zeros = torch.zeros_like(out)
                    if isinstance(output, tuple):
                        return (zeros,) + output[1:]
                    return zeros
                return output
            return hook

        ablation_hooks = []
        for li in range(n_layers):
            layer = model.model.language_model.layers[li]
            hook_fn = make_ablation_hook(heads, li, n_heads,
                                        model.config.text_config.head_dim)
            if hook_fn:
                ablation_hooks.append(
                    layer.self_attn.register_forward_hook(hook_fn))

        zero_active[0] = True
        t1, t5 = measure_accuracy(zero_active)
        zero_active[0] = False

        for h in ablation_hooks:
            h.remove()

        n_layers_affected = len(set(l for l, h in heads))
        print(f"  Remove {remove_type:15s} ({len(heads):>3} heads, {n_layers_affected} layers): "
              f"top-1={t1:.0%} (Δ={t1-base_t1:+.0%}), "
              f"top-5={t5:.0%} (Δ={t5-base_t5:+.0%})")

        ablation_results[remove_type] = {
            "n_heads": len(heads),
            "n_layers_affected": n_layers_affected,
            "top1": t1,
            "top5": t5,
            "delta_top1": t1 - base_t1,
            "delta_top5": t5 - base_t5,
        }

    return ablation_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  WHERE THE MAGIC ACTUALLY LIVES")
    print("  Prediction position + 46 non-BOS heads")
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
    n_heads = tc.num_attention_heads
    hidden_dim = tc.hidden_size
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers}L × {n_heads}H")

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENT A
    # ═══════════════════════════════════════════════════════════════════

    a_results, a_similarity, a_pred_ffn = experiment_a(
        model, tokenizer, device, n_layers, hidden_dim)

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENT B
    # ═══════════════════════════════════════════════════════════════════

    b_results = experiment_b(model, tokenizer, device, n_layers, n_heads)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    # Check if prediction position shows gradient
    mid = n_layers // 2
    if mid < len(a_similarity):
        st = a_similarity[mid]["same_template"]
        ct = a_similarity[mid]["cross_task"]
        gradient = st - ct

        print(f"\n  Experiment A: Prediction position at L{mid}")
        print(f"    Same-template cosine: {st:.4f}")
        print(f"    Cross-task cosine: {ct:.4f}")
        print(f"    Gradient: {gradient:.4f}")

        if gradient > 0.1:
            print(f"\n  ✓ PREDICTION POSITION IS THE CONTEXT REGISTER")
            print(f"    It carries input-specific information that diverges by task type.")
            print(f"    The routing signal accumulates at the last token, not BOS.")
        elif gradient > 0.02:
            print(f"\n  ~ PARTIAL CONTEXT REGISTER")
            print(f"    Some gradient visible but weak.")
        else:
            print(f"\n  ✗ PREDICTION POSITION IS ALSO CONSTANT")
            print(f"    Context is distributed across ALL positions.")

    # Check ablation results
    if b_results:
        print(f"\n  Experiment B: Head ablation")
        most_critical = min(b_results.items(), key=lambda x: x[1]["delta_top1"])
        least_critical = max(b_results.items(), key=lambda x: x[1]["delta_top1"])

        print(f"    Most critical type: {most_critical[0]} "
              f"(Δ top-1 = {most_critical[1]['delta_top1']:+.0%})")
        print(f"    Least critical type: {least_critical[0]} "
              f"(Δ top-1 = {least_critical[1]['delta_top1']:+.0%})")

        # The magic number: how many heads actually matter?
        critical_heads = sum(v["n_heads"] for k, v in b_results.items()
                           if v["delta_top1"] < -0.1)
        total_heads = sum(v["n_heads"] for v in b_results.values())
        print(f"\n  Critical heads (cause >10% top-1 drop): {critical_heads}/{n_layers*n_heads}")
        print(f"  Scaffold heads: {n_layers*n_heads - critical_heads}/{n_layers*n_heads}")

    # Save
    save_data = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "similarity": a_similarity,
        "pred_ffn_correlation": a_pred_ffn,
        "ablation": {k: {kk: vv for kk, vv in v.items()}
                     for k, v in b_results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
