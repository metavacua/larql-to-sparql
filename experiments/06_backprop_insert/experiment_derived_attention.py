#!/usr/bin/env python3
"""
Building Attention From Scratch

Experiment A: Replace Gemma 3-4B attention with derived components
  - BOS constants (pre-computed, 83% of heads)
  - Previous-token V×O projections (7%)
  - Function-word V×O projections (4%)
  - Self heads DELETED, relation heads SKIPPED

Experiment B (if A works): derive from FFN geometry instead of Gemma weights.

Zero training. Just re-interpreting and assembling existing weights.
"""

import os
import sys
import json
import time
import math
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
MAX_SEQ = 64
OUTPUT_DIR = "results_derived_attention"
ANATOMY_PATH = "results_attention_anatomy/results.json"

# Function words for structural detection
FUNCTION_WORDS = [
    "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "and", "but", "or", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "it", "this", "that", "these", "those",
    "def", "class", "for", "while", "if", "else", "return",
    "import", "from", "try", "except", "with", "as",
    ":", "(", ")", "[", "]", "{", "}", ",", ".",
]

# Test prompts
TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Egypt is", "Cairo"),
    ("The official language of France is", "French"),
    ("The official language of Japan is", "Japanese"),
    ("The chemical symbol for gold is", "Au"),
    ("The Earth orbits the", "Sun"),
    ("The currency of Japan is the", "yen"),
    ("def fibonacci(n):\n    if n <= 1:\n        return", "n"),
    ("import pandas as", "pd"),
    ("Once upon a time, there was a", None),
    ("The big dog runs near the", None),
    ("She quickly ran to the", None),
    ("To make scrambled eggs, first", None),
    ("I think the best approach would be to", None),
    ("If all cats are mammals, then", None),
    ("The most important discovery in physics was", None),
    ("In the year 2050, scientists discovered that", None),
]


# ---------------------------------------------------------------------------
# Phase 1: Extract components from Gemma
# ---------------------------------------------------------------------------

def extract_components(model, tokenizer, head_roles, device):
    """
    Extract attention components from ACTUAL forward passes.
    BOS constants = measured mean attention output (not raw embedding × V × O).
    V×O projections extracted from weights for prev/func heads.
    """
    print(f"\n  Extracting attention components...")

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    n_heads = tc.num_attention_heads
    n_kv = tc.num_key_value_heads
    head_dim = tc.head_dim
    hidden = tc.hidden_size
    gqa_ratio = n_heads // n_kv

    # Function word token IDs
    func_ids = set()
    for word in FUNCTION_WORDS:
        for variant in [word, " " + word, word.capitalize()]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                func_ids.add(encoded[0])
    print(f"  Function word token IDs: {len(func_ids)}")

    # --- Step 1: Measure ACTUAL mean attention output per layer ---
    print(f"  Measuring mean attention outputs from reference prompts...")

    ref_prompts = [
        "The capital of France is", "The capital of Japan is",
        "The capital of Germany is", "The capital of Egypt is",
        "The official language of France is", "The currency of India is the",
        "Once upon a time, there was a", "The detective opened the door and",
        "def fibonacci(n):\n    if n <= 1:", "import pandas as",
        "To make scrambled eggs, first", "I think the best approach is",
    ]

    # Per-position mean attention output at each layer
    attn_outputs = [[] for _ in range(n_layers)]
    hooks = []

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]
        def make_hook(idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                attn_outputs[idx].append(out.detach().float().cpu())
            return hook
        hooks.append(layer.self_attn.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for prompt in ref_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            _ = model(**inputs)

    for h in hooks:
        h.remove()

    # Mean attention output per layer — measured from actual forward passes.
    # Use the last position from each prompt (prediction position).
    mean_attn = {}
    for li in range(n_layers):
        if attn_outputs[li]:
            last_pos_outputs = [out[0, -1] for out in attn_outputs[li]]  # (hidden,) each
            mean_attn[li] = torch.stack(last_pos_outputs).mean(0).cpu()
        else:
            mean_attn[li] = torch.zeros(hidden)

    print(f"  Mean attn norms: L0={mean_attn[0].norm():.1f}, "
          f"L17={mean_attn[17].norm():.1f}, L33={mean_attn[33].norm():.1f}")

    # --- Step 2: Extract V×O for non-BOS heads ---
    prev_count = 0
    func_count = 0
    skip_count = 0
    bos_count = 0

    components = {}
    for li in range(n_layers):
        layer = model.model.language_model.layers[li]
        attn = layer.self_attn
        v_weight = attn.v_proj.weight.data.float()
        o_weight = attn.o_proj.weight.data.float()

        prev_projs = []
        func_projs = []

        for h in range(n_heads):
            key = f"L{li}H{h}"
            role = head_roles.get(key, "BOS")

            if role == "BOS":
                bos_count += 1
                continue
            elif role in ("self", "relation"):
                skip_count += 1
                continue

            kv_head = h // gqa_ratio
            v_slice = v_weight[kv_head * head_dim:(kv_head + 1) * head_dim, :]
            o_slice = o_weight[:, h * head_dim:(h + 1) * head_dim]

            if role == "previous":
                prev_projs.append((v_slice.cpu(), o_slice.cpu()))
                prev_count += 1
            elif role == "function_word":
                func_projs.append((v_slice.cpu(), o_slice.cpu()))
                func_count += 1

        components[li] = {
            "mean_attn": mean_attn[li],
            "prev_projs": prev_projs,
            "func_projs": func_projs,
        }

    print(f"  BOS: {bos_count} (measured mean), Previous: {prev_count}, "
          f"Function: {func_count}, Skipped: {skip_count}")

    return components, func_ids


# ---------------------------------------------------------------------------
# Phase 2: Build derived attention model
# ---------------------------------------------------------------------------

def run_derived_forward(model, components, func_ids, input_ids, device):
    """
    Run Gemma forward pass with attention replaced by derived components.
    Uses hooks to intercept and replace attention output.
    """
    tc = model.config.text_config
    n_layers = tc.num_hidden_layers

    active = [True]

    def make_replacement_hook(li):
        comp = components.get(li)
        if comp is None:
            # None = keep real attention (for hybrid mode)
            def hook(module, args, output):
                return output
            return hook

        mean_out = comp["mean_attn"].to(device)
        prev_list = comp["prev_projs"]
        func_list = comp["func_projs"]

        def hook(module, args, output):
            if not active[0]:
                return output

            # Get the normed input (what attention would have received)
            # The hook fires on self_attn — args should contain hidden_states
            # but Gemma passes via kwargs, so we read from the pre-hook capture

            out_tensor = output[0] if isinstance(output, tuple) else output
            B, S, D = out_tensor.shape

            # Build replacement
            replacement = torch.zeros_like(out_tensor)

            # 1. Mean attention output (measured from actual forward passes)
            replacement += mean_out.unsqueeze(0).unsqueeze(0)

            # 2. Previous-token projections
            # We need the input to attention (normed residual)
            # Use the captured pre-attn state
            pre_attn = captured_pre_attn.get(li)
            if pre_attn is not None:
                for v_slice, o_slice in prev_list:
                    v_s = v_slice.to(device)
                    o_s = o_slice.to(device)
                    for t in range(1, S):
                        prev_emb = pre_attn[0, t - 1]  # (hidden,)
                        v_out = prev_emb @ v_s.T  # (head_dim,)
                        head_out = o_s @ v_out  # (hidden,)
                        replacement[0, t] += head_out

                # 3. Function-word projections
                for v_slice, o_slice in func_list:
                    v_s = v_slice.to(device)
                    o_s = o_slice.to(device)
                    token_list = input_ids[0].tolist()
                    func_positions = [i for i, tid in enumerate(token_list)
                                     if tid in func_ids]
                    for t in range(S):
                        for fp in reversed(func_positions):
                            if fp <= t:
                                func_emb = pre_attn[0, fp]
                                v_out = func_emb @ v_s.T
                                head_out = o_s @ v_out
                                replacement[0, t] += head_out
                                break

            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement

        return hook

    # Pre-attention capture hooks
    captured_pre_attn = {}

    def make_pre_hook(li):
        def hook(module, args, kwargs):
            hs = kwargs.get('hidden_states')
            if hs is not None and active[0]:
                captured_pre_attn[li] = hs.detach().float()
        return hook

    # Install hooks
    hooks = []
    for li in range(n_layers):
        layer = model.model.language_model.layers[li]
        hooks.append(layer.self_attn.register_forward_pre_hook(
            make_pre_hook(li), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(
            make_replacement_hook(li)))

    # Run forward
    active[0] = True
    with torch.no_grad():
        outputs = model(input_ids)

    active[0] = False
    for h in hooks:
        h.remove()

    return outputs


# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, components, func_ids, test_prompts, device, mode="derived"):
    """Evaluate a configuration."""
    results = {
        "top1_correct": 0, "top5_correct": 0,
        "agreement_top1": 0, "agreement_top5": [],
        "total_factual": 0, "total_all": 0,
    }

    for prompt, expected in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)

        # Baseline (full attention)
        with torch.no_grad():
            baseline_out = model(**inputs)
        base_logits = baseline_out.logits[0, -1].float()
        base_top1 = base_logits.argmax().item()
        base_top5 = set(base_logits.topk(5).indices.tolist())

        # Derived attention
        if mode == "derived":
            derived_out = run_derived_forward(model, components, func_ids,
                                            inputs["input_ids"], device)
            der_logits = derived_out.logits[0, -1].float()
        elif mode == "skip":
            # Skip attention entirely (zero it out)
            skip_components = {
                li: {"mean_attn": torch.zeros(model.config.text_config.hidden_size),
                     "prev_projs": [], "func_projs": []}
                for li in range(model.config.text_config.num_hidden_layers)
            }
            derived_out = run_derived_forward(model, skip_components, set(),
                                            inputs["input_ids"], device)
            der_logits = derived_out.logits[0, -1].float()
        elif mode == "bos_only":
            bos_components = {
                li: {"mean_attn": components[li]["mean_attn"],
                     "prev_projs": [], "func_projs": []}
                for li in components
            }
            derived_out = run_derived_forward(model, bos_components, set(),
                                            inputs["input_ids"], device)
            der_logits = derived_out.logits[0, -1].float()
        elif mode == "hybrid_essential":
            # Keep REAL attention at essential layers (L0-5, L24)
            # Use derived (mean) at dispensable layers (L6-23, L25-33)
            essential = {0, 1, 2, 3, 4, 5, 24}
            hybrid_components = {}
            for li in components:
                if li in essential:
                    # Use a sentinel: None means "keep real attention"
                    hybrid_components[li] = None
                else:
                    hybrid_components[li] = components[li]
            derived_out = run_derived_forward(model, hybrid_components, func_ids,
                                            inputs["input_ids"], device)
            der_logits = derived_out.logits[0, -1].float()
        else:
            der_logits = base_logits

        der_top1 = der_logits.argmax().item()
        der_top5 = set(der_logits.topk(5).indices.tolist())

        # Agreement
        results["agreement_top1"] += (base_top1 == der_top1)
        results["agreement_top5"].append(len(base_top5 & der_top5) / 5)
        results["total_all"] += 1

        # Factual accuracy
        if expected:
            target_ids = set()
            for prefix in ["", " "]:
                ids = tokenizer.encode(prefix + expected, add_special_tokens=False)
                target_ids.update(ids[:2])

            if target_ids:
                if der_top1 in target_ids:
                    results["top1_correct"] += 1
                if target_ids & der_top5:
                    results["top5_correct"] += 1
                results["total_factual"] += 1

    return results


def print_results(name, results):
    n_all = results["total_all"]
    n_fact = results["total_factual"]
    t1 = results["top1_correct"]
    t5 = results["top5_correct"]
    agree_t1 = results["agreement_top1"]
    agree_t5 = sum(results["agreement_top5"]) / len(results["agreement_top5"]) if results["agreement_top5"] else 0

    print(f"  {name:<45} "
          f"fact_t1={t1}/{n_fact}({t1/n_fact:.0%}) "
          f"fact_t5={t5}/{n_fact}({t5/n_fact:.0%}) "
          f"agree_t1={agree_t1}/{n_all}({agree_t1/n_all:.0%}) "
          f"agree_t5={agree_t5:.0%}")


# ---------------------------------------------------------------------------
# Phase 4: Generation test
# ---------------------------------------------------------------------------

def generation_test(model, tokenizer, components, func_ids, prompts, device,
                    max_new=30, mode="derived"):
    """Generate text with derived attention."""
    print(f"\n  Generation ({mode}):")

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids[0].tolist()

        for _ in range(max_new):
            if len(generated) >= MAX_SEQ:
                break
            ids = torch.tensor([generated], device=device)

            if mode == "derived":
                out = run_derived_forward(model, components, func_ids, ids, device)
            else:
                with torch.no_grad():
                    out = model(ids)

            logits = out.logits[0, -1].float()
            # Greedy
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token in (0, 1):
                break

        text = tokenizer.decode(generated, skip_special_tokens=True)
        gen_part = text[len(prompt):] if text.startswith(prompt) else text
        print(f"    '{prompt[:50]}' → {gen_part[:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  BUILDING ATTENTION FROM SCRATCH")
    print("  Experiment A: Derived components from Gemma 3-4B weights")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"\n  Device: CPU (float32)")

    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Load head classifications
    if os.path.exists(ANATOMY_PATH):
        with open(ANATOMY_PATH) as f:
            anatomy = json.load(f)
        head_roles = anatomy.get("m8_role_summary", {})
        print(f"  Loaded {len(head_roles)} head classifications")
    else:
        print(f"  WARNING: No anatomy results, defaulting all to BOS")
        n_layers = model.config.text_config.num_hidden_layers
        n_heads = model.config.text_config.num_attention_heads
        head_roles = {f"L{l}H{h}": "BOS" for l in range(n_layers) for h in range(n_heads)}

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Extract components
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: EXTRACT COMPONENTS")
    print(f"{'='*70}")

    components, func_ids = extract_components(model, tokenizer, head_roles, device)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Baseline
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: BASELINE (full attention)")
    print(f"{'='*70}")

    baseline = evaluate(model, tokenizer, components, func_ids,
                       TEST_PROMPTS, device, mode="baseline")
    print_results("Gemma 3-4B (full attention)", baseline)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Derived attention configurations
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: DERIVED ATTENTION CONFIGURATIONS")
    print(f"{'='*70}")

    t0 = time.time()

    # Config 1: Skip attention entirely
    skip = evaluate(model, tokenizer, components, func_ids,
                   TEST_PROMPTS, device, mode="skip")
    print_results("Skip attention entirely", skip)
    print(f"    ({time.time()-t0:.0f}s)")

    # Config 2: BOS constants only
    t1 = time.time()
    bos = evaluate(model, tokenizer, components, func_ids,
                  TEST_PROMPTS, device, mode="bos_only")
    print_results("BOS constants only", bos)
    print(f"    ({time.time()-t1:.0f}s)")

    # Config 3: Full derived (BOS + prev + func)
    t2 = time.time()
    derived = evaluate(model, tokenizer, components, func_ids,
                      TEST_PROMPTS, device, mode="derived")
    print_results("Derived (BOS + prev + func)", derived)
    print(f"    ({time.time()-t2:.0f}s)")

    # Config 4: Hybrid — real attention at 7 essential layers, derived at 27 others
    t3 = time.time()
    hybrid = evaluate(model, tokenizer, components, func_ids,
                     TEST_PROMPTS, device, mode="hybrid_essential")
    print_results("Hybrid (real@L0-5,L24 + derived@rest)", hybrid)
    print(f"    ({time.time()-t3:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Generation test
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 4: GENERATION TEST")
    print(f"{'='*70}")

    gen_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "def fibonacci(n):\n    ",
        "To make scrambled eggs, first",
        "The most important discovery in physics was",
    ]

    generation_test(model, tokenizer, components, func_ids, gen_prompts,
                   device, mode="baseline")
    generation_test(model, tokenizer, components, func_ids, gen_prompts,
                   device, mode="derived")

    # Hybrid generation — real attention at essential layers
    print(f"\n  Generation (hybrid — real@L0-5,L24):")
    # For hybrid generation, need to wire it through run_derived_forward
    # with essential layers kept. Build hybrid components.
    essential = {0, 1, 2, 3, 4, 5, 24}
    hybrid_comp = {li: (None if li in essential else components[li])
                   for li in components}
    for prompt in gen_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids[0].tolist()
        for _ in range(30):
            if len(generated) >= MAX_SEQ:
                break
            ids = torch.tensor([generated], device=device)
            out = run_derived_forward(model, hybrid_comp, func_ids, ids, device)
            next_token = out.logits[0, -1].float().argmax().item()
            generated.append(next_token)
            if next_token in (0, 1):
                break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        gen_part = text[len(prompt):] if text.startswith(prompt) else text
        print(f"    '{prompt[:50]}' → {gen_part[:80]}")

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    n = baseline["total_all"]
    nf = baseline["total_factual"]

    print(f"\n  {'Config':<45} {'Fact T1':>8} {'Fact T5':>8} {'Agree T1':>9}")
    print(f"  {'─'*75}")
    for name, r in [("Full attention (baseline)", baseline),
                    ("Skip attention", skip),
                    ("BOS constants only", bos),
                    ("Derived (BOS + prev + func)", derived),
                    ("Hybrid (real@essential + derived)", hybrid)]:
        ft1 = f"{r['top1_correct']}/{nf}" if nf else "—"
        ft5 = f"{r['top5_correct']}/{nf}" if nf else "—"
        at1 = f"{r['agreement_top1']}/{n}"
        print(f"  {name:<45} {ft1:>8} {ft5:>8} {at1:>9}")

    # Key comparisons
    d_agree = derived["agreement_top1"] / n if n else 0
    s_agree = skip["agreement_top1"] / n if n else 0
    b_agree = bos["agreement_top1"] / n if n else 0
    h_agree = hybrid["agreement_top1"] / n if n else 0

    if d_agree > s_agree + 0.05:
        print(f"\n  ✓ DERIVED BEATS SKIP ({d_agree:.0%} vs {s_agree:.0%})")
        print(f"    The previous + function-word heads contribute real signal.")
    if d_agree > b_agree + 0.05:
        print(f"  ✓ DERIVED BEATS BOS-ONLY ({d_agree:.0%} vs {b_agree:.0%})")
        print(f"    Non-BOS heads add meaningful information.")

    # Hybrid result — the key test
    if h_agree > 0.8:
        print(f"\n  ✓ HYBRID WORKS ({h_agree:.0%} agreement)")
        print(f"    Real attention at 7 essential layers + derived at 27 others.")
        print(f"    79% of attention IS replaceable with measured constants.")
        print(f"    The 7 essential layers carry the input-specific routing.")
    elif h_agree > 0.5:
        print(f"\n  ~ HYBRID PARTIAL ({h_agree:.0%} agreement)")
        print(f"    Better than fully derived ({d_agree:.0%}) but gaps remain.")
    else:
        print(f"\n  ✗ HYBRID INSUFFICIENT ({h_agree:.0%} agreement)")

    if d_agree > 0.8:
        print(f"\n  ✓ FULL DERIVED WORKS ({d_agree:.0%} agreement)")
    elif d_agree > 0.5:
        print(f"\n  ~ PARTIAL ({d_agree:.0%} agreement)")
    else:
        print(f"\n  ✗ FULL DERIVED INSUFFICIENT ({d_agree:.0%} agreement)")

    # Save
    save_data = {
        "model": MODEL_NAME,
        "baseline": {k: v for k, v in baseline.items() if k != "agreement_top5"},
        "skip": {k: v for k, v in skip.items() if k != "agreement_top5"},
        "bos_only": {k: v for k, v in bos.items() if k != "agreement_top5"},
        "derived": {k: v for k, v in derived.items() if k != "agreement_top5"},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
