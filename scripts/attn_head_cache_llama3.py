#!/usr/bin/env python3
"""
Attention head cache: cross-model validation on Llama 3 8B.

Targets: deep low-rank heads (no layer-0 effect):
  L29H24  rank=2  (91% depth)   — most constrained deep head
  L14H26  rank=3  (44% depth)   — mid-network specialisation

Same three-phase protocol as attn_head_cache.py.
Validates that cacheable Q specialisation generalises beyond Gemma 3 4B.
"""

import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

# (layer, head, name, svd_rank, n_modes)
TARGETS = [
    (29, 24, "L29H24", 2, 2),
    (14, 26, "L14H26", 3, 2),
]

PROMPTS = [
    ("factual", "The capital of France is"),
    ("factual", "The capital of Germany is"),
    ("factual", "The capital of Japan is"),
    ("factual", "The capital of Australia is"),
    ("factual", "The capital of Brazil is"),
    ("factual", "The capital of Canada is"),
    ("factual", "The capital of Egypt is"),
    ("factual", "The capital of India is"),
    ("factual", "Albert Einstein was born in"),
    ("factual", "Marie Curie was born in"),
    ("factual", "Python was created by"),
    ("factual", "The theory of relativity was proposed by"),
    ("factual", "The Eiffel Tower is located in"),
    ("factual", "Mount Everest is located in"),
    ("factual", "The Amazon River flows through"),
    ("factual", "The Great Wall is located in"),
    ("arithmetic", "2 + 2 ="),
    ("arithmetic", "7 × 8 ="),
    ("arithmetic", "15 - 6 ="),
    ("arithmetic", "100 / 4 ="),
    ("arithmetic", "3 × 3 ="),
    ("arithmetic", "12 + 7 ="),
    ("arithmetic", "What is 5 times 9?"),
    ("arithmetic", "The square root of 144 is"),
    ("code", "def fibonacci(n):"),
    ("code", "import numpy as"),
    ("code", "for i in range("),
    ("code", "class Animal:"),
    ("code", "def __init__(self):"),
    ("code", "import torch"),
    ("code", "x = np.zeros("),
    ("code", "if __name__ =="),
    ("conversation", "Hello, how are"),
    ("conversation", "What is your name?"),
    ("conversation", "How was your day"),
    ("conversation", "Nice to meet"),
    ("conversation", "Thank you for"),
    ("conversation", "I would like to"),
    ("question", "Why does the sky appear blue?"),
    ("question", "How do plants make food through"),
    ("question", "What causes earthquakes?"),
    ("question", "How does gravity work?"),
    ("question", "Why is the ocean salty?"),
    ("logical", "If A implies B and B implies C, then A implies"),
    ("logical", "All mammals are warm-blooded. Dolphins are mammals. Therefore dolphins are"),
    ("logical", "The opposite of hot is"),
    ("logical", "If today is Monday, tomorrow is"),
    ("creative", "Once upon a time there was a"),
    ("creative", "In a galaxy far, far"),
    ("creative", "The old man walked slowly through the"),
    ("creative", "She opened the door and"),
    ("technical", "The time complexity of binary search is O("),
    ("technical", "In machine learning, gradient descent minimizes the"),
    ("technical", "A neural network consists of"),
    ("technical", "The HTTP protocol uses port"),
    ("instruction", "Please translate 'hello' to French:"),
    ("instruction", "Summarize the following text:"),
    ("instruction", "Write a Python function that"),
    ("instruction", "Explain the concept of"),
    ("emotional", "I am feeling very happy because"),
    ("emotional", "Today was a difficult day because"),
    ("emotional", "My favorite food is"),
    ("emotional", "I have always wanted to"),
]


def get_layers(model):
    for path in [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = path(model)
            if layers:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers")


def get_cfg(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    if not hasattr(cfg, "head_dim"):
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg


def qk_svd(W_Q_h, W_K_h):
    M = (W_Q_h @ W_K_h.T).float()
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return U, S


def cosine_sim(a, b):
    a = a.float() / (a.float().norm() + 1e-12)
    b = b.float() / (b.float().norm() + 1e-12)
    return float((a * b).sum())


def gather_head_outputs(model, tokenizer, layers, cfg):
    head_dim = cfg.head_dim
    gqa_group = cfg.num_attention_heads // cfg.num_key_value_heads

    state = {
        "x_last": {},
        "head_out": {(li, hi): [] for li, hi, *_ in TARGETS},
        "base_preds": [],
    }

    def make_decoder_pre(layer_idx):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            if x.ndim == 3:
                x = x[0]
            state["x_last"][layer_idx] = x[-1].detach().clone()
        return hook

    def make_o_proj_pre(layer_idx):
        def hook(module, args, kwargs):
            heads_in = args[0] if args else kwargs["input"]
            last = heads_in[0, -1].detach().clone() if heads_in.ndim == 3 else heads_in[-1].detach().clone()
            for li, hi, *_ in TARGETS:
                if li == layer_idx:
                    state["head_out"][(li, hi)].append(last[hi * head_dim:(hi + 1) * head_dim].clone())
        return hook

    hooks = []
    for li in set(li for li, *_ in TARGETS):
        hooks.append(layers[li].register_forward_pre_hook(make_decoder_pre(li), with_kwargs=True))
        hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(make_o_proj_pre(li), with_kwargs=True))

    with torch.no_grad():
        for pi, (_, prompt) in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            out = model(**enc)
            state["base_preds"].append(int(out.logits[0, -1].argmax()))
            if (pi + 1) % 10 == 0:
                print(f"    {pi+1}/{len(PROMPTS)}", flush=True)

    for h in hooks:
        h.remove()
    return state


def build_cache(state, layers, cfg):
    head_dim = cfg.head_dim
    gqa_group = cfg.num_attention_heads // cfg.num_key_value_heads
    cache = {}

    for li, hi, name, svd_rank, n_modes in TARGETS:
        print(f"\n  Building cache for {name} ({n_modes} mode(s))...")
        attn = layers[li].self_attn
        W_Q = attn.q_proj.weight.detach()
        W_K = attn.k_proj.weight.detach()
        q_slice = W_Q[hi * head_dim:(hi + 1) * head_dim, :]
        k_slice = W_K[(hi // gqa_group) * head_dim:((hi // gqa_group) + 1) * head_dim, :]
        U, S = qk_svd(q_slice, k_slice)
        print(f"    SVD top-{svd_rank} σ = {S[:svd_rank].tolist()}")

        head_outs = state["head_out"][(li, hi)]
        cos_sample = [cosine_sim(head_outs[i], head_outs[j])
                      for i in range(len(head_outs))
                      for j in range(i + 1, min(i + 5, len(head_outs)))]
        print(f"    Pairwise cos (sample): mean={np.mean(cos_sample):.4f} "
              f"std={np.std(cos_sample):.4f} min={np.min(cos_sample):.4f}")

        head_out_mat = torch.stack([h.float() for h in head_outs])
        out_norm = normalize(head_out_mat.numpy())
        km = KMeans(n_clusters=n_modes, n_init=30, random_state=42)
        km.fit(out_norm)
        labels = km.labels_

        templates, cluster_centers = [], []
        for m in range(n_modes):
            mask = labels == m
            group = head_out_mat[mask]
            tmpl = group.mean(0)
            templates.append(tmpl)
            cluster_centers.append(torch.tensor(km.cluster_centers_[m], dtype=torch.float32))
            if mask.sum() > 1:
                sims = [cosine_sim(group[i], group[j])
                        for i in range(min(int(mask.sum()), 10))
                        for j in range(i + 1, min(int(mask.sum()), 10))]
                print(f"    Mode {m} ({mask.sum()} prompts): within-cos mean={np.mean(sims):.4f}  "
                      f"template_norm={tmpl.norm():.3f}")
            else:
                print(f"    Mode {m} (1 prompt): template_norm={tmpl.norm():.3f}")

        cache[(li, hi)] = {
            "templates": templates,
            "cluster_centers": cluster_centers,
            "labels": labels,
            "n_modes": n_modes,
        }
        print(f"    Built {n_modes} templates from {len(head_outs)} outputs.")

    return cache


def test_patched(model, tokenizer, layers, cfg, cache):
    head_dim = cfg.head_dim
    patched_preds = []
    state = {"x_last": {}}

    def make_decoder_pre(layer_idx):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            if x.ndim == 3:
                x = x[0]
            state["x_last"][layer_idx] = x[-1].detach().clone()
        return hook

    def make_o_proj_pre_patched(layer_idx):
        def hook(module, args, kwargs):
            heads_in = (args[0] if args else kwargs["input"]).clone()
            batched = heads_in.ndim == 3
            for li, hi, name, svd_rank, n_modes in TARGETS:
                if li != layer_idx:
                    continue
                c = cache[(li, hi)]
                if n_modes == 1:
                    mode = 0
                else:
                    cur_out = (heads_in[0, -1, hi*head_dim:(hi+1)*head_dim] if batched
                               else heads_in[-1, hi*head_dim:(hi+1)*head_dim]).detach().float()
                    cur_norm = cur_out / (cur_out.norm() + 1e-12)
                    mode = max(range(n_modes),
                               key=lambda m: float((cur_norm * c["cluster_centers"][m]).sum()))

                tmpl = c["templates"][mode].to(heads_in.dtype)
                if batched:
                    heads_in[0, :, hi*head_dim:(hi+1)*head_dim] = tmpl.unsqueeze(0)
                else:
                    heads_in[:, hi*head_dim:(hi+1)*head_dim] = tmpl.unsqueeze(0)

            return ((heads_in,) + args[1:], kwargs) if args else (args, {**kwargs, "input": heads_in})
        return hook

    hooks = []
    for li in set(li for li, *_ in TARGETS):
        hooks.append(layers[li].register_forward_pre_hook(make_decoder_pre(li), with_kwargs=True))
        hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_o_proj_pre_patched(li), with_kwargs=True))

    with torch.no_grad():
        for pi, (_, prompt) in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            out = model(**enc)
            patched_preds.append(int(out.logits[0, -1].argmax()))
            if (pi + 1) % 10 == 0:
                print(f"    {pi+1}/{len(PROMPTS)}", flush=True)

    for h in hooks:
        h.remove()
    return patched_preds


def main():
    targets_str = ", ".join(f"{n} (rank={r}, {m} modes)" for _, _, n, r, m in TARGETS)
    print(f"=== Attention Head Cache: Llama 3 8B cross-model validation ===\n")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Targets: {targets_str}")
    print(f"  Prompts: {len(PROMPTS)}\n")

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    layers = get_layers(model)
    cfg = get_cfg(model)
    print(f"  {len(layers)} layers, head_dim={cfg.head_dim}, "
          f"GQA group={cfg.num_attention_heads // cfg.num_key_value_heads}\n")

    print("Phase 1: gathering baseline head outputs...")
    state = gather_head_outputs(model, tokenizer, layers, cfg)
    print(f"  Baseline predictions captured ({len(state['base_preds'])} prompts)")

    print("\nPhase 2: building cache...")
    cache = build_cache(state, layers, cfg)

    print("\n  Template reconstruction error:")
    for li, hi, name, svd_rank, n_modes in TARGETS:
        head_outs = [h.float() for h in state["head_out"][(li, hi)]]
        c = cache[(li, hi)]
        errs = [1.0 - cosine_sim(head_outs[i], c["templates"][c["labels"][i]])
                for i in range(len(head_outs))]
        print(f"    {name} ({n_modes} templates): cos_dist mean={np.mean(errs):.4f} max={np.max(errs):.4f}")

    print("\nPhase 3: running patched forward pass...")
    patched_preds = test_patched(model, tokenizer, layers, cfg, cache)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    base_preds = state["base_preds"]
    matches = sum(b == p for b, p in zip(base_preds, patched_preds))
    print(f"\n  Top-1 match rate: {matches}/{len(PROMPTS)} ({100*matches/len(PROMPTS):.1f}%)")

    mismatches = [(i, PROMPTS[i], base_preds[i], patched_preds[i])
                  for i in range(len(PROMPTS)) if base_preds[i] != patched_preds[i]]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for i, (ptype, prompt), base, pat in mismatches:
            print(f"    [{ptype}] {prompt[:50]}")
            print(f"      base={repr(tokenizer.decode([base]))}  patched={repr(tokenizer.decode([pat]))}")
    else:
        print("  No mismatches.")

    print("\n  Per-category match rate:")
    by_type = defaultdict(lambda: [0, 0])
    for i, (ptype, _) in enumerate(PROMPTS):
        by_type[ptype][1] += 1
        if base_preds[i] == patched_preds[i]:
            by_type[ptype][0] += 1
    for ptype in sorted(by_type.keys()):
        ok, total = by_type[ptype]
        print(f"    {ptype:>12}: {ok}/{total}")

    print("\nDone.")


if __name__ == "__main__":
    main()
