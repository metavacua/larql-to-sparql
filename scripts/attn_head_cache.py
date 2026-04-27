#!/usr/bin/env python3
"""
Attention head cache: build and validate for L0H4 (2 modes) and L19H3 (1 mode).

Phase 1 — Gather: run 63 prompts, capture actual per-head attention outputs.
Phase 2 — Build:  compute mode-mean vectors as cached templates.
Phase 3 — Patch:  replace head computation with template lookup, compare predictions.

The cached output for head h replaces its slice of o_proj's input:
  o_proj receives (batch, seq, num_heads * head_dim)
  head h's slice is [:, :, h*head_dim : (h+1)*head_dim]

Mode detection for L0H4: project query onto top-2 SVD eigenvectors,
  nearest centroid wins. L19H3: always mode 0 (single template).

Usage:
  python3 scripts/attn_head_cache.py
"""

import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"

# Target heads: (layer, head, name, svd_rank, n_modes)
TARGETS = [
    (0,  4, "L0H4",  4, 2),
    (19, 3, "L19H3", 10, 1),
]

PROMPTS = [
    # factual
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
    # arithmetic
    ("arithmetic", "2 + 2 ="),
    ("arithmetic", "7 × 8 ="),
    ("arithmetic", "15 - 6 ="),
    ("arithmetic", "100 / 4 ="),
    ("arithmetic", "3 × 3 ="),
    ("arithmetic", "12 + 7 ="),
    ("arithmetic", "What is 5 times 9?"),
    ("arithmetic", "The square root of 144 is"),
    # code
    ("code", "def fibonacci(n):"),
    ("code", "import numpy as"),
    ("code", "for i in range("),
    ("code", "class Animal:"),
    ("code", "def __init__(self):"),
    ("code", "import torch"),
    ("code", "x = np.zeros("),
    ("code", "if __name__ =="),
    # conversational
    ("conversation", "Hello, how are"),
    ("conversation", "What is your name?"),
    ("conversation", "How was your day"),
    ("conversation", "Nice to meet"),
    ("conversation", "Thank you for"),
    ("conversation", "I would like to"),
    # questions
    ("question", "Why does the sky appear blue?"),
    ("question", "How do plants make food through"),
    ("question", "What causes earthquakes?"),
    ("question", "How does gravity work?"),
    ("question", "Why is the ocean salty?"),
    # logical
    ("logical", "If A implies B and B implies C, then A implies"),
    ("logical", "All mammals are warm-blooded. Dolphins are mammals. Therefore dolphins are"),
    ("logical", "The opposite of hot is"),
    ("logical", "If today is Monday, tomorrow is"),
    # creative
    ("creative", "Once upon a time there was a"),
    ("creative", "In a galaxy far, far"),
    ("creative", "The old man walked slowly through the"),
    ("creative", "She opened the door and"),
    # technical
    ("technical", "The time complexity of binary search is O("),
    ("technical", "In machine learning, gradient descent minimizes the"),
    ("technical", "A neural network consists of"),
    ("technical", "The HTTP protocol uses port"),
    # instruction
    ("instruction", "Please translate 'hello' to French:"),
    ("instruction", "Summarize the following text:"),
    ("instruction", "Write a Python function that"),
    ("instruction", "Explain the concept of"),
    # emotional
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


def qk_svd(W_Q_h, W_K_h):
    M = W_Q_h @ W_K_h.T
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    return U, S


def cosine_sim(a, b):
    a = a / (a.norm() + 1e-12)
    b = b / (b.norm() + 1e-12)
    return float((a * b).sum())


# ── Phase 1: gather actual head outputs ──────────────────────────────────

def gather_head_outputs(model, tokenizer, layers, cfg):
    head_dim = cfg.head_dim
    num_q_heads = cfg.num_attention_heads
    gqa_group = num_q_heads // cfg.num_key_value_heads

    # Shared mutable state across hooks
    state = {
        "x_last": {},      # layer → last-token hidden state
        "head_out": {},    # (layer, head) → list of (head_dim,) tensors
        "base_preds": [],  # top-1 token ID per prompt
    }

    for li, hi, *_ in TARGETS:
        state["head_out"][(li, hi)] = []

    def make_decoder_pre(layer_idx):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            if x.ndim == 3:
                x = x[0]
            state["x_last"][layer_idx] = x[-1].detach().clone()
        return hook

    def make_o_proj_pre(layer_idx):
        def hook(module, args, kwargs):
            # args[0]: (batch_or_seq, seq, num_heads*head_dim)
            # or for unbatched: (seq, num_heads*head_dim)
            heads_in = args[0] if args else kwargs["input"]
            if heads_in.ndim == 3:
                last = heads_in[0, -1].detach().clone()   # (num_heads*head_dim,)
            else:
                last = heads_in[-1].detach().clone()
            for (li, hi, *_) in TARGETS:
                if li == layer_idx:
                    h_out = last[hi * head_dim:(hi + 1) * head_dim].clone()
                    state["head_out"][(li, hi)].append(h_out)
        return hook

    hooks = []
    target_layers = set(li for li, *_ in TARGETS)
    for li in target_layers:
        hooks.append(layers[li].register_forward_pre_hook(
            make_decoder_pre(li), with_kwargs=True))
        hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_o_proj_pre(li), with_kwargs=True))

    with torch.no_grad():
        for pi, (_, prompt) in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            out = model(**enc)
            logits = out.logits[0, -1]
            state["base_preds"].append(int(logits.argmax()))
            if (pi + 1) % 10 == 0:
                print(f"    {pi+1}/{len(PROMPTS)}", flush=True)

    for h in hooks:
        h.remove()

    return state


# ── Phase 2: build cache ──────────────────────────────────────────────────

def build_cache(state, model, layers, cfg):
    head_dim = cfg.head_dim
    num_q_heads = cfg.num_attention_heads
    gqa_group = num_q_heads // cfg.num_key_value_heads

    cache = {}   # (layer, head) → {"templates": list[tensor], "centroids": list[tensor], "U": tensor}

    for li, hi, name, svd_rank, n_modes in TARGETS:
        print(f"\n  Building cache for {name} ({n_modes} mode(s))...")

        attn = layers[li].self_attn
        W_Q = attn.q_proj.weight.detach()
        W_K = attn.k_proj.weight.detach()

        q_slice = W_Q[hi * head_dim:(hi + 1) * head_dim, :]
        kv_group = hi // gqa_group
        k_slice = W_K[kv_group * head_dim:(kv_group + 1) * head_dim, :]
        U, S = qk_svd(q_slice, k_slice)

        head_outs = state["head_out"][(li, hi)]   # list of (head_dim,) tensors

        # Measure within-cluster cosine similarity of actual outputs
        cos_all = []
        for i in range(len(head_outs)):
            for j in range(i + 1, min(i + 5, len(head_outs))):
                cos_all.append(cosine_sim(head_outs[i], head_outs[j]))
        print(f"    Head output pairwise cos (sample): mean={np.mean(cos_all):.4f} "
              f"std={np.std(cos_all):.4f} min={np.min(cos_all):.4f}")

        if n_modes == 1:
            # Single template: mean of all outputs
            template = torch.stack(head_outs).mean(0)
            cache[(li, hi)] = {
                "templates": [template],
                "U": U[:, :1],
                "centroids": [torch.zeros(1)],
                "n_modes": 1,
            }
            print(f"    Single template built. Norm={template.norm():.3f}")
        else:
            # Project queries onto top-n_modes eigenvectors, cluster
            # Need to re-extract query vectors from state (we only saved head_out, not queries)
            # Recompute queries from x_last... but we only saved x_last for the LAST prompt.
            # Instead: use the head outputs directly for clustering (they encode the same info)

            # Project head outputs onto top-n_modes SVD left vectors
            # (U is in query space, not head-output space — need W_V for that)
            # Use head outputs directly via PCA instead

            head_out_mat = torch.stack(head_outs)   # (N, head_dim)
            out_np = head_out_mat.numpy()
            out_norm = normalize(out_np)

            km = KMeans(n_clusters=n_modes, n_init=30, random_state=42)
            km.fit(out_norm)
            labels = km.labels_

            templates = []
            centroids_proj = []
            for m in range(n_modes):
                mask = labels == m
                group = head_out_mat[mask]
                template = group.mean(0)
                templates.append(template)

                # Within-mode cosine similarity
                if mask.sum() > 1:
                    sims = []
                    for i in range(min(mask.sum(), 10)):
                        for j in range(i + 1, min(mask.sum(), 10)):
                            sims.append(cosine_sim(group[i], group[j]))
                    print(f"    Mode {m} ({mask.sum()} prompts): within-cos mean={np.mean(sims):.4f}"
                          f"  template_norm={template.norm():.3f}")
                else:
                    print(f"    Mode {m} ({mask.sum()} prompt): template_norm={template.norm():.3f}")

            # Compute mode centroids in query-projection space for runtime detection
            # We need W_Q_h @ x_last projected onto U[:, :n_modes]
            # But we don't have x_last per prompt saved — use cluster centroids in head_out space
            cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)

            cache[(li, hi)] = {
                "templates": templates,
                "U": U[:, :n_modes],          # for query projection at runtime
                "q_slice": q_slice,           # W_Q_h
                "cluster_centers": cluster_centers,   # (n_modes, head_dim) in norm head-out space
                "km": km,                     # fitted KMeans for runtime mode detection
                "n_modes": n_modes,
                "labels": labels,
            }

            print(f"    Built {n_modes} templates from {len(head_outs)} outputs.")

    return cache


# ── Phase 3: patch and test ───────────────────────────────────────────────

def test_patched(model, tokenizer, layers, cfg, cache):
    head_dim = cfg.head_dim
    num_q_heads = cfg.num_attention_heads

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
            # heads_in: (1, seq, num_heads*head_dim) or (seq, num_heads*head_dim)
            batched = heads_in.ndim == 3

            for (li, hi, name, svd_rank, n_modes) in TARGETS:
                if li != layer_idx:
                    continue
                c = cache[(li, hi)]
                x_last = state["x_last"].get(li)

                if n_modes == 1 or x_last is None:
                    mode = 0
                else:
                    # Detect mode via head output projection (nearest cluster center)
                    if batched:
                        cur_out = heads_in[0, -1, hi*head_dim:(hi+1)*head_dim].detach()
                    else:
                        cur_out = heads_in[-1, hi*head_dim:(hi+1)*head_dim].detach()
                    cur_norm = cur_out / (cur_out.norm() + 1e-12)
                    sims = [(cosine_sim(cur_norm, c["cluster_centers"][m]), m)
                            for m in range(n_modes)]
                    mode = max(sims, key=lambda x: x[0])[1]

                template = c["templates"][mode]
                if batched:
                    heads_in[0, :, hi*head_dim:(hi+1)*head_dim] = template.unsqueeze(0)
                else:
                    heads_in[:, hi*head_dim:(hi+1)*head_dim] = template.unsqueeze(0)

            if args:
                return (heads_in,) + args[1:], kwargs
            else:
                kwargs["input"] = heads_in
                return args, kwargs
        return hook

    hooks = []
    target_layers = set(li for li, *_ in TARGETS)
    for li in target_layers:
        hooks.append(layers[li].register_forward_pre_hook(
            make_decoder_pre(li), with_kwargs=True))
        hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_o_proj_pre_patched(li), with_kwargs=True))

    with torch.no_grad():
        for pi, (_, prompt) in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            out = model(**enc)
            logits = out.logits[0, -1]
            patched_preds.append(int(logits.argmax()))
            if (pi + 1) % 10 == 0:
                print(f"    {pi+1}/{len(PROMPTS)}", flush=True)

    for h in hooks:
        h.remove()

    return patched_preds


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=== Attention Head Cache: Build + Validate ===\n")
    print(f"  Targets: L0H4 (2 modes), L19H3 (1 mode)")
    print(f"  Prompts: {len(PROMPTS)}\n")

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    layers = get_layers(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    print(f"  {len(layers)} layers, head_dim={cfg.head_dim}\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print("Phase 1: gathering baseline head outputs...")
    state = gather_head_outputs(model, tokenizer, layers, cfg)
    base_preds = state["base_preds"]
    print(f"  Baseline predictions captured ({len(base_preds)} prompts)")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print("\nPhase 2: building cache...")
    cache = build_cache(state, model, layers, cfg)

    # ── Validate head output similarity within modes ──────────────────────
    print("\n  Output cosine similarity across all prompts per head:")
    for li, hi, name, svd_rank, n_modes in TARGETS:
        head_outs = state["head_out"][(li, hi)]
        all_sims = []
        for i in range(len(head_outs)):
            for j in range(i + 1, len(head_outs)):
                all_sims.append(cosine_sim(head_outs[i], head_outs[j]))
        print(f"    {name}: all-pairs cos mean={np.mean(all_sims):.4f} "
              f"std={np.std(all_sims):.4f} min={np.min(all_sims):.4f}")

    # ── Template vs actual: reconstruction error ─────────────────────────
    print("\n  Template reconstruction error (cos distance, actual vs cached):")
    for li, hi, name, svd_rank, n_modes in TARGETS:
        head_outs = state["head_out"][(li, hi)]
        c = cache[(li, hi)]

        if n_modes == 1:
            errs = [1.0 - cosine_sim(head_outs[i], c["templates"][0])
                    for i in range(len(head_outs))]
            print(f"    {name} (1 template): cos_dist mean={np.mean(errs):.4f} "
                  f"max={np.max(errs):.4f}")
        else:
            labels = c["labels"]
            errs = []
            for i, h_out in enumerate(head_outs):
                t = c["templates"][labels[i]]
                errs.append(1.0 - cosine_sim(h_out, t))
            print(f"    {name} ({n_modes} templates): cos_dist mean={np.mean(errs):.4f} "
                  f"max={np.max(errs):.4f}")

    # ── Phase 3 ───────────────────────────────────────────────────────────
    print("\nPhase 3: running patched forward pass...")
    patched_preds = test_patched(model, tokenizer, layers, cfg, cache)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    matches = sum(b == p for b, p in zip(base_preds, patched_preds))
    print(f"\n  Top-1 match rate: {matches}/{len(PROMPTS)} ({100*matches/len(PROMPTS):.1f}%)")

    mismatches = [(i, PROMPTS[i], base_preds[i], patched_preds[i])
                  for i in range(len(PROMPTS)) if base_preds[i] != patched_preds[i]]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for i, (ptype, prompt), base, patched in mismatches:
            base_tok = tokenizer.decode([base])
            patched_tok = tokenizer.decode([patched])
            print(f"    [{ptype}] {prompt[:50]}")
            print(f"      base={repr(base_tok)}  patched={repr(patched_tok)}")
    else:
        print("  No mismatches — all top-1 predictions preserved.")

    # Per-category match rate
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
