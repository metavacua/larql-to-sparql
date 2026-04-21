#!/usr/bin/env python3
"""
Behavioral classification of low-rank attention heads.

For each target head, projects last-token query vectors onto the SVD
eigenvectors of W_Q_h × W_K_h^T, then clusters to find actual runtime modes.

The SVD rank is the *capacity* upper bound. This script measures *utilization*:
how many distinct modes each head actually uses across diverse inputs.

Target heads:
  L0H4   rank=4   most constrained head in model
  L0H3   rank=7
  L19H3  rank=10
  L20H4  rank=10
  L20H5  rank=13   (for comparison)

Usage:
  python3 scripts/low_rank_head_modes.py [--n-clusters auto]
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Target heads ─────────────────────────────────────────────────────────

TARGET_HEADS = [
    (0,  4, "L0H4",  4),
    (0,  3, "L0H3",  7),
    (19, 3, "L19H3", 10),
    (20, 4, "L20H4", 10),
    (20, 5, "L20H5", 13),
]

# ── Diverse prompt set ────────────────────────────────────────────────────

PROMPTS = [
    # factual: geography
    ("factual", "The capital of France is"),
    ("factual", "The capital of Germany is"),
    ("factual", "The capital of Japan is"),
    ("factual", "The capital of Australia is"),
    ("factual", "The capital of Brazil is"),
    ("factual", "The capital of Canada is"),
    ("factual", "The capital of Egypt is"),
    ("factual", "The capital of India is"),
    # factual: people
    ("factual", "Albert Einstein was born in"),
    ("factual", "Marie Curie was born in"),
    ("factual", "Python was created by"),
    ("factual", "The theory of relativity was proposed by"),
    # factual: places/things
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
    # questions / reasoning
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
    # creative / narrative
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
    # emotional / personal
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


def compute_qk_svd(W_Q_h, W_K_h):
    """SVD of W_Q_h @ W_K_h^T → (U, S, Vt)."""
    M = W_Q_h @ W_K_h.T   # (head_dim, head_dim)
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    return U, S, Vt


def project_query(q_vec, U, top_k):
    """Project query vector onto top-k left singular vectors."""
    # q_vec: (head_dim,)  U: (head_dim, head_dim)
    proj = U[:, :top_k].T @ q_vec   # (top_k,)
    return proj


def cluster_projections(projections, n_clusters, labels):
    """K-means cluster the projected query vectors, report mode assignments."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    X = np.stack([p.numpy() for p in projections])
    X_norm = normalize(X)  # cluster by direction, not magnitude

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(X_norm)
    cluster_ids = km.labels_

    # Report which prompt types fall in each cluster
    mode_contents = defaultdict(list)
    for i, (cid, (ptype, prompt)) in enumerate(zip(cluster_ids, labels)):
        mode_contents[cid].append((ptype, prompt))

    return cluster_ids, mode_contents, km.inertia_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--n-clusters", default="auto",
                        help="Number of clusters per head, or 'auto' (= SVD rank)")
    args = parser.parse_args()

    print(f"=== Low-rank Head Mode Analysis — {args.model} ===\n")
    print(f"  Prompts:  {len(PROMPTS)}")
    print(f"  Targets:  {', '.join(t[2] for t in TARGET_HEADS)}\n")

    print("Loading model + tokenizer...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    layers = get_layers(model)

    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    gqa_group = num_q_heads // num_kv_heads

    print(f"  {len(layers)} layers, {num_q_heads} Q-heads, head_dim={head_dim}\n")

    # ── Precompute SVDs for target heads ─────────────────────────────────

    head_svds = {}
    for layer_idx, head_idx, name, svd_rank in TARGET_HEADS:
        attn = layers[layer_idx].self_attn
        W_Q = attn.q_proj.weight.detach()   # (num_q * head_dim, hidden)
        W_K = attn.k_proj.weight.detach()   # (num_kv * head_dim, hidden)

        q_slice = W_Q[head_idx * head_dim:(head_idx + 1) * head_dim, :]
        kv_group = head_idx // gqa_group
        k_slice = W_K[kv_group * head_dim:(kv_group + 1) * head_dim, :]

        U, S, Vt = compute_qk_svd(q_slice, k_slice)
        head_svds[(layer_idx, head_idx)] = (name, svd_rank, q_slice, U, S)

        print(f"  SVD {name}: top-{svd_rank} singular values = "
              f"{S[:min(svd_rank,8)].tolist()[:8]}")

    print()

    # ── Q-norm weights (Gemma 3 uses per-head RMSNorm on Q and K) ─────────
    # If q_norm exists, apply it before projecting

    def get_q_norm(layer_idx, head_idx):
        attn = layers[layer_idx].self_attn
        # Gemma 3 has q_norm and k_norm (per-head RMSNorm)
        if hasattr(attn, "q_norm"):
            # q_norm weight shape is (head_dim,) or (num_q_heads, head_dim)?
            w = attn.q_norm.weight.detach()
            if w.ndim == 2:
                return w[head_idx]   # (head_dim,)
            return w   # (head_dim,)
        return None

    # ── Capture query vectors via forward hooks ────────────────────────────

    captured_queries = defaultdict(list)   # (layer, head) → list of query vectors

    def make_pre_hook(layer_idx):
        # pre-hook on the decoder layer: first positional arg is hidden_states
        def hook(module, args, kwargs):
            # args[0] or kwargs['hidden_states'] is (batch, seq, hidden)
            if args:
                x = args[0].detach()
            else:
                x = kwargs.get("hidden_states", kwargs.get("x")).detach()
            if x.ndim == 3:
                x = x[0]   # (seq, hidden)
            x_last = x[-1]   # (hidden,)

            attn = layers[layer_idx].self_attn
            W_Q = attn.q_proj.weight.detach()
            for head_idx in [h for (li, h, _, _) in TARGET_HEADS if li == layer_idx]:
                q_slice = W_Q[head_idx * head_dim:(head_idx + 1) * head_dim, :]
                q_vec = q_slice @ x_last   # (head_dim,)

                # Apply Q-norm if present (Gemma 3 uses per-head RMSNorm)
                q_norm_w = get_q_norm(layer_idx, head_idx)
                if q_norm_w is not None:
                    rms = torch.sqrt((q_vec ** 2).mean() + 1e-6)
                    q_vec = (q_vec / rms) * q_norm_w

                captured_queries[(layer_idx, head_idx)].append(q_vec)
        return hook

    # Register pre-hooks on target decoder layers
    target_layers = set(li for li, _, _, _ in TARGET_HEADS)
    hooks = []
    for li in target_layers:
        h = layers[li].register_forward_pre_hook(make_pre_hook(li), with_kwargs=True)
        hooks.append(h)

    # ── Run forward passes ────────────────────────────────────────────────

    print("Running forward passes...", flush=True)
    with torch.no_grad():
        for pi, (ptype, prompt) in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            _ = model(**enc)
            if (pi + 1) % 10 == 0:
                print(f"  {pi + 1}/{len(PROMPTS)}", flush=True)

    for h in hooks:
        h.remove()

    print(f"  Done. Captured {len(PROMPTS)} query vectors per head.\n")

    # ── Cluster and report ────────────────────────────────────────────────

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        have_sklearn = True
    except ImportError:
        have_sklearn = False
        print("WARNING: sklearn not available, skipping clustering\n")

    for layer_idx, head_idx, name, svd_rank, *_ in TARGET_HEADS:
        print(f"{'='*60}")
        print(f"  {name}  (SVD rank={svd_rank})")
        print(f"{'='*60}\n")

        q_vecs = captured_queries.get((layer_idx, head_idx), [])
        if not q_vecs:
            print("  No data captured.\n")
            continue

        _, _, q_slice, U, S = head_svds[(layer_idx, head_idx)]

        # Project all query vectors onto top-rank eigenvectors
        projections = [project_query(q, U, svd_rank) for q in q_vecs]

        # Show projection norms and variance
        proj_array = torch.stack(projections).numpy()   # (N, rank)
        print(f"  Projection statistics (onto top-{svd_rank} eigenvectors):")
        print(f"  {'Mode':>6}  {'σ_weight':>9}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
        for k in range(svd_rank):
            col = proj_array[:, k]
            print(f"  {k:>6}  {S[k].item():>9.2f}  {col.mean():>8.3f}  "
                  f"{col.std():>8.3f}  {col.min():>8.3f}  {col.max():>8.3f}")

        print()

        # Clustering
        if have_sklearn:
            n_clusters = svd_rank if args.n_clusters == "auto" else int(args.n_clusters)
            cluster_ids, mode_contents, inertia = cluster_projections(
                projections, n_clusters, PROMPTS
            )

            print(f"  K-means (k={n_clusters}, inertia={inertia:.4f}):\n")
            for cid in sorted(mode_contents.keys()):
                items = mode_contents[cid]
                type_counts = defaultdict(int)
                for ptype, _ in items:
                    type_counts[ptype] += 1
                types_str = ", ".join(f"{t}×{c}" for t, c in sorted(type_counts.items()))
                print(f"  Mode {cid} ({len(items)} prompts): {types_str}")
                for ptype, prompt in items[:4]:
                    print(f"    [{ptype}] {prompt[:60]}")
                if len(items) > 4:
                    print(f"    ... +{len(items)-4} more")
                print()

            # Try elbow: fit k=2..rank+2 and show inertia drop
            print(f"  Inertia by k (looking for elbow):")
            prev_inertia = None
            for k in range(1, svd_rank + 3):
                km = KMeans(n_clusters=k, n_init=20, random_state=42)
                X = normalize(proj_array)
                km.fit(X)
                drop = f"  Δ={prev_inertia - km.inertia_:.4f}" if prev_inertia else ""
                print(f"    k={k}: {km.inertia_:.4f}{drop}")
                prev_inertia = km.inertia_
            print()

        # Print raw projection table for first 20 prompts
        print(f"  First 20 projected queries (top-{min(svd_rank,4)} dims shown):")
        show_dims = min(svd_rank, 4)
        header = "  " + f"{'Type':>12}  " + "  ".join(f"{'e'+str(k):>7}" for k in range(show_dims))
        print(header)
        print("  " + "─" * (14 + 9 * show_dims))
        for i, (ptype, prompt) in enumerate(PROMPTS[:20]):
            row = proj_array[i]
            vals = "  ".join(f"{row[k]:>7.3f}" for k in range(show_dims))
            print(f"  {ptype:>12}  {vals}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
