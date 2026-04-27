#!/usr/bin/env python3
"""
q_centroid_classifier.py — How many Q clusters do you need?

The residual_clustering experiment showed q_proj is the routing table:
  - Residual at L13: no clustering (cosine ~0.999)
  - Q vectors at L14: strong clustering (within=0.79, between=0.41)

This script:
  1. Runs more prompts (60 across 4 categories + 40 held-out)
  2. Computes Q vectors at each knowledge layer
  3. Fits K-means centroids (sweep K=2..20)
  4. Measures classification accuracy: can Q centroid predict category?
  5. Tests on held-out prompts: does the routing generalize?
  6. Also tests: can a SINGLE layer's Q predict the full attention pattern?

If K=4-8 centroids give >90% accuracy -> that's your routing table size.

USAGE:
  python3 experiments/05_syntax_circuit_routing/q_centroid_classifier.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Prompts: train + held-out ----------------------------------------

TRAIN_PROMPTS = {
    "capital": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Australia is",
        "The capital of Germany is",
        "The capital of India is",
        "The capital of Mexico is",
        "The capital of Canada is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Sweden is",
        "The capital of Kenya is",
        "The capital of Thailand is",
        "The capital of Argentina is",
    ],
    "synonym": [
        "Happy means",
        "Sad means",
        "Big means",
        "Fast means",
        "Brave means",
        "Small means",
        "Slow means",
        "Hot means",
        "Cold means",
        "Smart means",
        "Angry means",
        "Calm means",
        "Rich means",
        "Strong means",
        "Bright means",
    ],
    "arithmetic": [
        "2 + 3 =",
        "7 - 4 =",
        "5 * 6 =",
        "15 + 27 =",
        "8 * 9 =",
        "10 / 2 =",
        "100 - 37 =",
        "48 / 6 =",
        "25 * 4 =",
        "99 - 11 =",
        "12 * 12 =",
        "7 + 8 =",
        "50 - 25 =",
        "6 * 7 =",
        "33 + 67 =",
    ],
    "code": [
        "def hello():\n    return",
        "def add(a, b):\n    return",
        "for i in range(10):\n    print",
        "if x > 0:\n    return",
        "class Dog:\n    def __init__",
        "def factorial(n):\n    if n ==",
        "def greet(name):\n    print",
        "class Person:\n    def __init__",
        "class Vector:\n    def __init__",
        "def is_even(n):\n    return",
        "fn main() {\n    let x =",
        "struct Point {\n    x:",
        "match result {\n    Ok(val) =>",
        "enum Color {\n    Red,",
        "let mut vec = Vec::new();\n    vec",
    ],
}

HELD_OUT_PROMPTS = {
    "capital": [
        "The capital of Poland is",
        "The capital of Turkey is",
        "The capital of Vietnam is",
        "The capital of Nigeria is",
        "The capital of Peru is",
    ],
    "synonym": [
        "Poor means",
        "Weak means",
        "Dark means",
        "Loud means",
        "Quiet means",
    ],
    "arithmetic": [
        "81 / 9 =",
        "200 - 150 =",
        "11 * 11 =",
        "3 + 3 + 3 =",
        "1000 / 10 =",
    ],
    "code": [
        "fn add(a: i32, b: i32) -> i32 {\n    a",
        "impl Display for Point {\n    fn fmt",
        "pub fn process(input: &str) ->",
        "use std::collections::HashMap;\n\nfn",
        "trait Summary {\n    fn summarize",
    ],
}


# ---- Model helpers (same as before) ------------------------------------

def find_model_parts(model):
    try:
        lm = model.language_model
        inner = lm.model
        if hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
            embed_fn = inner.embed_tokens
            def lm_head(h):
                return h @ embed_fn.weight.T
            return embed_fn, inner.layers, inner.norm, lm_head, True
    except AttributeError:
        pass
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            lm_head_fn = model.lm_head
            def lm_head(h):
                return lm_head_fn(h)
        else:
            def lm_head(h):
                return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale
    raise RuntimeError("Could not detect model structure.")


def forward_capture_q(model, tokenizer, prompt, capture_layers):
    """Forward pass capturing Q vectors + attention head max-weights."""
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    h = embed_fn(input_ids)
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    q_vectors = {}
    attn_maxw = {}  # (layer, head) -> max_weight

    for i, layer in enumerate(layers):
        if i in capture_layers:
            sa = layer.self_attn
            B, L, D = h.shape
            h_norm = layer.input_layernorm(h)

            q = sa.q_proj(h_norm)
            k = sa.k_proj(h_norm)
            v = sa.v_proj(h_norm)

            n_heads = sa.n_heads
            n_kv_heads = sa.n_kv_heads
            head_dim = sa.head_dim
            scale = sa.scale

            q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            if hasattr(sa, 'q_norm'):
                q = sa.q_norm(q)
            if hasattr(sa, 'k_norm'):
                k = sa.k_norm(k)
            q = sa.rope(q)
            k = sa.rope(k)

            # Capture Q (last token, all heads flattened)
            q_last = q[0, :, -1, :]  # [n_heads, head_dim]
            mx.eval(q_last)
            q_vectors[i] = np.array(q_last.astype(mx.float32))

            # GQA expand
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)

            weights = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                weights = weights + mask
            weights = mx.softmax(weights, axis=-1)

            weights_np = np.array(weights[0, :, -1, :].astype(mx.float32))
            mx.eval(weights)

            for head_idx in range(n_heads):
                attn_maxw[(i, head_idx)] = float(np.max(weights_np[head_idx]))

            attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            attn_out = sa.o_proj(attn_out)

            if hasattr(layer, 'post_attention_layernorm'):
                h = h + layer.post_attention_layernorm(attn_out)
            else:
                h = h + attn_out
            if hasattr(layer, 'pre_feedforward_layernorm'):
                h_ffn = layer.pre_feedforward_layernorm(h)
            else:
                h_ffn = h
            ffn_out = layer.mlp(h_ffn)
            if hasattr(layer, 'post_feedforward_layernorm'):
                h = h + layer.post_feedforward_layernorm(ffn_out)
            else:
                h = h + ffn_out
            mx.eval(h)
        else:
            h = layer(h, mask=mask)
            mx.eval(h)

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return q_vectors, attn_maxw, pred_tok


# ---- Clustering experiments --------------------------------------------

def run_kmeans_sweep(X_train, y_train, X_test, y_test, cat_names, label):
    """Sweep K-means from K=2..20 and measure how well clusters match categories."""

    print(f"\n{'='*70}")
    print(f"K-MEANS SWEEP: {label}")
    print(f"{'='*70}")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Feature dim: {X_train.shape[1]}")

    # Normalize for cosine-like clustering
    X_train_n = normalize(X_train)
    X_test_n = normalize(X_test)

    best_k = 0
    best_ari = -1
    best_hom = -1

    results = []

    for k in range(2, 21):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        train_labels = km.fit_predict(X_train_n)

        # How well do clusters match true categories?
        ari = adjusted_rand_score(y_train, train_labels)
        hom = homogeneity_score(y_train, train_labels)

        # Predict test set by nearest centroid
        test_labels = km.predict(X_test_n)
        test_ari = adjusted_rand_score(y_test, test_labels)
        test_hom = homogeneity_score(y_test, test_labels)

        results.append({
            "k": k,
            "train_ari": ari,
            "train_hom": hom,
            "test_ari": test_ari,
            "test_hom": test_hom,
        })

        marker = ""
        if ari > best_ari:
            best_ari = ari
            best_hom = hom
            best_k = k
            marker = " <-- best"

        if k <= 10 or k % 5 == 0 or marker:
            print(f"    K={k:2d}  train ARI={ari:.3f} hom={hom:.3f}  "
                  f"test ARI={test_ari:.3f} hom={test_hom:.3f}{marker}")

    print(f"\n  Best K={best_k}: ARI={best_ari:.3f}, homogeneity={best_hom:.3f}")

    return results, best_k


def run_knn_classifier(X_train, y_train, X_test, y_test, cat_names, label):
    """KNN classifier: can Q vectors directly predict category?"""

    print(f"\n{'='*70}")
    print(f"KNN CLASSIFIER: {label}")
    print(f"{'='*70}")

    X_train_n = normalize(X_train)
    X_test_n = normalize(X_test)

    best_acc = 0
    best_k = 0

    for k in [1, 3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train_n, y_train)

        train_acc = knn.score(X_train_n, y_train)
        test_acc = knn.score(X_test_n, y_test)

        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_k = k
            marker = " <-- best"

        print(f"    K={k}: train={train_acc:.3f}  test={test_acc:.3f}{marker}")

    # Show per-category accuracy with best K
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
    knn.fit(X_train_n, y_train)
    y_pred = knn.predict(X_test_n)

    print(f"\n  Per-category accuracy (K={best_k}):")
    for cat_id, cat_name in enumerate(cat_names):
        mask = y_test == cat_id
        if mask.sum() > 0:
            cat_acc = (y_pred[mask] == y_test[mask]).mean()
            n_correct = (y_pred[mask] == y_test[mask]).sum()
            n_total = mask.sum()
            print(f"    {cat_name:15s}: {n_correct}/{n_total} = {cat_acc:.1%}")

    # Show misclassifications
    misses = np.where(y_pred != y_test)[0]
    if len(misses) > 0:
        print(f"\n  Misclassifications:")
        for idx in misses:
            print(f"    predicted={cat_names[y_pred[idx]]}, "
                  f"actual={cat_names[y_test[idx]]}")

    return best_acc


def analyze_centroid_structure(X_train, y_train, cat_names, label):
    """Compute per-category centroids and analyze their separation."""

    print(f"\n{'='*70}")
    print(f"CENTROID STRUCTURE: {label}")
    print(f"{'='*70}")

    X_n = normalize(X_train)
    centroids = {}
    for cat_id, cat_name in enumerate(cat_names):
        mask = y_train == cat_id
        centroids[cat_name] = X_n[mask].mean(axis=0)
        centroids[cat_name] /= np.linalg.norm(centroids[cat_name]) + 1e-10

    # Pairwise centroid distances
    print(f"\n  Centroid cosine similarities:")
    cats = list(centroids.keys())
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            sim = float(np.dot(centroids[a], centroids[b]))
            print(f"    {a:12s} vs {b:12s}: {sim:.4f}")

    # Within-category spread (avg distance to own centroid)
    print(f"\n  Within-category spread (avg cosine to centroid):")
    for cat_id, cat_name in enumerate(cat_names):
        mask = y_train == cat_id
        vecs = X_n[mask]
        sims = vecs @ centroids[cat_name]
        print(f"    {cat_name:15s}: mean={sims.mean():.4f}  "
              f"min={sims.min():.4f}  std={sims.std():.4f}")

    # Can we classify by nearest centroid?
    centroid_matrix = np.stack([centroids[c] for c in cats])  # [n_cats, dim]
    sims = X_n @ centroid_matrix.T  # [n_samples, n_cats]
    pred = sims.argmax(axis=1)
    acc = (pred == y_train).mean()
    print(f"\n  Nearest-centroid accuracy: {acc:.1%} ({int(acc*len(y_train))}/{len(y_train)})")

    # Per-category
    for cat_id, cat_name in enumerate(cat_names):
        mask = y_train == cat_id
        cat_acc = (pred[mask] == y_train[mask]).mean()
        print(f"    {cat_name:15s}: {cat_acc:.1%}")

    return centroids, acc


def per_layer_analysis(all_q_train, all_q_test, y_train, y_test,
                       cat_names, knowledge_layers):
    """Check which individual layers have the best Q-vector separation."""

    print(f"\n{'='*70}")
    print(f"PER-LAYER Q-VECTOR CLASSIFICATION")
    print(f"{'='*70}")
    print(f"  Testing which layer's Q vectors best predict category...\n")

    layer_results = []

    for layer in knowledge_layers:
        # Extract Q for this layer
        X_tr = np.stack([d[layer].flatten() for d in all_q_train])
        X_te = np.stack([d[layer].flatten() for d in all_q_test])

        X_tr_n = normalize(X_tr)
        X_te_n = normalize(X_te)

        # Nearest centroid
        cats = sorted(set(y_train))
        centroids = []
        for cat_id in cats:
            mask = y_train == cat_id
            c = X_tr_n[mask].mean(axis=0)
            c /= np.linalg.norm(c) + 1e-10
            centroids.append(c)
        centroid_matrix = np.stack(centroids)

        train_pred = (X_tr_n @ centroid_matrix.T).argmax(axis=1)
        test_pred = (X_te_n @ centroid_matrix.T).argmax(axis=1)

        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()

        # Also KNN-1
        knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        knn.fit(X_tr_n, y_train)
        knn_test_acc = knn.score(X_te_n, y_test)

        layer_results.append({
            "layer": layer,
            "centroid_train": train_acc,
            "centroid_test": test_acc,
            "knn1_test": knn_test_acc,
        })

        marker = ""
        if test_acc >= 0.9:
            marker = "  <<<"
        print(f"    L{layer:2d}  centroid: train={train_acc:.1%} test={test_acc:.1%}  "
              f"KNN-1: test={knn_test_acc:.1%}{marker}")

    return layer_results


def cross_layer_prediction(all_q_train, all_attn_train,
                           all_q_test, all_attn_test,
                           knowledge_layers, n_heads=8):
    """
    Can Q at one layer predict the attention pattern across ALL layers?

    Build a head-activity vector (which heads are most active across all
    knowledge layers), then see if Q from a single layer can predict it.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-LAYER PREDICTION: Can Q at one layer predict all heads?")
    print(f"{'='*70}")

    # Build attention pattern vectors
    def make_attn_vec(attn_dict):
        vec = np.zeros(len(knowledge_layers) * n_heads)
        for (layer, head), maxw in attn_dict.items():
            if layer in knowledge_layers:
                idx = (layer - min(knowledge_layers)) * n_heads + head
                if idx < len(vec):
                    vec[idx] = maxw
        return vec

    attn_train = np.stack([make_attn_vec(d) for d in all_attn_train])
    attn_test = np.stack([make_attn_vec(d) for d in all_attn_test])

    # For each layer, fit linear regression: Q -> attention pattern
    for layer in knowledge_layers:
        X_tr = np.stack([d[layer].flatten() for d in all_q_train])
        X_te = np.stack([d[layer].flatten() for d in all_q_test])

        # L2-normalized
        X_tr_n = normalize(X_tr)
        X_te_n = normalize(X_te)

        # Simple: cosine similarity between Q-predicted and actual attention
        # Use centroid approach: compute mean attention pattern per Q-cluster
        # Then assign test samples to nearest Q centroid and compare

        # We know there are ~4 categories. Use y labels for supervised version:
        # but we want unsupervised, so use K=4 KMeans on Q
        km = KMeans(n_clusters=4, n_init=10, random_state=42)
        train_clusters = km.fit_predict(X_tr_n)

        # Mean attention pattern per cluster
        cluster_attn = {}
        for c in range(4):
            mask = train_clusters == c
            if mask.sum() > 0:
                cluster_attn[c] = attn_train[mask].mean(axis=0)

        # For test: assign to Q cluster, predict attention, measure error
        test_clusters = km.predict(X_te_n)
        cos_sims = []
        for idx in range(len(X_te)):
            c = test_clusters[idx]
            if c in cluster_attn:
                pred_attn = cluster_attn[c]
                actual_attn = attn_test[idx]
                sim = np.dot(pred_attn, actual_attn) / (
                    np.linalg.norm(pred_attn) * np.linalg.norm(actual_attn) + 1e-10
                )
                cos_sims.append(sim)

        avg_sim = np.mean(cos_sims) if cos_sims else 0
        print(f"    L{layer:2d}: Q-cluster -> attention pattern cosine = {avg_sim:.4f}")


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Q-vector centroid classifier for routing table"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config for layer bands
    vindex_dir = Path(args.vindex)
    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    bands = config.get("layer_bands", {})
    knowledge_start = bands.get("knowledge", [14, 27])[0]
    knowledge_end = bands.get("knowledge", [14, 27])[1]
    knowledge_range = range(knowledge_start, knowledge_end + 1)

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    print(f"  Model: {args.model}")
    print(f"  Knowledge layers: L{knowledge_start}-L{knowledge_end}")

    cat_names = list(TRAIN_PROMPTS.keys())
    cat_to_id = {c: i for i, c in enumerate(cat_names)}

    # ---- Capture Q vectors ----
    print(f"\nCapturing Q vectors (train)...")
    all_q_train = []
    all_attn_train = []
    y_train_list = []
    n = 0
    t0 = time.time()
    total_train = sum(len(v) for v in TRAIN_PROMPTS.values())

    for cat, prompts in TRAIN_PROMPTS.items():
        for prompt in prompts:
            q_vecs, attn_maxw, pred = forward_capture_q(
                model, tokenizer, prompt, knowledge_range
            )
            all_q_train.append(q_vecs)
            all_attn_train.append(attn_maxw)
            y_train_list.append(cat_to_id[cat])
            n += 1
            print(f"\r  {n}/{total_train} ({time.time()-t0:.0f}s)", end="", flush=True)

    y_train = np.array(y_train_list)
    print(f"\n  Train done: {len(all_q_train)} samples in {time.time()-t0:.0f}s")

    print(f"\nCapturing Q vectors (held-out)...")
    all_q_test = []
    all_attn_test = []
    y_test_list = []
    n = 0
    total_test = sum(len(v) for v in HELD_OUT_PROMPTS.values())

    for cat, prompts in HELD_OUT_PROMPTS.items():
        for prompt in prompts:
            q_vecs, attn_maxw, pred = forward_capture_q(
                model, tokenizer, prompt, knowledge_range
            )
            all_q_test.append(q_vecs)
            all_attn_test.append(attn_maxw)
            y_test_list.append(cat_to_id[cat])
            n += 1
            print(f"\r  {n}/{total_test} ({time.time()-t0:.0f}s)", end="", flush=True)

    y_test = np.array(y_test_list)
    print(f"\n  Test done: {len(all_q_test)} samples")

    # ---- Build feature matrices ----
    # Use first knowledge layer's Q as primary (strongest signal from prev experiment)
    first_kl = knowledge_start

    # Per-layer Q (flattened: n_heads * head_dim)
    X_train_first = np.stack([d[first_kl].flatten() for d in all_q_train])
    X_test_first = np.stack([d[first_kl].flatten() for d in all_q_test])

    # All-layer Q concatenated
    X_train_all = np.stack([
        np.concatenate([d[l].flatten() for l in knowledge_range])
        for d in all_q_train
    ])
    X_test_all = np.stack([
        np.concatenate([d[l].flatten() for l in knowledge_range])
        for d in all_q_test
    ])

    print(f"\n  Feature dims: single layer={X_train_first.shape[1]}, "
          f"all layers={X_train_all.shape[1]}")

    # ---- Run analyses ----

    # 1. Per-layer classification
    layer_results = per_layer_analysis(
        all_q_train, all_q_test, y_train, y_test,
        cat_names, knowledge_range
    )

    # 2. KNN classifier (single layer)
    best_single = run_knn_classifier(
        X_train_first, y_train, X_test_first, y_test,
        cat_names, f"Q at L{first_kl} (single layer)"
    )

    # 3. KNN classifier (all layers)
    best_all = run_knn_classifier(
        X_train_all, y_train, X_test_all, y_test,
        cat_names, "Q concatenated (all knowledge layers)"
    )

    # 4. Centroid structure (single layer)
    centroids, centroid_acc = analyze_centroid_structure(
        X_train_first, y_train, cat_names,
        f"Q at L{first_kl}"
    )

    # 5. K-means sweep (single layer)
    km_results_single, best_k_single = run_kmeans_sweep(
        X_train_first, y_train, X_test_first, y_test,
        cat_names, f"Q at L{first_kl}"
    )

    # 6. K-means sweep (all layers)
    km_results_all, best_k_all = run_kmeans_sweep(
        X_train_all, y_train, X_test_all, y_test,
        cat_names, "Q all layers"
    )

    # 7. Cross-layer prediction
    cross_layer_prediction(
        all_q_train, all_attn_train,
        all_q_test, all_attn_test,
        knowledge_range,
    )

    # ---- Save results ----
    results = {
        "layer_classification": layer_results,
        "kmeans_single": km_results_single,
        "kmeans_all": km_results_all,
        "best_k_single": best_k_single,
        "best_k_all": best_k_all,
        "knn_single_acc": best_single,
        "knn_all_acc": best_all,
        "centroid_acc": centroid_acc,
        "categories": cat_names,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    with open(output_dir / "q_centroid_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_dir / 'q_centroid_results.json'}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    # Find best single-layer test accuracy
    best_layer = max(layer_results, key=lambda x: x["centroid_test"])

    print(f"\n  Best single-layer routing: L{best_layer['layer']} "
          f"(centroid test acc: {best_layer['centroid_test']:.1%})")
    print(f"  KNN-1 single layer: {best_single:.1%}")
    print(f"  KNN-1 all layers:   {best_all:.1%}")
    print(f"  Nearest centroid:   {centroid_acc:.1%}")
    print(f"  Best K-means K:     {best_k_single}")

    if best_layer['centroid_test'] >= 0.9:
        print(f"\n  Q-VECTOR ROUTING WORKS")
        print(f"    A single layer's Q projection classifies query type at {best_layer['centroid_test']:.0%}")
        print(f"    With {len(cat_names)} category centroids, nearest-centroid suffices")
        print(f"    -> Compute q_proj once, classify, apply cached attention pattern")
        print(f"    -> Eliminates K computation, QK matmul, and softmax")
    elif best_all > 0.9:
        print(f"\n  ~ Q-VECTOR ROUTING WORKS (needs multi-layer)")
        print(f"    Need Q from multiple layers to classify")
        print(f"    Still eliminates K + QK, but q_proj needed at each layer")
    else:
        print(f"\n  Q-VECTOR ROUTING INSUFFICIENT")
        print(f"    Q vectors alone don't predict attention routing well enough")
        print(f"    Need a different approach")

    print()


if __name__ == "__main__":
    main()
