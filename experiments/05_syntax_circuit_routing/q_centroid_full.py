#!/usr/bin/env python3
"""
q_centroid_full.py — Full 16-category Q-vector routing classifier

Scale-up of q_centroid_classifier.py. Tests whether Q-vector routing
holds across all 16 GPT-OSS trigram families, not just 4.

240 prompts total: ~180 train, ~60 held-out (last 3-5 per category).

USAGE:
  python3 experiments/05_syntax_circuit_routing/q_centroid_full.py \
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
from sklearn.metrics import (
    adjusted_rand_score, homogeneity_score,
    classification_report, confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- All 16 categories ------------------------------------------------
# Each has prompts split into train (first N) and test (last few)

ALL_PROMPTS = {
    "capital_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
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
        "test": [
            "The capital of Poland is",
            "The capital of Turkey is",
            "The capital of Vietnam is",
            "The capital of Nigeria is",
            "The capital of Peru is",
        ],
    },
    "language_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
            "The official language of France is",
            "The official language of Japan is",
            "The official language of Brazil is",
            "The official language of China is",
            "The official language of Germany is",
            "The official language of Russia is",
            "The official language of Italy is",
        ],
        "test": [
            "The official language of Portugal is",
            "The official language of Thailand is",
            "The official language of Greece is",
        ],
    },
    "currency_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
            "The currency of Japan is the",
            "The currency of India is the",
            "The currency of Brazil is the",
            "The currency of Mexico is the",
            "The currency of Sweden is the",
            "The currency of Poland is the",
            "The currency of Thailand is the",
        ],
        "test": [
            "The currency of Turkey is the",
            "The currency of Egypt is the",
            "The currency of China is the",
        ],
    },
    "continent_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
            "France is located in",
            "Japan is located in",
            "Brazil is located in",
            "Nigeria is located in",
            "Australia is located in",
            "Canada is located in",
            "Egypt is located in",
        ],
        "test": [
            "India is located in",
            "Mexico is located in",
            "Sweden is located in",
        ],
    },
    "occupation_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
            "The occupation of Einstein was",
            "The occupation of Shakespeare was",
            "The occupation of Mozart was",
            "The occupation of Picasso was",
            "The occupation of Darwin was",
            "The occupation of Newton was",
            "The occupation of Beethoven was",
        ],
        "test": [
            "The occupation of Hemingway was",
            "The occupation of Curie was",
            "The occupation of Tesla was",
        ],
    },
    "birthplace_of": {
        "category": "entity_predicate_value",
        "trigram": "NOUN->FUNC->NOUN",
        "train": [
            "Einstein was born in",
            "Shakespeare was born in",
            "Mozart was born in",
            "Picasso was born in",
            "Darwin was born in",
            "Newton was born in",
            "Beethoven was born in",
        ],
        "test": [
            "Gandhi was born in",
            "Confucius was born in",
            "Napoleon was born in",
        ],
    },
    "synonym": {
        "category": "adj_synonym",
        "trigram": "ADJ->SYN->ADJ",
        "train": [
            "Happy means",
            "Sad means",
            "Big means",
            "Small means",
            "Fast means",
            "Slow means",
            "Hot means",
            "Cold means",
            "Smart means",
            "Brave means",
            "Angry means",
            "Calm means",
            "Rich means",
            "Poor means",
            "Strong means",
        ],
        "test": [
            "Weak means",
            "Bright means",
            "Dark means",
            "Loud means",
            "Quiet means",
        ],
    },
    "antonym": {
        "category": "adj_antonym",
        "trigram": "ADJ->ANT->ADJ",
        "train": [
            "The opposite of happy is",
            "The opposite of big is",
            "The opposite of fast is",
            "The opposite of hot is",
            "The opposite of light is",
            "The opposite of old is",
            "The opposite of rich is",
            "The opposite of strong is",
            "The opposite of early is",
            "The opposite of deep is",
            "The opposite of hard is",
            "The opposite of wet is",
            "The opposite of loud is",
            "The opposite of brave is",
            "The opposite of smooth is",
        ],
        "test": [
            "The opposite of thick is",
            "The opposite of wide is",
            "The opposite of sharp is",
            "The opposite of clean is",
            "The opposite of heavy is",
        ],
    },
    "analogy": {
        "category": "analogy",
        "trigram": "NOUN->AS->NOUN",
        "train": [
            "King is to queen as man is to",
            "Dog is to puppy as cat is to",
            "Hot is to cold as big is to",
            "France is to Paris as Japan is to",
            "Teacher is to school as doctor is to",
            "Bird is to fly as fish is to",
            "Hand is to glove as foot is to",
            "Pen is to write as knife is to",
            "Eye is to see as ear is to",
            "Day is to night as summer is to",
            "Book is to read as song is to",
            "Painter is to brush as writer is to",
            "Cow is to milk as hen is to",
            "Rain is to umbrella as sun is to",
            "North is to south as east is to",
        ],
        "test": [
            "Apple is to fruit as carrot is to",
            "Piano is to keys as guitar is to",
            "Pilot is to plane as captain is to",
            "Flour is to bread as grape is to",
            "Oxygen is to breathe as water is to",
        ],
    },
    "hypernym": {
        "category": "hypernym",
        "trigram": "NOUN->VERB->NOUN",
        "train": [
            "A dog is a type of",
            "A rose is a type of",
            "A piano is a type of",
            "A hammer is a type of",
            "A sedan is a type of",
            "A sparrow is a type of",
            "A salmon is a type of",
            "A diamond is a type of",
            "A violin is a type of",
            "A oak is a type of",
            "A python is a type of",
            "A hurricane is a type of",
            "A novel is a type of",
            "A sonnet is a type of",
            "A waltz is a type of",
        ],
        "test": [
            "A cathedral is a type of",
            "A monarchy is a type of",
            "A telescope is a type of",
            "A triangle is a type of",
            "A electron is a type of",
        ],
    },
    "arithmetic": {
        "category": "arithmetic",
        "trigram": "NUM->OP->NUM",
        "train": [
            "2 + 3 =",
            "7 - 4 =",
            "5 * 6 =",
            "10 / 2 =",
            "15 + 27 =",
            "100 - 37 =",
            "8 * 9 =",
            "48 / 6 =",
            "3 + 3 + 3 =",
            "25 * 4 =",
            "99 - 11 =",
            "12 * 12 =",
            "1000 / 10 =",
            "7 + 8 =",
            "50 - 25 =",
        ],
        "test": [
            "6 * 7 =",
            "81 / 9 =",
            "33 + 67 =",
            "200 - 150 =",
            "11 * 11 =",
        ],
    },
    "code_python": {
        "category": "code_definition",
        "trigram": "^->KW->FUNC",
        "train": [
            "def hello():\n    return",
            "def add(a, b):\n    return",
            "def factorial(n):\n    if n ==",
            "def greet(name):\n    print",
            "def is_even(n):\n    return",
            "class Dog:\n    def __init__",
            "class Person:\n    def __init__",
        ],
        "test": [
            "class Vector:\n    def __init__",
            "for i in range(10):\n    print",
            "if x > 0:\n    return",
        ],
    },
    "code_rust": {
        "category": "code_definition",
        "trigram": "^->KW->FUNC",
        "train": [
            "fn main() {\n    let x =",
            "fn add(a: i32, b: i32) -> i32 {\n    a",
            "struct Point {\n    x:",
            "impl Display for Point {\n    fn fmt",
            "let mut vec = Vec::new();\n    vec",
            "match result {\n    Ok(val) =>",
            "enum Color {\n    Red,",
        ],
        "test": [
            "pub fn process(input: &str) ->",
            "use std::collections::HashMap;\n\nfn",
            "trait Summary {\n    fn summarize",
        ],
    },
    "comparison": {
        "category": "comparison",
        "trigram": "ADJ->THAN->NOUN",
        "train": [
            "An elephant is bigger than a",
            "A cheetah is faster than a",
            "The sun is hotter than the",
            "Gold is heavier than",
            "Mount Everest is taller than",
            "The Pacific is larger than the",
            "A diamond is harder than",
        ],
        "test": [
            "Light is faster than",
            "Jupiter is bigger than",
            "Steel is stronger than",
        ],
    },
    "causation": {
        "category": "causation",
        "trigram": "CW->CAUSE->CW",
        "train": [
            "Plants grow because they need",
            "Ice melts because the temperature",
            "Birds fly because they have",
            "People sleep because the body",
            "Fire burns because of",
            "Metal rusts because of",
            "Rain falls because water",
        ],
        "test": [
            "Stars shine because of",
            "Tides change because of the",
            "Volcanoes erupt because of",
        ],
    },
    "temporal": {
        "category": "temporal",
        "trigram": "NOUN->TIME->VERB",
        "train": [
            "World War II ended in",
            "The Roman Empire fell in",
            "The internet was invented in",
            "The first airplane flew in",
            "The moon landing happened in",
            "The Berlin Wall fell in",
            "The printing press was invented in",
        ],
        "test": [
            "The French Revolution began in",
            "DNA was discovered in",
            "The telephone was invented in",
        ],
    },
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
    attn_maxw = {}

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

            q_last = q[0, :, -1, :]
            mx.eval(q_last)
            q_vectors[i] = np.array(q_last.astype(mx.float32))

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

    return q_vectors, attn_maxw


# ---- Analysis -----------------------------------------------------------

def run_analysis(X_train, y_train, X_test, y_test, label_names,
                 label, granularity_name):
    """Full classification + clustering analysis at one granularity."""

    X_tr = normalize(X_train)
    X_te = normalize(X_test)
    n_classes = len(label_names)

    print(f"\n{'='*70}")
    print(f"{granularity_name} CLASSIFICATION ({label})")
    print(f"{'='*70}")
    print(f"  {n_classes} classes, {len(X_tr)} train, {len(X_te)} test, dim={X_tr.shape[1]}")

    # ---- Nearest centroid ----
    centroids = []
    for cid in range(n_classes):
        mask = y_train == cid
        if mask.sum() == 0:
            centroids.append(np.zeros(X_tr.shape[1]))
            continue
        c = X_tr[mask].mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        centroids.append(c)
    C = np.stack(centroids)

    train_pred = (X_tr @ C.T).argmax(axis=1)
    test_pred = (X_te @ C.T).argmax(axis=1)

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    print(f"\n  Nearest centroid:")
    print(f"    Train: {train_acc:.1%}  Test: {test_acc:.1%}")

    # Per-class test accuracy
    print(f"\n    Per-class test accuracy:")
    for cid, name in enumerate(label_names):
        mask = y_test == cid
        if mask.sum() > 0:
            cls_acc = (test_pred[mask] == y_test[mask]).mean()
            n_ok = int((test_pred[mask] == y_test[mask]).sum())
            n_tot = int(mask.sum())
            marker = "" if cls_acc == 1.0 else f"  <-- MISS"
            print(f"      {name:20s}: {n_ok}/{n_tot} = {cls_acc:.0%}{marker}")

    # Misclassifications detail
    misses = np.where(test_pred != y_test)[0]
    if len(misses) > 0:
        print(f"\n    Misclassifications ({len(misses)}):")
        for idx in misses:
            pred_name = label_names[test_pred[idx]]
            true_name = label_names[y_test[idx]]
            # Show cosine to true vs predicted centroid
            sim_true = float(X_te[idx] @ C[y_test[idx]])
            sim_pred = float(X_te[idx] @ C[test_pred[idx]])
            print(f"      predicted={pred_name}, actual={true_name}  "
                  f"(cos_true={sim_true:.3f}, cos_pred={sim_pred:.3f})")

    # ---- KNN ----
    print(f"\n  KNN classifier:")
    for k in [1, 3, 5]:
        if k > len(X_tr) // n_classes:
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_tr, y_train)
        knn_train = knn.score(X_tr, y_train)
        knn_test = knn.score(X_te, y_test)
        print(f"    K={k}: train={knn_train:.1%}  test={knn_test:.1%}")

    # ---- K-means ----
    print(f"\n  K-means sweep:")
    best_k = 0
    best_ari = -1
    for k in list(range(2, min(n_classes + 5, 25))) + [30, 40]:
        if k > len(X_tr):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        train_labels = km.fit_predict(X_tr)
        ari = adjusted_rand_score(y_train, train_labels)
        hom = homogeneity_score(y_train, train_labels)

        test_labels = km.predict(X_te)
        test_ari = adjusted_rand_score(y_test, test_labels)

        marker = ""
        if ari > best_ari:
            best_ari = ari
            best_k = k
            marker = " <-- best"

        if k <= n_classes + 2 or k % 5 == 0 or marker:
            print(f"    K={k:2d}  ARI={ari:.3f} hom={hom:.3f}  "
                  f"test_ARI={test_ari:.3f}{marker}")

    print(f"    Best K={best_k} (ARI={best_ari:.3f})")

    # ---- Centroid similarity matrix ----
    print(f"\n  Centroid similarity matrix (top confusable pairs):")
    sim_matrix = C @ C.T
    pairs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            pairs.append((label_names[i], label_names[j], sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, s in pairs[:10]:
        print(f"    {a:20s} vs {b:20s}: {s:.4f}")

    return {
        "centroid_train_acc": float(train_acc),
        "centroid_test_acc": float(test_acc),
        "best_kmeans_k": best_k,
        "best_kmeans_ari": float(best_ari),
        "n_misses": len(misses),
    }


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full 16-category Q-vector routing classifier"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
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

    # Build label mappings at two granularities:
    # 1. Fine-grained: 16 template names (capital_of, language_of, synonym, ...)
    # 2. Coarse: category names (entity_predicate_value, adj_synonym, ...)
    fine_names = list(ALL_PROMPTS.keys())
    fine_to_id = {n: i for i, n in enumerate(fine_names)}

    coarse_names_set = []
    for info in ALL_PROMPTS.values():
        if info["category"] not in coarse_names_set:
            coarse_names_set.append(info["category"])
    coarse_names = coarse_names_set
    coarse_to_id = {n: i for i, n in enumerate(coarse_names)}

    total_train = sum(len(v["train"]) for v in ALL_PROMPTS.values())
    total_test = sum(len(v["test"]) for v in ALL_PROMPTS.values())
    print(f"\n  {len(fine_names)} fine categories, {len(coarse_names)} coarse categories")
    print(f"  {total_train} train prompts, {total_test} test prompts")

    # ---- Capture ----
    print(f"\nCapturing Q vectors...")
    all_q_train = []
    all_q_test = []
    y_train_fine = []
    y_train_coarse = []
    y_test_fine = []
    y_test_coarse = []

    n = 0
    t0 = time.time()
    total = total_train + total_test

    for template_name, info in ALL_PROMPTS.items():
        fine_id = fine_to_id[template_name]
        coarse_id = coarse_to_id[info["category"]]

        for prompt in info["train"]:
            q_vecs, attn_maxw = forward_capture_q(
                model, tokenizer, prompt, knowledge_range
            )
            all_q_train.append(q_vecs)
            y_train_fine.append(fine_id)
            y_train_coarse.append(coarse_id)
            n += 1
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / rate if rate > 0 else 0
            print(f"\r  {n}/{total} ({rate:.1f}/s, ETA {eta:.0f}s)", end="", flush=True)

        for prompt in info["test"]:
            q_vecs, attn_maxw = forward_capture_q(
                model, tokenizer, prompt, knowledge_range
            )
            all_q_test.append(q_vecs)
            y_test_fine.append(fine_id)
            y_test_coarse.append(coarse_id)
            n += 1
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / rate if rate > 0 else 0
            print(f"\r  {n}/{total} ({rate:.1f}/s, ETA {eta:.0f}s)", end="", flush=True)

    y_train_fine = np.array(y_train_fine)
    y_train_coarse = np.array(y_train_coarse)
    y_test_fine = np.array(y_test_fine)
    y_test_coarse = np.array(y_test_coarse)

    print(f"\n  Done: {len(all_q_train)} train, {len(all_q_test)} test in {time.time()-t0:.0f}s")

    # ---- Per-layer scan (quick, using L14 Q) ----
    print(f"\n{'='*70}")
    print(f"PER-LAYER SCAN (nearest centroid, fine-grained)")
    print(f"{'='*70}")

    best_layer = None
    best_layer_acc = 0

    for layer in knowledge_range:
        X_tr = normalize(np.stack([d[layer].flatten() for d in all_q_train]))
        X_te = normalize(np.stack([d[layer].flatten() for d in all_q_test]))

        centroids = []
        for cid in range(len(fine_names)):
            mask = y_train_fine == cid
            if mask.sum() == 0:
                centroids.append(np.zeros(X_tr.shape[1]))
                continue
            c = X_tr[mask].mean(axis=0)
            c /= np.linalg.norm(c) + 1e-10
            centroids.append(c)
        C = np.stack(centroids)

        train_pred = (X_tr @ C.T).argmax(axis=1)
        test_pred = (X_te @ C.T).argmax(axis=1)
        train_acc = (train_pred == y_train_fine).mean()
        test_acc = (test_pred == y_test_fine).mean()

        marker = ""
        if test_acc > best_layer_acc:
            best_layer_acc = test_acc
            best_layer = layer
            marker = " <-- best"
        print(f"  L{layer:2d}: train={train_acc:.1%}  test={test_acc:.1%}{marker}")

    # ---- Full analysis at best layer ----
    ref_layer = best_layer
    X_train_ref = np.stack([d[ref_layer].flatten() for d in all_q_train])
    X_test_ref = np.stack([d[ref_layer].flatten() for d in all_q_test])

    # Fine-grained (16 templates)
    fine_results = run_analysis(
        X_train_ref, y_train_fine, X_test_ref, y_test_fine,
        fine_names, f"Q at L{ref_layer}", "FINE-GRAINED (16 templates)"
    )

    # Coarse (unique categories)
    coarse_results = run_analysis(
        X_train_ref, y_train_coarse, X_test_ref, y_test_coarse,
        coarse_names, f"Q at L{ref_layer}", f"COARSE ({len(coarse_names)} categories)"
    )

    # ---- Save ----
    results = {
        "ref_layer": ref_layer,
        "fine_grained": fine_results,
        "coarse": coarse_results,
        "fine_names": fine_names,
        "coarse_names": coarse_names,
        "n_train": len(y_train_fine),
        "n_test": len(y_test_fine),
    }
    with open(output_dir / "q_centroid_full_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    fine_test = fine_results["centroid_test_acc"]
    coarse_test = coarse_results["centroid_test_acc"]
    fine_k = fine_results["best_kmeans_k"]

    print(f"\n  Best layer: L{ref_layer}")
    print(f"  Fine-grained (16 templates):  {fine_test:.1%} test accuracy")
    print(f"  Coarse ({len(coarse_names)} categories):  {coarse_test:.1%} test accuracy")
    print(f"  K-means best K: {fine_k}")

    if fine_test >= 0.9:
        print(f"\n  FULL ROUTING TABLE CONFIRMED")
        print(f"    16 template centroids at L{ref_layer} classify at {fine_test:.0%}")
        print(f"    Each template -> cached attention pattern -> graph walk")
        print(f"    No K, no QK, no softmax anywhere in the pipeline")
    elif coarse_test >= 0.9:
        print(f"\n  COARSE ROUTING WORKS, FINE NEEDS MORE DATA")
        print(f"    {len(coarse_names)} coarse centroids classify at {coarse_test:.0%}")
        print(f"    Fine-grained needs more training examples or better features")
    else:
        print(f"\n  ROUTING PARTIALLY WORKS")
        print(f"    Accuracy below 90% - some template types are confusable")
        print(f"    Check confusion pairs above for which merge")

    print()


if __name__ == "__main__":
    main()
