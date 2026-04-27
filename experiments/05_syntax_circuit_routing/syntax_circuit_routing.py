#!/usr/bin/env python3
"""
syntax_circuit_routing.py — Syntax->Circuit Routing Experiment

HYPOTHESIS: Syntax features (L0-12) predict which attention circuits (L13-26)
activate, analogous to how trigram types predicted MoE expert routing in GPT-OSS.

APPROACH:
  1. Run diverse prompts through model, capture:
     - L0-12 gate activations (syntax features)
     - L13-26 attention patterns (which heads are active)
  2. Map attention patterns to known circuits (from circuit-discover)
  3. Build co-occurrence matrix: syntax_features x circuits
  4. Measure sparsity -- sparse = routing table exists, dense = compositional

USAGE:
  python3 experiments/05_syntax_circuit_routing/syntax_circuit_routing.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex \
      --circuits output/gemma3-4b-f16.vindex/circuits.json \
      --output output/syntax_circuit_routing/

EXPECTS:
  - vindex with feature_labels.json (from probe_mlx.py runs)
  - circuits.json from circuit-discover (192 circuits)
  - MLX model access
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import mlx.core as mx
import mlx.nn as nn


# ---- Prompt Templates -------------------------------------------------
# 240 prompts across 16 template families covering GPT-OSS trigram types
# + knowledge relations

PROMPT_TEMPLATES = {
    # -- NOUN->RELATION->NOUN (factual knowledge) --
    "capital_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
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
            "The capital of Poland is",
            "The capital of Turkey is",
            "The capital of Vietnam is",
            "The capital of Nigeria is",
            "The capital of Peru is",
        ],
    },
    "language_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
            "The official language of France is",
            "The official language of Japan is",
            "The official language of Brazil is",
            "The official language of China is",
            "The official language of Germany is",
            "The official language of Russia is",
            "The official language of Italy is",
            "The official language of Portugal is",
            "The official language of Thailand is",
            "The official language of Greece is",
        ],
    },
    "currency_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
            "The currency of Japan is the",
            "The currency of India is the",
            "The currency of Brazil is the",
            "The currency of Mexico is the",
            "The currency of Sweden is the",
            "The currency of Poland is the",
            "The currency of Thailand is the",
            "The currency of Turkey is the",
            "The currency of Egypt is the",
            "The currency of China is the",
        ],
    },
    "continent_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
            "France is located in",
            "Japan is located in",
            "Brazil is located in",
            "Nigeria is located in",
            "Australia is located in",
            "Canada is located in",
            "Egypt is located in",
            "India is located in",
            "Mexico is located in",
            "Sweden is located in",
        ],
    },

    # -- PERSON->RELATION->VALUE (biographical) --
    "occupation_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
            "The occupation of Einstein was",
            "The occupation of Shakespeare was",
            "The occupation of Mozart was",
            "The occupation of Picasso was",
            "The occupation of Darwin was",
            "The occupation of Newton was",
            "The occupation of Beethoven was",
            "The occupation of Hemingway was",
            "The occupation of Curie was",
            "The occupation of Tesla was",
        ],
    },
    "birthplace_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->NOUN",
        "prompts": [
            "Einstein was born in",
            "Shakespeare was born in",
            "Mozart was born in",
            "Picasso was born in",
            "Darwin was born in",
            "Newton was born in",
            "Beethoven was born in",
            "Gandhi was born in",
            "Confucius was born in",
            "Napoleon was born in",
        ],
    },
    "nationality_of": {
        "category": "entity_predicate_value",
        "trigram_type": "NOUN->FUNC->ADJ",
        "prompts": [
            "Einstein was",
            "Shakespeare was",
            "Mozart was",
            "Picasso was",
            "Confucius was",
            "Gandhi was",
            "Napoleon was",
            "Beethoven was",
            "Tesla was",
            "Da Vinci was",
        ],
    },

    # -- ADJ->SYN->ADJ (synonym pattern from GPT-OSS) --
    "synonym": {
        "category": "adj_synonym",
        "trigram_type": "ADJ->SYN->ADJ",
        "prompts": [
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
            "Weak means",
            "Bright means",
            "Dark means",
            "Loud means",
            "Quiet means",
        ],
    },

    # -- ADJ->ANT->ADJ (antonym pattern from GPT-OSS) --
    "antonym": {
        "category": "adj_antonym",
        "trigram_type": "ADJ->ANT->ADJ",
        "prompts": [
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
            "The opposite of thick is",
            "The opposite of wide is",
            "The opposite of sharp is",
            "The opposite of clean is",
            "The opposite of heavy is",
        ],
    },

    # -- NOUN->AS->NOUN (analogy pattern from GPT-OSS) --
    "analogy": {
        "category": "analogy",
        "trigram_type": "NOUN->AS->NOUN",
        "prompts": [
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
            "Apple is to fruit as carrot is to",
            "Piano is to keys as guitar is to",
            "Pilot is to plane as captain is to",
            "Flour is to bread as grape is to",
            "Oxygen is to breathe as water is to",
        ],
    },

    # -- NOUN->VERB->NOUN (hypernym / is-a) --
    "hypernym": {
        "category": "hypernym",
        "trigram_type": "NOUN->VERB->NOUN",
        "prompts": [
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
            "A cathedral is a type of",
            "A monarchy is a type of",
            "A telescope is a type of",
            "A triangle is a type of",
            "A electron is a type of",
        ],
    },

    # -- NUM->OP->NUM (arithmetic from GPT-OSS) --
    "arithmetic": {
        "category": "arithmetic",
        "trigram_type": "NUM->OP->NUM",
        "prompts": [
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
            "6 * 7 =",
            "81 / 9 =",
            "33 + 67 =",
            "200 - 150 =",
            "11 * 11 =",
        ],
    },

    # -- CODE: ^->KW->CW (code structure from syntax probes) --
    "code_python_def": {
        "category": "code_definition",
        "trigram_type": "^->KW->FUNC",
        "prompts": [
            "def hello():\n    return",
            "def add(a, b):\n    return",
            "def factorial(n):\n    if n ==",
            "def greet(name):\n    print",
            "def is_even(n):\n    return",
            "class Dog:\n    def __init__",
            "class Person:\n    def __init__",
            "class Vector:\n    def __init__",
            "for i in range(10):\n    print",
            "if x > 0:\n    return",
        ],
    },
    "code_rust": {
        "category": "code_definition",
        "trigram_type": "^->KW->FUNC",
        "prompts": [
            "fn main() {\n    let x =",
            "fn add(a: i32, b: i32) -> i32 {\n    a",
            "struct Point {\n    x:",
            "impl Display for Point {\n    fn fmt",
            "let mut vec = Vec::new();\n    vec",
            "match result {\n    Ok(val) =>",
            "enum Color {\n    Red,",
            "pub fn process(input: &str) ->",
            "use std::collections::HashMap;\n\nfn",
            "trait Summary {\n    fn summarize",
        ],
    },

    # -- COMPARISON: ADJ->THAN->NOUN --
    "comparison": {
        "category": "comparison",
        "trigram_type": "ADJ->THAN->NOUN",
        "prompts": [
            "An elephant is bigger than a",
            "A cheetah is faster than a",
            "The sun is hotter than the",
            "Gold is heavier than",
            "Mount Everest is taller than",
            "The Pacific is larger than the",
            "A diamond is harder than",
            "Light is faster than",
            "Jupiter is bigger than",
            "Steel is stronger than",
        ],
    },

    # -- CAUSATION: CW->CAUSE->CW --
    "causation": {
        "category": "causation",
        "trigram_type": "CW->CAUSE->CW",
        "prompts": [
            "Plants grow because they need",
            "Ice melts because the temperature",
            "Birds fly because they have",
            "People sleep because the body",
            "Fire burns because of",
            "Metal rusts because of",
            "Rain falls because water",
            "Stars shine because of",
            "Tides change because of the",
            "Volcanoes erupt because of",
        ],
    },

    # -- TEMPORAL: NOUN->TIME->VERB --
    "temporal": {
        "category": "temporal",
        "trigram_type": "NOUN->TIME->VERB",
        "prompts": [
            "World War II ended in",
            "The Roman Empire fell in",
            "The internet was invented in",
            "The first airplane flew in",
            "The moon landing happened in",
            "The Berlin Wall fell in",
            "The printing press was invented in",
            "The French Revolution began in",
            "DNA was discovered in",
            "The telephone was invented in",
        ],
    },
}


# ---- Vindex / Model Loading -------------------------------------------

def load_vindex(vindex_path):
    """Load vindex gates + feature labels."""
    vindex_path = Path(vindex_path)

    with open(vindex_path / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    n_layers = config["num_layers"]

    # Gate vectors (per-layer offset loading)
    gate_path = vindex_path / "gate_vectors.bin"
    gate_file_size = gate_path.stat().st_size
    total_elements = sum(li["num_features"] for li in config["layers"]) * hidden_size

    if gate_file_size == total_elements * 2:
        gate_dtype, bpe = np.float16, 2
    else:
        gate_dtype, bpe = np.float32, 4

    gate_raw = np.fromfile(gate_path, dtype=gate_dtype)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // bpe
        chunk = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)
        gates[layer] = chunk.astype(np.float32)

    # Feature labels
    labels = {}
    labels_path = vindex_path / "feature_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    print(f"  Vindex: {n_layers}L, {hidden_size}d, {len(labels)} labels")
    return config, gates, labels


def load_circuits(circuits_path):
    """Load circuit communities from circuit-discover output."""
    with open(circuits_path) as f:
        circuits = json.load(f)
    print(f"  Circuits: {len(circuits)} communities loaded")
    return circuits


def find_model_parts(model):
    """Auto-detect model internals (handles Gemma 3, Llama, etc)."""
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


def load_model(model_name):
    """Load MLX model."""
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_name)
    print(f"  Model loaded: {model_name}")
    return model, tokenizer


# ---- Core Capture ------------------------------------------------------

def forward_with_attention(model, tokenizer, prompt, syntax_layers, knowledge_layers):
    """
    Layer-by-layer forward pass capturing residuals + attention weights.

    Handles Gemma 3 GQA: 8 Q heads, 4 KV heads (repeats=2),
    head_dim=256, q_norm/k_norm, RoPE.
    """
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    h = embed_fn(input_ids)
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    residuals = {}
    attn_info = {}
    attn_patterns = {}  # (layer, head) -> [seq_len] last-token attention

    for i, layer in enumerate(layers):
        if i in knowledge_layers:
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

            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)

            weights = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                weights = weights + mask
            weights = mx.softmax(weights, axis=-1)

            # Capture per-head attention stats
            weights_np = np.array(weights[0, :, -1, :].astype(mx.float32))
            mx.eval(weights)

            for head_idx in range(n_heads):
                w = weights_np[head_idx]
                entropy = -np.sum(w * np.log(w + 1e-10))
                attn_info[(i, head_idx)] = {
                    "entropy": float(entropy),
                    "max_w": float(np.max(w)),
                    "argmax": int(np.argmax(w)),
                }
                attn_patterns[(i, head_idx)] = w.tolist()

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

        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return residuals, attn_info, attn_patterns, pred_tok


def capture_one(model, tokenizer, prompt, gates, labels, config,
                syntax_range, knowledge_range, gate_threshold=3.0):
    """Run one prompt, return syntax features + attention patterns."""

    residuals, attn_info, attn_patterns, pred_tok = forward_with_attention(
        model, tokenizer, prompt,
        syntax_layers=syntax_range,
        knowledge_layers=knowledge_range,
    )

    # Syntax features (L0-12)
    syntax_hits = []
    for layer in syntax_range:
        if layer not in residuals or layer not in gates:
            continue
        res = residuals[layer]
        acts = gates[layer] @ res

        top_idx = np.argsort(np.abs(acts))[-5:][::-1]
        for fi in top_idx:
            val = float(acts[fi])
            if abs(val) < gate_threshold:
                continue
            key = f"L{layer}_F{fi}"
            label_info = labels.get(key, {})
            rel = label_info.get("relation", "-") if isinstance(label_info, dict) else str(label_info)
            syntax_hits.append((layer, int(fi), val, rel))

    syntax_hits.sort(key=lambda x: abs(x[2]), reverse=True)
    syntax_hits = syntax_hits[:10]

    # Attention head patterns
    attn_hits = []
    for (layer, head), info in sorted(attn_info.items()):
        attn_hits.append((layer, head, info["entropy"], info["max_w"]))
    attn_hits.sort(key=lambda x: x[3], reverse=True)
    attn_hits = attn_hits[:10]

    # Attention fingerprints (for circuit matching)
    fingerprints = {}
    for (layer, head), pattern in attn_patterns.items():
        active_positions = [
            i for i, w in enumerate(pattern) if w > 0.1
        ]
        fingerprints[(layer, head)] = tuple(active_positions)

    return {
        "syntax": syntax_hits,
        "attention": attn_hits,
        "fingerprints": fingerprints,
        "prediction": pred_tok,
    }


# ---- Circuit Matching --------------------------------------------------

def match_to_circuits(fingerprints, circuits, threshold=0.8):
    """
    Determine which circuits are active based on attention fingerprints.

    A circuit is "active" if enough of its constituent heads show
    non-trivial attention patterns.
    """
    active_circuits = []

    for circuit in circuits:
        circuit_id = circuit.get("id", circuits.index(circuit))
        heads = circuit.get("heads", [])

        if not heads:
            continue

        n_active = 0
        for layer, head in heads:
            fp = fingerprints.get((layer, head))
            if fp and len(fp) > 0:
                n_active += 1

        score = n_active / len(heads) if heads else 0
        if score > threshold:
            active_circuits.append((circuit_id, score))

    return active_circuits


# ---- Co-occurrence Matrix -----------------------------------------------

def build_cooccurrence_matrix(results):
    """
    Build syntax_feature x circuit co-occurrence matrix.

    If sparse/block-diagonal: syntax features predict circuits.
    If dense: routing is compositional.
    """
    all_syntax = set()
    all_circuits = set()

    for r in results:
        for layer, feat, val, rel in r["syntax"]:
            all_syntax.add(f"L{layer}_F{feat}")
        for cid, score in r.get("active_circuits", []):
            all_circuits.add(cid)

    syntax_list = sorted(all_syntax)
    circuit_list = sorted(all_circuits)
    syntax_idx = {f: i for i, f in enumerate(syntax_list)}
    circuit_idx = {c: i for i, c in enumerate(circuit_list)}

    n_s = len(syntax_list)
    n_c = len(circuit_list)

    print(f"\n  Co-occurrence matrix: {n_s} syntax features x {n_c} circuits")

    matrix = np.zeros((n_s, n_c), dtype=np.float32)

    for r in results:
        active_syn = []
        for layer, feat, val, rel in r["syntax"]:
            key = f"L{layer}_F{feat}"
            if key in syntax_idx:
                active_syn.append((syntax_idx[key], val))

        active_cir = []
        for cid, score in r.get("active_circuits", []):
            if cid in circuit_idx:
                active_cir.append((circuit_idx[cid], score))

        for si, gate_val in active_syn:
            for ci, circ_score in active_cir:
                matrix[si, ci] += abs(gate_val) * circ_score

    return {
        "matrix": matrix,
        "syntax_features": syntax_list,
        "circuit_ids": circuit_list,
    }


def build_cooccurrence_by_heads(results):
    """
    Alternative co-occurrence: syntax_feature x attention_head.
    Works without circuit-discover output.
    """
    all_syntax = set()
    all_heads = set()

    for r in results:
        for layer, feat, val, rel in r["syntax"]:
            all_syntax.add(f"L{layer}_F{feat}")
        for layer, head, ent, maxw in r["attention"][:5]:
            all_heads.add(f"L{layer}_H{head}")

    syntax_list = sorted(all_syntax)
    head_list = sorted(all_heads)
    syntax_idx = {f: i for i, f in enumerate(syntax_list)}
    head_idx = {h: i for i, h in enumerate(head_list)}

    n_s = len(syntax_list)
    n_h = len(head_list)

    print(f"\n  Head co-occurrence matrix: {n_s} syntax features x {n_h} heads")

    matrix = np.zeros((n_s, n_h), dtype=np.float32)

    for r in results:
        active_syn = []
        for layer, feat, val, rel in r["syntax"]:
            key = f"L{layer}_F{feat}"
            if key in syntax_idx:
                active_syn.append((syntax_idx[key], abs(val)))

        active_hd = []
        for layer, head, ent, maxw in r["attention"][:5]:
            key = f"L{layer}_H{head}"
            if key in head_idx:
                active_hd.append((head_idx[key], maxw))

        for si, gate_val in active_syn:
            for hi, head_weight in active_hd:
                matrix[si, hi] += gate_val * head_weight

    return {
        "matrix": matrix,
        "syntax_features": syntax_list,
        "heads": head_list,
    }


# ---- Analysis -----------------------------------------------------------

def analyze_cooccurrence(cooc, results, label="Circuit"):
    """Analyze a co-occurrence matrix for routing structure."""

    matrix = cooc["matrix"]
    syntax_features = cooc["syntax_features"]
    col_labels = cooc.get("circuit_ids") or cooc.get("heads", [])

    n_s, n_c = matrix.shape

    print(f"\n{'='*70}")
    print(f"CO-OCCURRENCE ANALYSIS ({label})")
    print(f"{'='*70}")

    # Sparsity
    nonzero = np.count_nonzero(matrix)
    total = n_s * n_c
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    print(f"\n  Matrix size: {n_s} x {n_c}")
    print(f"  Non-zero entries: {nonzero} / {total}")
    print(f"  Sparsity: {sparsity:.3f}")
    if sparsity > 0.8:
        print(f"  -> SPARSE (routing table likely!)")
    elif sparsity > 0.6:
        print(f"  -> MODERATELY SPARSE")
    else:
        print(f"  -> DENSE (compositional routing)")

    # Row sparsity
    row_nonzero = np.count_nonzero(matrix, axis=1)
    avg_cols_per_feature = np.mean(row_nonzero) if len(row_nonzero) > 0 else 0
    print(f"\n  Avg {label.lower()}s per syntax feature: {avg_cols_per_feature:.1f}")
    print(f"  -> {'SELECTIVE' if avg_cols_per_feature < 5 else 'BROAD'}")

    # Column sparsity
    col_nonzero = np.count_nonzero(matrix, axis=0)
    avg_features_per_col = np.mean(col_nonzero) if len(col_nonzero) > 0 else 0
    print(f"  Avg syntax features per {label.lower()}: {avg_features_per_col:.1f}")
    print(f"  -> {'SELECTIVE' if avg_features_per_col < 10 else 'BROAD'}")

    # Block structure
    if n_s > 1:
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        normed = matrix / row_norms
        sim = normed @ normed.T
        avg_sim = np.mean(sim[np.triu_indices(n_s, k=1)])
        print(f"\n  Avg pairwise similarity: {avg_sim:.3f}")
        print(f"  -> {'CLUSTERED' if avg_sim < 0.3 else 'UNIFORM'}")
    else:
        avg_sim = 0.0

    # Per-category analysis
    print(f"\n  Per-category breakdown:")
    category_cols = defaultdict(lambda: defaultdict(float))

    for r in results:
        cat = r["category"]
        if "active_circuits" in r:
            for cid, score in r["active_circuits"]:
                category_cols[cat][cid] += score
        else:
            for layer, head, ent, maxw in r["attention"][:5]:
                category_cols[cat][f"L{layer}_H{head}"] += maxw

    for cat in sorted(category_cols.keys()):
        cols = category_cols[cat]
        top = sorted(cols.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"{c}({s:.1f})" for c, s in top)
        n_unique = len(cols)
        print(f"    {cat:30s} -> {n_unique:3d} {label.lower()}s  top: {top_str}")

    # Top routing rules
    print(f"\n  Top syntax->{label.lower()} routing rules:")
    rules = []
    for si in range(n_s):
        for ci in range(n_c):
            if matrix[si, ci] > 0:
                rules.append((syntax_features[si], col_labels[ci], matrix[si, ci]))
    rules.sort(key=lambda x: x[2], reverse=True)

    for feat, col, score in rules[:20]:
        print(f"    {feat:25s} -> {str(col):15s}  (co-occurrence: {score:.1f})")

    return {
        "sparsity": sparsity,
        "avg_cols_per_feature": avg_cols_per_feature,
        "avg_features_per_col": avg_features_per_col,
        "avg_pairwise_similarity": float(avg_sim),
        "category_cols": {k: dict(v) for k, v in category_cols.items()},
    }


# ---- Output Helpers ----------------------------------------------------

def save_matrix_plot(cooc, output_path, label="Circuit"):
    """Save co-occurrence matrix as heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        matrix = cooc["matrix"]
        display = np.log1p(matrix) if matrix.max() > 0 else matrix

        fig, ax = plt.subplots(figsize=(16, 12))
        im = ax.imshow(display, aspect='auto', cmap='viridis')
        ax.set_xlabel(f'{label} ID')
        ax.set_ylabel('Syntax Feature')
        ax.set_title(f'Syntax Feature x {label} Co-occurrence')
        plt.colorbar(im, label='log(1 + co-occurrence)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {output_path}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


def save_routing_table(cooc, output_path):
    """Save the routing table as JSON for LARQL engine."""
    matrix = cooc["matrix"]
    syntax_features = cooc["syntax_features"]
    col_labels = cooc.get("circuit_ids") or cooc.get("heads", [])

    routing_table = []
    for si, feat in enumerate(syntax_features):
        row = matrix[si, :]
        if row.max() == 0:
            continue

        top_indices = np.argsort(row)[::-1]
        top_entries = []
        for ci in top_indices:
            if row[ci] > 0:
                top_entries.append({
                    "target": col_labels[ci] if isinstance(col_labels[ci], str) else int(col_labels[ci]),
                    "score": float(row[ci]),
                    "normalized": float(row[ci] / row.sum()),
                })
            if len(top_entries) >= 5:
                break

        if top_entries:
            routing_table.append({
                "syntax_feature": feat,
                "top_targets": top_entries,
                "selectivity": float(1.0 - (np.count_nonzero(row) / len(row))),
            })

    routing_table.sort(key=lambda x: x["selectivity"], reverse=True)

    with open(output_path, 'w') as f:
        json.dump(routing_table, f, indent=2)

    print(f"  Routing table saved: {output_path} ({len(routing_table)} rules)")


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Syntax->Circuit Routing Experiment"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., google/gemma-3-4b-it)")
    parser.add_argument("--vindex", required=True,
                        help="Path to vindex directory")
    parser.add_argument("--circuits",
                        help="Path to circuits.json (optional)")
    parser.add_argument("--output", default="output/syntax_circuit_routing/",
                        help="Output directory")
    parser.add_argument("--gate-threshold", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SYNTAX -> CIRCUIT ROUTING EXPERIMENT")
    print("=" * 70)
    print(f"\n  Hypothesis: Syntax features (L0-12) predict which attention")
    print(f"  circuits (L13-26) activate, forming a cacheable routing table.")
    print()

    # Load data
    print("Loading vindex...")
    config, gates, labels = load_vindex(args.vindex)

    bands = config.get("layer_bands", {})
    syntax_end = bands.get("syntax", [0, 12])[1]
    knowledge_start = bands.get("knowledge", [13, 27])[0]
    knowledge_end = bands.get("knowledge", [13, 27])[1]
    syntax_range = range(0, syntax_end + 1)
    knowledge_range = range(knowledge_start, knowledge_end + 1)

    circuits = None
    if args.circuits and Path(args.circuits).exists():
        print("Loading circuits...")
        circuits = load_circuits(args.circuits)

    print("Loading model...")
    model, tokenizer = load_model(args.model)

    total_prompts = sum(len(t["prompts"]) for t in PROMPT_TEMPLATES.values())
    print(f"\n  Total prompts: {total_prompts} across "
          f"{len(PROMPT_TEMPLATES)} template families")

    # Run experiment
    print(f"\nRunning probes...")
    results = []
    t0 = time.time()
    n_done = 0

    for template_name, template in PROMPT_TEMPLATES.items():
        category = template["category"]
        trigram_type = template["trigram_type"]

        for prompt in template["prompts"]:
            try:
                capture = capture_one(
                    model, tokenizer, prompt, gates, labels, config,
                    syntax_range, knowledge_range,
                    gate_threshold=args.gate_threshold,
                )

                result = {
                    "prompt": prompt,
                    "template": template_name,
                    "category": category,
                    "trigram_type": trigram_type,
                    "syntax": capture["syntax"],
                    "attention": capture["attention"],
                    "prediction": capture["prediction"],
                }

                # Circuit matching (if circuits available)
                if circuits:
                    active = match_to_circuits(
                        capture["fingerprints"], circuits
                    )
                    result["active_circuits"] = active

                results.append(result)

            except Exception as e:
                print(f"\n  ERROR on '{prompt[:30]}': {e}")
                import traceback
                traceback.print_exc()

            n_done += 1
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total_prompts - n_done) / rate if rate > 0 else 0
            print(f"\r  {n_done}/{total_prompts} "
                  f"({rate:.1f}/s, ETA {eta:.0f}s)", end="", flush=True)

    print(f"\n  Done: {len(results)} captures in {time.time()-t0:.0f}s")

    # Build co-occurrence matrices
    if circuits:
        print("\nBuilding circuit co-occurrence matrix...")
        cooc_circuit = build_cooccurrence_matrix(results)
        analysis_circuit = analyze_cooccurrence(
            cooc_circuit, results, label="Circuit"
        )
        save_matrix_plot(
            cooc_circuit,
            str(output_dir / "cooccurrence_circuits.png"),
            label="Circuit"
        )
        save_routing_table(
            cooc_circuit, str(output_dir / "routing_table_circuits.json")
        )

    # Always build head-level co-occurrence (works without circuits)
    print("\nBuilding head co-occurrence matrix...")
    cooc_heads = build_cooccurrence_by_heads(results)
    analysis_heads = analyze_cooccurrence(
        cooc_heads, results, label="Head"
    )
    save_matrix_plot(
        cooc_heads,
        str(output_dir / "cooccurrence_heads.png"),
        label="Head"
    )
    save_routing_table(
        cooc_heads, str(output_dir / "routing_table_heads.json")
    )

    # Save results
    print(f"\nSaving outputs to {output_dir}/...")

    # Compact results (no raw attention patterns)
    compact = []
    for r in results:
        compact.append({
            "prompt": r["prompt"],
            "template": r["template"],
            "category": r["category"],
            "trigram_type": r["trigram_type"],
            "syntax": [(l, f, v, rel) for l, f, v, rel in r["syntax"]],
            "attention_top5": [(l, h, e, m) for l, h, e, m in r["attention"][:5]],
            "prediction": r["prediction"],
            "active_circuits": r.get("active_circuits", []),
        })

    with open(output_dir / "results.json", 'w') as f:
        json.dump(compact, f, indent=2)

    analysis = {"heads": analysis_heads}
    if circuits:
        analysis["circuits"] = analysis_circuit

    with open(output_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Verdict
    primary = analysis_heads
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    sparsity = primary["sparsity"]
    avg_per_feat = primary["avg_cols_per_feature"]

    if sparsity > 0.8 and avg_per_feat < 5:
        print(f"\n  SPARSE ROUTING TABLE EXISTS")
        print(f"    Syntax features selectively predict attention heads.")
        print(f"    Attention can be replaced with cached lookup.")
        print(f"    -> LARQL can eliminate QK computation.")
    elif sparsity > 0.6:
        print(f"\n  ~ PARTIALLY SPARSE")
        print(f"    Some syntax features are selective, others are broad.")
        print(f"    Partial caching possible for selective features.")
        print(f"    -> LARQL may need hybrid approach.")
    else:
        print(f"\n  DENSE / COMPOSITIONAL")
        print(f"    Syntax features don't cleanly predict heads/circuits.")
        print(f"    Attention routing is compositional, not template-based.")
        print(f"    -> Need different approach for attention replacement.")

    print()


if __name__ == "__main__":
    main()
