#!/usr/bin/env python3
"""
q_subclusters.py — Sub-centroid routing: let each template have 2-3 variants

K-means found K=19 optimal for 16 templates. Some templates need multiple
centroids (declarative vs question form). Instead of forcing one centroid
per template, use K-means within each template to find natural sub-clusters,
then route to the nearest sub-centroid.

Also exports the final routing table as a binary blob for larql-inference.

USAGE:
  python3 experiments/05_syntax_circuit_routing/q_subclusters.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Diverse training prompts (same as q_centroid_diverse.py) ----------

DIVERSE_TRAIN = {
    "capital_of": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Germany is",
        "The capital of India is",
        "The capital of Mexico is",
        "The capital of Canada is",
        "The capital of Italy is",
        "The capital of Spain is",
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Germany?",
        "What's the capital of India?",
        "Do you know the capital of Brazil?",
        "Tell me the capital of Mexico.",
        "France's capital is",
        "The capital city of Egypt is",
        "Name the capital of Canada.",
    ],
    "language_of": [
        "The official language of France is",
        "The official language of Japan is",
        "The official language of Brazil is",
        "The official language of China is",
        "The official language of Germany is",
        "The official language of Russia is",
        "The official language of Italy is",
        "What language do people speak in France?",
        "What language is spoken in Japan?",
        "What do they speak in Brazil?",
        "Tell me the language of China.",
        "The language spoken in Germany is",
        "People in Russia speak",
    ],
    "currency_of": [
        "The currency of Japan is the",
        "The currency of India is the",
        "The currency of Brazil is the",
        "The currency of Mexico is the",
        "The currency of Sweden is the",
        "The currency of Poland is the",
        "The currency of Thailand is the",
        "What currency does Japan use?",
        "What money do they use in India?",
        "What is the currency of Brazil?",
        "Tell me what currency Mexico uses.",
        "The money used in Sweden is the",
        "In Poland they pay with the",
    ],
    "continent_of": [
        "France is located in",
        "Japan is located in",
        "Brazil is located in",
        "Nigeria is located in",
        "Australia is located in",
        "Canada is located in",
        "Egypt is located in",
        "What continent is France on?",
        "Which continent is Japan in?",
        "Where is Brazil located?",
        "On which continent is Nigeria?",
        "Tell me what continent Australia is on.",
        "Egypt is on the continent of",
    ],
    "occupation_of": [
        "The occupation of Einstein was",
        "The occupation of Shakespeare was",
        "The occupation of Mozart was",
        "The occupation of Picasso was",
        "The occupation of Darwin was",
        "The occupation of Newton was",
        "The occupation of Beethoven was",
        "What did Einstein do for a living?",
        "What was Shakespeare's profession?",
        "What was Mozart's job?",
        "Tell me what Picasso did.",
        "Darwin worked as a",
        "Newton's profession was",
    ],
    "birthplace_of": [
        "Einstein was born in",
        "Shakespeare was born in",
        "Mozart was born in",
        "Picasso was born in",
        "Darwin was born in",
        "Newton was born in",
        "Beethoven was born in",
        "Where was Einstein born?",
        "Where was Shakespeare from?",
        "Where did Mozart come from?",
        "Tell me where Picasso was born.",
        "The birthplace of Darwin is",
        "Newton came from",
    ],
    "synonym": [
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
        "What's another word for happy?",
        "What does sad mean?",
        "Give me a synonym for big.",
        "A word that means the same as fast is",
        "Another way to say brave is",
    ],
    "antonym": [
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
        "What is the opposite of happy?",
        "What's the opposite of big?",
        "The antonym of fast is",
        "The reverse of hot is",
        "Tell me the opposite of strong.",
    ],
    "analogy": [
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
    ],
    "hypernym": [
        "A dog is a type of",
        "A rose is a type of",
        "A piano is a type of",
        "A hammer is a type of",
        "A sedan is a type of",
        "A sparrow is a type of",
        "A salmon is a type of",
        "A diamond is a type of",
        "A violin is a type of",
        "What kind of thing is a dog?",
        "What category does a rose belong to?",
        "A piano is a kind of",
        "Is a sparrow a type of bird or fish?",
        "Tell me what type of thing a hammer is.",
        "A sedan is a kind of",
    ],
    "arithmetic": [
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
        "What is 7 + 8?",
        "Calculate 50 - 25.",
        "How much is 6 * 7?",
    ],
    "code_python": [
        "def hello():\n    return",
        "def add(a, b):\n    return",
        "def factorial(n):\n    if n ==",
        "def greet(name):\n    print",
        "def is_even(n):\n    return",
        "class Dog:\n    def __init__",
        "class Person:\n    def __init__",
    ],
    "code_rust": [
        "fn main() {\n    let x =",
        "fn add(a: i32, b: i32) -> i32 {\n    a",
        "struct Point {\n    x:",
        "impl Display for Point {\n    fn fmt",
        "let mut vec = Vec::new();\n    vec",
        "match result {\n    Ok(val) =>",
        "enum Color {\n    Red,",
    ],
    "comparison": [
        "An elephant is bigger than a",
        "A cheetah is faster than a",
        "The sun is hotter than the",
        "Gold is heavier than",
        "Mount Everest is taller than",
        "The Pacific is larger than the",
        "A diamond is harder than",
        "Which is bigger, an elephant or a mouse?",
        "Is a cheetah faster than a lion?",
        "What is heavier, gold or silver?",
    ],
    "causation": [
        "Plants grow because they need",
        "Ice melts because the temperature",
        "Birds fly because they have",
        "People sleep because the body",
        "Fire burns because of",
        "Metal rusts because of",
        "Rain falls because water",
        "Why do plants grow?",
        "Why does ice melt?",
        "What causes birds to fly?",
    ],
    "temporal": [
        "World War II ended in",
        "The Roman Empire fell in",
        "The internet was invented in",
        "The first airplane flew in",
        "The moon landing happened in",
        "The Berlin Wall fell in",
        "The printing press was invented in",
        "When did World War II end?",
        "When was the internet invented?",
        "What year did the moon landing happen?",
        "Tell me when the Berlin Wall fell.",
        "In what year was the printing press invented?",
    ],
}

# ---- Same stress prompts -----------------------------------------------

STRESS_PROMPTS = {
    "long_context": [
        {"text": "In the early 19th century, the capital of the country that borders Germany to the west is", "expected": "capital_of"},
        {"text": "After years of research and many publications, the occupation of the famous physicist Albert Einstein was", "expected": "occupation_of"},
        {"text": "If you travel to the largest country in South America and ask someone what language they speak, the official language of Brazil is", "expected": "language_of"},
        {"text": "Among all the currencies used in Asian countries, the currency of Japan is the", "expected": "currency_of"},
        {"text": "Looking at a map of the world and considering the major continents, Australia is located in", "expected": "continent_of"},
    ],
    "compositional": [
        {"text": "The French-speaking capital of a country in Africa is", "expected": ["capital_of", "language_of", "continent_of"]},
        {"text": "The European country whose currency is the krona has its capital in", "expected": ["capital_of", "currency_of", "continent_of"]},
        {"text": "The birthplace of the famous German composer Beethoven was", "expected": ["birthplace_of", "occupation_of"]},
        {"text": "A fast animal that is bigger than a dog is a type of", "expected": ["hypernym", "comparison"]},
        {"text": "The opposite of the word that means happy is", "expected": ["antonym", "synonym"]},
    ],
    "ambiguous": [
        {"text": "Paris is", "expected": ["capital_of", "continent_of", "birthplace_of"]},
        {"text": "Mozart was", "expected": ["occupation_of", "birthplace_of"]},
        {"text": "Japan is", "expected": ["continent_of", "capital_of", "language_of"]},
        {"text": "Light is", "expected": ["comparison", "synonym", "hypernym"]},
        {"text": "Python is", "expected": ["hypernym", "code_python"]},
    ],
    "natural_language": [
        {"text": "So what money do they use in Japan anyway?", "expected": "currency_of"},
        {"text": "What language do people speak in Brazil?", "expected": "language_of"},
        {"text": "Where was Einstein from originally?", "expected": "birthplace_of"},
        {"text": "What did Picasso do for a living?", "expected": "occupation_of"},
        {"text": "What's another word for happy?", "expected": "synonym"},
        {"text": "What's the opposite of strong?", "expected": "antonym"},
        {"text": "Which continent is Nigeria on?", "expected": "continent_of"},
        {"text": "Tell me the capital city of Thailand.", "expected": "capital_of"},
        {"text": "When did the French Revolution start?", "expected": "temporal"},
        {"text": "Is a whale a kind of fish or mammal?", "expected": "hypernym"},
    ],
    "multi_hop": [
        {"text": "The currency of the country where Einstein was born is the", "expected": ["currency_of", "birthplace_of"]},
        {"text": "The language spoken in the capital of Japan is", "expected": ["language_of", "capital_of"]},
        {"text": "The continent where the birthplace of Mozart is located is", "expected": ["continent_of", "birthplace_of"]},
        {"text": "The occupation of the person who was born in Stratford-upon-Avon was", "expected": ["occupation_of", "birthplace_of"]},
        {"text": "A word that means the opposite of the synonym of sad is", "expected": ["antonym", "synonym"]},
    ],
}


# ---- Model helpers -----------------------------------------------------

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


def forward_capture_q_at_layer(model, tokenizer, prompt, target_layer):
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    h = embed_fn(input_ids)
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])

    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    q_vector = None

    for i, layer in enumerate(layers):
        if i == target_layer:
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
            q_vector = np.array(q_last.astype(mx.float32))
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)
            weights = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                weights = weights + mask
            weights = mx.softmax(weights, axis=-1)
            mx.eval(weights)
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

    return q_vector


# ---- Sub-clustering ----------------------------------------------------

def find_subclusters(vecs_n, max_k=3, min_split_gain=0.15):
    """
    Find optimal number of sub-clusters for a template's Q vectors.

    Uses silhouette-like criterion: split only if within-cluster spread
    improves significantly (the sub-clusters are tighter than one cluster).
    """
    n = len(vecs_n)
    if n < 4:
        # Too few samples to split
        c = vecs_n.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        return [c], [list(range(n))]

    # K=1 baseline: spread = 1 - mean(cosine to centroid)
    c1 = vecs_n.mean(axis=0)
    c1 /= np.linalg.norm(c1) + 1e-10
    spread_1 = 1.0 - float((vecs_n @ c1).mean())

    best_k = 1
    best_centroids = [c1]
    best_assignments = [list(range(n))]
    best_spread = spread_1

    for k in range(2, min(max_k + 1, n)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(vecs_n)

        centroids_k = []
        assignments_k = []
        spreads = []

        for ci in range(k):
            mask = labels == ci
            if mask.sum() == 0:
                continue
            c = vecs_n[mask].mean(axis=0)
            c /= np.linalg.norm(c) + 1e-10
            centroids_k.append(c)
            assignments_k.append(np.where(mask)[0].tolist())
            spreads.append(1.0 - float((vecs_n[mask] @ c).mean()))

        avg_spread_k = np.mean(spreads)
        gain = spread_1 - avg_spread_k

        if gain > min_split_gain and avg_spread_k < best_spread:
            best_k = k
            best_centroids = centroids_k
            best_assignments = assignments_k
            best_spread = avg_spread_k

    return best_centroids, best_assignments


def classify_with_subclusters(q_n, all_subclusters):
    """
    Classify a Q vector against sub-centroid routing table.

    Returns list of (template_name, sub_id, cosine_similarity) sorted desc.
    """
    results = []
    for template_name, subclusters in all_subclusters.items():
        for sub_id, centroid in enumerate(subclusters):
            sim = float(q_n @ centroid)
            results.append((template_name, sub_id, sim))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ---- Binary export ------------------------------------------------------

def export_routing_table(all_subclusters, template_names, output_path, dim):
    """
    Export routing table as binary blob for larql-inference.

    Format:
      Header (16 bytes):
        magic: u32 = 0x51525442 ("QRTB")
        version: u32 = 1
        n_templates: u32
        n_total_centroids: u32

      Template index (n_templates * 12 bytes each):
        template_name_len: u32
        n_subclusters: u32
        centroid_dim: u32

      Template names (variable, padded to 4-byte boundary):
        name bytes (utf-8, null-terminated)

      Centroid vectors (n_total_centroids * dim * 4 bytes):
        f32 vectors, contiguous

      Template-to-centroid mapping (n_total_centroids * 4 bytes):
        template_id: u32 for each centroid
    """
    n_templates = len(template_names)
    total_centroids = sum(len(v) for v in all_subclusters.values())

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<IIII',
            0x51525442,  # "QRTB"
            1,           # version
            n_templates,
            total_centroids,
        ))

        # Write dim
        f.write(struct.pack('<I', dim))

        # Template metadata
        for name in template_names:
            n_sub = len(all_subclusters[name])
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<II', len(name_bytes), n_sub))
            f.write(name_bytes)
            # Pad to 4-byte boundary
            pad = (4 - len(name_bytes) % 4) % 4
            f.write(b'\x00' * pad)

        # Centroid vectors + template mapping
        centroid_data = []
        template_ids = []
        for tid, name in enumerate(template_names):
            for centroid in all_subclusters[name]:
                centroid_data.append(centroid.astype(np.float32))
                template_ids.append(tid)

        # Write centroids
        for c in centroid_data:
            f.write(c.tobytes())

        # Write template IDs
        for tid in template_ids:
            f.write(struct.pack('<I', tid))

    file_size = Path(output_path).stat().st_size
    print(f"  Routing table exported: {output_path}")
    print(f"    {total_centroids} centroids x {dim}d = {file_size / 1024:.1f} KB")

    # Also export JSON metadata
    meta = {
        "magic": "QRTB",
        "version": 1,
        "n_templates": n_templates,
        "n_centroids": total_centroids,
        "dim": dim,
        "templates": [],
    }
    offset = 0
    for name in template_names:
        n_sub = len(all_subclusters[name])
        meta["templates"].append({
            "name": name,
            "n_subclusters": n_sub,
            "centroid_offset": offset,
        })
        offset += n_sub

    meta_path = str(output_path).replace('.bin', '.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"    Metadata: {meta_path}")


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sub-centroid routing + binary export"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--max-sub", type=int, default=3,
                        help="Max sub-clusters per template")
    parser.add_argument("--min-split-gain", type=float, default=0.05,
                        help="Min spread improvement to justify splitting")
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = args.layer

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    print(f"  Model: {args.model}, Layer: L{target_layer}")

    template_names = list(DIVERSE_TRAIN.keys())
    total_train = sum(len(v) for v in DIVERSE_TRAIN.values())

    # ---- Capture Q vectors ----
    print(f"\nCapturing Q vectors ({total_train} prompts)...")
    t0 = time.time()
    n = 0
    raw_vecs = {}

    for name, prompts in DIVERSE_TRAIN.items():
        vecs = []
        for prompt in prompts:
            q = forward_capture_q_at_layer(model, tokenizer, prompt, target_layer)
            vecs.append(q.flatten())
            n += 1
            print(f"\r  {n}/{total_train} ({time.time()-t0:.0f}s)", end="", flush=True)
        raw_vecs[name] = np.stack(vecs)

    dim = raw_vecs[template_names[0]].shape[1]
    print(f"\n  Done in {time.time()-t0:.0f}s, dim={dim}")

    # ---- Find sub-clusters ----
    print(f"\n{'='*70}")
    print(f"SUB-CLUSTER DISCOVERY (max_sub={args.max_sub}, min_gain={args.min_split_gain})")
    print(f"{'='*70}")

    all_subclusters = {}
    total_centroids = 0

    for name in template_names:
        vecs_n = normalize(raw_vecs[name])

        # Single centroid baseline
        c1 = vecs_n.mean(axis=0)
        c1 /= np.linalg.norm(c1) + 1e-10
        spread_1 = 1.0 - float((vecs_n @ c1).mean())

        subclusters, assignments = find_subclusters(
            vecs_n, max_k=args.max_sub, min_split_gain=args.min_split_gain
        )
        all_subclusters[name] = subclusters

        k = len(subclusters)
        total_centroids += k

        if k > 1:
            sub_sizes = [len(a) for a in assignments]
            sub_spreads = []
            for sc, assign in zip(subclusters, assignments):
                sub_vecs = vecs_n[assign]
                sub_spreads.append(1.0 - float((sub_vecs @ sc).mean()))
            avg_sub_spread = np.mean(sub_spreads)
            print(f"\n  {name:20s}: SPLIT into {k} sub-clusters")
            print(f"    Single spread: {spread_1:.4f}")
            print(f"    Sub-cluster spreads: {[f'{s:.4f}' for s in sub_spreads]}")
            print(f"    Sizes: {sub_sizes}")
            print(f"    Gain: {spread_1 - avg_sub_spread:.4f}")

            # Show example prompts per sub-cluster
            for si, assign in enumerate(assignments):
                prompts = DIVERSE_TRAIN[name]
                examples = [prompts[i][:50] for i in assign[:3]]
                print(f"    Sub-{si}: {examples}")
        else:
            print(f"  {name:20s}: 1 centroid (spread={spread_1:.4f})")

    print(f"\n  Total centroids: {total_centroids} (was 16 single)")

    # ---- Build comparison: single vs sub-centroid ----
    # Single centroids for baseline
    single_centroids = {}
    for name, vecs in raw_vecs.items():
        vecs_n = normalize(vecs)
        c = vecs_n.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        single_centroids[name] = [c]

    # ---- Run stress tests ----
    total_stress = sum(len(v) for v in STRESS_PROMPTS.values())
    print(f"\n{'='*70}")
    print(f"STRESS TEST ({total_stress} prompts)")
    print(f"{'='*70}")

    cat_stats_sub = {}
    cat_stats_single = {}

    for cat_name, prompts in STRESS_PROMPTS.items():
        n_top1_sub = 0
        n_top3_sub = 0
        n_top1_single = 0
        n_top3_single = 0

        print(f"\n  -- {cat_name.upper()} --")

        for p in prompts:
            prompt = p["text"]
            expected = p["expected"]
            if isinstance(expected, str):
                expected = [expected]

            q = forward_capture_q_at_layer(model, tokenizer, prompt, target_layer)
            q_flat = q.flatten()
            q_n = q_flat / (np.linalg.norm(q_flat) + 1e-10)

            # Sub-centroid classification
            ranked_sub = classify_with_subclusters(q_n, all_subclusters)
            # Deduplicate by template (keep best sub-cluster per template)
            seen = set()
            top_templates_sub = []
            for tname, sid, sim in ranked_sub:
                if tname not in seen:
                    seen.add(tname)
                    top_templates_sub.append((tname, sid, sim))

            top1_sub = top_templates_sub[0][0]
            top3_sub = [t[0] for t in top_templates_sub[:3]]

            top1_hit_sub = top1_sub in expected
            top3_hit_sub = any(e in top3_sub for e in expected)
            if top1_hit_sub:
                n_top1_sub += 1
            if top3_hit_sub:
                n_top3_sub += 1

            # Single centroid classification
            ranked_single = classify_with_subclusters(q_n, single_centroids)
            seen2 = set()
            top_templates_single = []
            for tname, sid, sim in ranked_single:
                if tname not in seen2:
                    seen2.add(tname)
                    top_templates_single.append((tname, sid, sim))

            top1_single = top_templates_single[0][0]
            top3_single = [t[0] for t in top_templates_single[:3]]

            top1_hit_single = top1_single in expected
            top3_hit_single = any(e in top3_single for e in expected)
            if top1_hit_single:
                n_top1_single += 1
            if top3_hit_single:
                n_top3_single += 1

            # Display
            s_sub = "OK" if top1_hit_sub else ("t3" if top3_hit_sub else "XX")
            s_sin = "OK" if top1_hit_single else ("t3" if top3_hit_single else "XX")
            improved = ""
            if s_sub != s_sin:
                if (s_sub == "OK" and s_sin != "OK") or (s_sub == "t3" and s_sin == "XX"):
                    improved = " IMPROVED"
                elif (s_sin == "OK" and s_sub != "OK") or (s_sin == "t3" and s_sub == "XX"):
                    improved = " REGRESSED"

            prompt_short = prompt[:50] + ("..." if len(prompt) > 50 else "")
            t1_sub_info = f"{top_templates_sub[0][0]}({top_templates_sub[0][2]:.3f})"
            t1_sin_info = f"{top_templates_single[0][0]}({top_templates_single[0][2]:.3f})"
            print(f"    [{s_sub}|{s_sin}] \"{prompt_short}\"")
            print(f"             sub={t1_sub_info}  single={t1_sin_info}{improved}")

        n_p = len(prompts)
        cat_stats_sub[cat_name] = {"n": n_p, "top1": n_top1_sub/n_p, "top3": n_top3_sub/n_p}
        cat_stats_single[cat_name] = {"n": n_p, "top1": n_top1_single/n_p, "top3": n_top3_single/n_p}

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"SUMMARY: SUB-CENTROID ({total_centroids}) vs SINGLE (16)")
    print(f"{'='*70}")

    print(f"\n  {'Category':<20s}  {'Sub-centroid':>12s}  {'Single':>12s}  {'Delta':>6s}")
    print(f"  {'':20s}  {'t1    t3':>12s}  {'t1    t3':>12s}  {'t1':>6s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*6}")

    total_n = 0
    total_sub1 = total_sub3 = total_sin1 = total_sin3 = 0

    for cat in STRESS_PROMPTS:
        ss = cat_stats_sub[cat]
        si = cat_stats_single[cat]
        n_p = ss["n"]
        total_n += n_p
        total_sub1 += int(ss["top1"] * n_p)
        total_sub3 += int(ss["top3"] * n_p)
        total_sin1 += int(si["top1"] * n_p)
        total_sin3 += int(si["top3"] * n_p)

        d = ss["top1"] - si["top1"]
        ds = f"{d:+.0%}" if d != 0 else "  ="
        print(f"  {cat:<20s}  {ss['top1']:3.0%}  {ss['top3']:3.0%}    "
              f"{si['top1']:3.0%}  {si['top3']:3.0%}    {ds}")

    o_sub1 = total_sub1/total_n
    o_sub3 = total_sub3/total_n
    o_sin1 = total_sin1/total_n
    o_sin3 = total_sin3/total_n
    od = o_sub1 - o_sin1

    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*6}")
    print(f"  {'OVERALL':<20s}  {o_sub1:3.0%}  {o_sub3:3.0%}    "
          f"{o_sin1:3.0%}  {o_sin3:3.0%}    {od:+.0%}")

    # ---- Export routing table ----
    print(f"\n{'='*70}")
    print(f"EXPORTING ROUTING TABLE")
    print(f"{'='*70}")

    export_routing_table(
        all_subclusters, template_names,
        str(output_dir / "routing_table.bin"),
        dim,
    )

    # Also save JSON version for easy inspection
    json_table = {}
    for name in template_names:
        json_table[name] = {
            "n_subclusters": len(all_subclusters[name]),
            "centroids": [c.tolist() for c in all_subclusters[name]],
        }
    with open(output_dir / "routing_centroids.json", 'w') as f:
        json.dump(json_table, f)
    print(f"  JSON centroids: {output_dir / 'routing_centroids.json'}")

    # ---- Final verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    print(f"\n  {total_centroids} sub-centroids across {len(template_names)} templates")
    print(f"  Overall: top-1={o_sub1:.0%}  top-3={o_sub3:.0%}")
    print(f"  vs single: top-1={o_sin1:.0%}  top-3={o_sin3:.0%}")
    print(f"  Improvement: top-1 {od:+.0%}")

    if o_sub3 >= 0.95:
        print(f"\n  ROUTING TABLE PRODUCTION-READY")
        print(f"    {total_centroids} centroids, {dim}d, top-3 fallback = {o_sub3:.0%}")
    elif o_sub3 >= 0.90:
        print(f"\n  ROUTING TABLE VIABLE WITH TOP-3 FALLBACK")
        print(f"    {o_sub3:.0%} coverage, remaining {1-o_sub3:.0%} falls back to neural")

    print()


if __name__ == "__main__":
    main()
