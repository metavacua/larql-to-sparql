#!/usr/bin/env python3
"""
confidence_threshold.py — Confidence-gated routing

The question isn't "what % of prompts does the centroid classify correctly?"
It's "what % of prompts can we CONFIDENTLY route, with zero error?"

Clean matches land at cosine 0.85+. Misses are 0.55-0.65.
Set a threshold. Above it: trust centroid, skip attention.
Below it: fall back to neural attention.

Measure: at each threshold, what % of prompts route (coverage),
and what % of routed prompts are correct (precision)?

The target: a threshold where precision=100% and coverage is maximized.

Uses the 44 sub-centroids from q_subclusters.py, tested against:
  1. The 206 diverse training prompts (sanity check)
  2. The 30 stress prompts (hard cases)
  3. A new set of 50 "realistic corpus" prompts (mixed difficulty)

USAGE:
  python3 experiments/05_syntax_circuit_routing/confidence_threshold.py \
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
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Diverse training prompts (for centroid building) -------------------
# Same as q_centroid_diverse.py / q_subclusters.py

DIVERSE_TRAIN = {
    "capital_of": [
        "The capital of France is", "The capital of Japan is",
        "The capital of Brazil is", "The capital of Egypt is",
        "The capital of Germany is", "The capital of India is",
        "The capital of Mexico is", "The capital of Canada is",
        "The capital of Italy is", "The capital of Spain is",
        "What is the capital of France?", "What is the capital of Japan?",
        "What is the capital of Germany?", "What's the capital of India?",
        "Do you know the capital of Brazil?", "Tell me the capital of Mexico.",
        "France's capital is", "The capital city of Egypt is",
        "Name the capital of Canada.",
    ],
    "language_of": [
        "The official language of France is", "The official language of Japan is",
        "The official language of Brazil is", "The official language of China is",
        "The official language of Germany is", "The official language of Russia is",
        "The official language of Italy is",
        "What language do people speak in France?",
        "What language is spoken in Japan?", "What do they speak in Brazil?",
        "Tell me the language of China.", "The language spoken in Germany is",
        "People in Russia speak",
    ],
    "currency_of": [
        "The currency of Japan is the", "The currency of India is the",
        "The currency of Brazil is the", "The currency of Mexico is the",
        "The currency of Sweden is the", "The currency of Poland is the",
        "The currency of Thailand is the",
        "What currency does Japan use?", "What money do they use in India?",
        "What is the currency of Brazil?", "Tell me what currency Mexico uses.",
        "The money used in Sweden is the", "In Poland they pay with the",
    ],
    "continent_of": [
        "France is located in", "Japan is located in",
        "Brazil is located in", "Nigeria is located in",
        "Australia is located in", "Canada is located in", "Egypt is located in",
        "What continent is France on?", "Which continent is Japan in?",
        "Where is Brazil located?", "On which continent is Nigeria?",
        "Tell me what continent Australia is on.", "Egypt is on the continent of",
    ],
    "occupation_of": [
        "The occupation of Einstein was", "The occupation of Shakespeare was",
        "The occupation of Mozart was", "The occupation of Picasso was",
        "The occupation of Darwin was", "The occupation of Newton was",
        "The occupation of Beethoven was",
        "What did Einstein do for a living?", "What was Shakespeare's profession?",
        "What was Mozart's job?", "Tell me what Picasso did.",
        "Darwin worked as a", "Newton's profession was",
    ],
    "birthplace_of": [
        "Einstein was born in", "Shakespeare was born in",
        "Mozart was born in", "Picasso was born in",
        "Darwin was born in", "Newton was born in", "Beethoven was born in",
        "Where was Einstein born?", "Where was Shakespeare from?",
        "Where did Mozart come from?", "Tell me where Picasso was born.",
        "The birthplace of Darwin is", "Newton came from",
    ],
    "synonym": [
        "Happy means", "Sad means", "Big means", "Small means",
        "Fast means", "Slow means", "Hot means", "Cold means",
        "Smart means", "Brave means", "Angry means", "Calm means", "Rich means",
        "What's another word for happy?", "What does sad mean?",
        "Give me a synonym for big.",
        "A word that means the same as fast is", "Another way to say brave is",
    ],
    "antonym": [
        "The opposite of happy is", "The opposite of big is",
        "The opposite of fast is", "The opposite of hot is",
        "The opposite of light is", "The opposite of old is",
        "The opposite of rich is", "The opposite of strong is",
        "The opposite of early is", "The opposite of deep is",
        "What is the opposite of happy?", "What's the opposite of big?",
        "The antonym of fast is", "The reverse of hot is",
        "Tell me the opposite of strong.",
    ],
    "analogy": [
        "King is to queen as man is to", "Dog is to puppy as cat is to",
        "Hot is to cold as big is to", "France is to Paris as Japan is to",
        "Teacher is to school as doctor is to", "Bird is to fly as fish is to",
        "Hand is to glove as foot is to", "Pen is to write as knife is to",
        "Eye is to see as ear is to", "Day is to night as summer is to",
        "Book is to read as song is to", "Painter is to brush as writer is to",
        "Cow is to milk as hen is to",
    ],
    "hypernym": [
        "A dog is a type of", "A rose is a type of", "A piano is a type of",
        "A hammer is a type of", "A sedan is a type of",
        "A sparrow is a type of", "A salmon is a type of",
        "A diamond is a type of", "A violin is a type of",
        "What kind of thing is a dog?", "What category does a rose belong to?",
        "A piano is a kind of", "Is a sparrow a type of bird or fish?",
        "Tell me what type of thing a hammer is.", "A sedan is a kind of",
    ],
    "arithmetic": [
        "2 + 3 =", "7 - 4 =", "5 * 6 =", "10 / 2 =", "15 + 27 =",
        "100 - 37 =", "8 * 9 =", "48 / 6 =", "3 + 3 + 3 =", "25 * 4 =",
        "99 - 11 =", "12 * 12 =",
        "What is 7 + 8?", "Calculate 50 - 25.", "How much is 6 * 7?",
    ],
    "code_python": [
        "def hello():\n    return", "def add(a, b):\n    return",
        "def factorial(n):\n    if n ==", "def greet(name):\n    print",
        "def is_even(n):\n    return",
        "class Dog:\n    def __init__", "class Person:\n    def __init__",
    ],
    "code_rust": [
        "fn main() {\n    let x =",
        "fn add(a: i32, b: i32) -> i32 {\n    a",
        "struct Point {\n    x:", "impl Display for Point {\n    fn fmt",
        "let mut vec = Vec::new();\n    vec",
        "match result {\n    Ok(val) =>", "enum Color {\n    Red,",
    ],
    "comparison": [
        "An elephant is bigger than a", "A cheetah is faster than a",
        "The sun is hotter than the", "Gold is heavier than",
        "Mount Everest is taller than", "The Pacific is larger than the",
        "A diamond is harder than",
        "Which is bigger, an elephant or a mouse?",
        "Is a cheetah faster than a lion?", "What is heavier, gold or silver?",
    ],
    "causation": [
        "Plants grow because they need", "Ice melts because the temperature",
        "Birds fly because they have", "People sleep because the body",
        "Fire burns because of", "Metal rusts because of",
        "Rain falls because water",
        "Why do plants grow?", "Why does ice melt?", "What causes birds to fly?",
    ],
    "temporal": [
        "World War II ended in", "The Roman Empire fell in",
        "The internet was invented in", "The first airplane flew in",
        "The moon landing happened in", "The Berlin Wall fell in",
        "The printing press was invented in",
        "When did World War II end?", "When was the internet invented?",
        "What year did the moon landing happen?",
        "Tell me when the Berlin Wall fell.",
        "In what year was the printing press invented?",
    ],
}

# ---- Realistic corpus: mixed difficulty, varied phrasing ----------------

REALISTIC_CORPUS = [
    # Clean templates (should route with high confidence)
    {"text": "The capital of South Korea is", "expected": "capital_of", "difficulty": "easy"},
    {"text": "The official language of Portugal is", "expected": "language_of", "difficulty": "easy"},
    {"text": "The opposite of cold is", "expected": "antonym", "difficulty": "easy"},
    {"text": "A whale is a type of", "expected": "hypernym", "difficulty": "easy"},
    {"text": "14 + 29 =", "expected": "arithmetic", "difficulty": "easy"},
    {"text": "fn process(data: &[u8]) ->", "expected": "code_rust", "difficulty": "easy"},
    {"text": "def sort(arr):\n    return", "expected": "code_python", "difficulty": "easy"},
    {"text": "Loud means", "expected": "synonym", "difficulty": "easy"},
    {"text": "Ink is to pen as paint is to", "expected": "analogy", "difficulty": "easy"},
    {"text": "Water boils because the temperature", "expected": "causation", "difficulty": "easy"},

    # Question form (medium)
    {"text": "What's the capital of South Korea?", "expected": "capital_of", "difficulty": "medium"},
    {"text": "What language do they speak in Portugal?", "expected": "language_of", "difficulty": "medium"},
    {"text": "What currency do they use in South Korea?", "expected": "currency_of", "difficulty": "medium"},
    {"text": "What continent is Peru on?", "expected": "continent_of", "difficulty": "medium"},
    {"text": "Where was Galileo born?", "expected": "birthplace_of", "difficulty": "medium"},
    {"text": "What did Aristotle do?", "expected": "occupation_of", "difficulty": "medium"},
    {"text": "When was the telephone invented?", "expected": "temporal", "difficulty": "medium"},
    {"text": "What's the opposite of gentle?", "expected": "antonym", "difficulty": "medium"},
    {"text": "What is another word for angry?", "expected": "synonym", "difficulty": "medium"},
    {"text": "Is a penguin a type of bird or mammal?", "expected": "hypernym", "difficulty": "medium"},

    # Long context (medium-hard)
    {"text": "Considering the geographic location of the island nation in the Pacific, Japan is located in", "expected": "continent_of", "difficulty": "medium"},
    {"text": "After the fall of the Berlin Wall, the reunification of Germany happened in", "expected": "temporal", "difficulty": "medium"},
    {"text": "The small European country known for its chocolate and watches uses the currency called the", "expected": "currency_of", "difficulty": "medium"},
    {"text": "Among the great Renaissance artists of Italy, the occupation of Leonardo da Vinci was", "expected": "occupation_of", "difficulty": "medium"},
    {"text": "Looking at the nations that make up the European Union, the capital of Belgium is", "expected": "capital_of", "difficulty": "medium"},

    # Natural language / colloquial (hard)
    {"text": "Hey, what money do people use in Thailand?", "expected": "currency_of", "difficulty": "hard"},
    {"text": "I need to know where Mozart was born, can you help?", "expected": "birthplace_of", "difficulty": "hard"},
    {"text": "Can you give me a word that means the same thing as beautiful?", "expected": "synonym", "difficulty": "hard"},
    {"text": "So basically, what's the reverse of up?", "expected": "antonym", "difficulty": "hard"},
    {"text": "What kind of animal is a dolphin exactly?", "expected": "hypernym", "difficulty": "hard"},

    # Compositional (hard)
    {"text": "The language spoken in the capital of South Korea is", "expected": ["language_of", "capital_of"], "difficulty": "hard"},
    {"text": "The currency used in the country where Beethoven was born is", "expected": ["currency_of", "birthplace_of"], "difficulty": "hard"},
    {"text": "A word meaning the opposite of a synonym of sad is", "expected": ["antonym", "synonym"], "difficulty": "hard"},

    # Multi-hop (hard)
    {"text": "The continent where the capital of Thailand is located is", "expected": ["continent_of", "capital_of"], "difficulty": "hard"},
    {"text": "The official language of the country where the Eiffel Tower is located is", "expected": ["language_of", "capital_of"], "difficulty": "hard"},

    # Ambiguous (hard)
    {"text": "London is", "expected": ["capital_of", "continent_of", "birthplace_of"], "difficulty": "hard"},
    {"text": "Bach was", "expected": ["occupation_of", "birthplace_of"], "difficulty": "hard"},
    {"text": "Gold is", "expected": ["comparison", "hypernym"], "difficulty": "hard"},

    # Very natural / messy (hard)
    {"text": "So like what do you call the money they have over in Brazil?", "expected": "currency_of", "difficulty": "hard"},
    {"text": "You know that thing where one word means the opposite of another? What's that for quiet?", "expected": "antonym", "difficulty": "hard"},
    {"text": "Quick math: what's fifteen times twelve?", "expected": "arithmetic", "difficulty": "hard"},
    {"text": "Name an animal faster than a horse.", "expected": "comparison", "difficulty": "hard"},
    {"text": "Why exactly does the sun set?", "expected": "causation", "difficulty": "hard"},
    {"text": "Who invented the lightbulb and when?", "expected": "temporal", "difficulty": "hard"},

    # Edge cases
    {"text": "Translate 'hello' to French.", "expected": "language_of", "difficulty": "edge"},
    {"text": "What's 0 divided by 0?", "expected": "arithmetic", "difficulty": "edge"},
    {"text": "Is water wet?", "expected": ["causation", "hypernym"], "difficulty": "edge"},
    {"text": "How do you say goodbye in Japanese?", "expected": "language_of", "difficulty": "edge"},
    {"text": "The square root of 144 is", "expected": "arithmetic", "difficulty": "edge"},
]

# ---- Stress prompts (same as before) -----------------------------------

STRESS_PROMPTS = [
    {"text": "In the early 19th century, the capital of the country that borders Germany to the west is", "expected": "capital_of"},
    {"text": "After years of research and many publications, the occupation of the famous physicist Albert Einstein was", "expected": "occupation_of"},
    {"text": "If you travel to the largest country in South America and ask someone what language they speak, the official language of Brazil is", "expected": "language_of"},
    {"text": "Among all the currencies used in Asian countries, the currency of Japan is the", "expected": "currency_of"},
    {"text": "Looking at a map of the world and considering the major continents, Australia is located in", "expected": "continent_of"},
    {"text": "The French-speaking capital of a country in Africa is", "expected": ["capital_of", "language_of", "continent_of"]},
    {"text": "The European country whose currency is the krona has its capital in", "expected": ["capital_of", "currency_of", "continent_of"]},
    {"text": "The birthplace of the famous German composer Beethoven was", "expected": ["birthplace_of", "occupation_of"]},
    {"text": "A fast animal that is bigger than a dog is a type of", "expected": ["hypernym", "comparison"]},
    {"text": "The opposite of the word that means happy is", "expected": ["antonym", "synonym"]},
    {"text": "Paris is", "expected": ["capital_of", "continent_of", "birthplace_of"]},
    {"text": "Mozart was", "expected": ["occupation_of", "birthplace_of"]},
    {"text": "Japan is", "expected": ["continent_of", "capital_of", "language_of"]},
    {"text": "Light is", "expected": ["comparison", "synonym", "hypernym"]},
    {"text": "Python is", "expected": ["hypernym", "code_python"]},
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
    {"text": "The currency of the country where Einstein was born is the", "expected": ["currency_of", "birthplace_of"]},
    {"text": "The language spoken in the capital of Japan is", "expected": ["language_of", "capital_of"]},
    {"text": "The continent where the birthplace of Mozart is located is", "expected": ["continent_of", "birthplace_of"]},
    {"text": "The occupation of the person who was born in Stratford-upon-Avon was", "expected": ["occupation_of", "birthplace_of"]},
    {"text": "A word that means the opposite of the synonym of sad is", "expected": ["antonym", "synonym"]},
]


# ---- Model helpers (same as all previous) ------------------------------

def find_model_parts(model):
    try:
        lm = model.language_model
        inner = lm.model
        if hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
            embed_fn = inner.embed_tokens
            def lm_head(h): return h @ embed_fn.weight.T
            return embed_fn, inner.layers, inner.norm, lm_head, True
    except AttributeError:
        pass
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            f = model.lm_head
            def lm_head(h): return f(h)
        else:
            def lm_head(h): return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale
    raise RuntimeError("Could not detect model structure.")


def forward_capture_q_at_layer(model, tokenizer, prompt, target_layer):
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)
    tokens = tokenizer.encode(prompt)
    h = embed_fn(mx.array([tokens]))
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens)).astype(h.dtype)

    for i, layer in enumerate(layers):
        if i == target_layer:
            sa = layer.self_attn
            B, L, D = h.shape
            h_norm = layer.input_layernorm(h)
            q = sa.q_proj(h_norm)
            k, v = sa.k_proj(h_norm), sa.v_proj(h_norm)
            n_h, n_kv, hd, sc = sa.n_heads, sa.n_kv_heads, sa.head_dim, sa.scale
            q = q.reshape(B,L,n_h,hd).transpose(0,2,1,3)
            k = k.reshape(B,L,n_kv,hd).transpose(0,2,1,3)
            v = v.reshape(B,L,n_kv,hd).transpose(0,2,1,3)
            if hasattr(sa,'q_norm'): q = sa.q_norm(q)
            if hasattr(sa,'k_norm'): k = sa.k_norm(k)
            q, k = sa.rope(q), sa.rope(k)
            q_last = q[0,:,-1,:]
            mx.eval(q_last)
            q_vec = np.array(q_last.astype(mx.float32))
            if n_kv < n_h:
                k = mx.repeat(k, n_h//n_kv, axis=1)
                v = mx.repeat(v, n_h//n_kv, axis=1)
            w = mx.softmax((q @ k.transpose(0,1,3,2))*sc + mask, axis=-1)
            mx.eval(w)
            ao = (w @ v).transpose(0,2,1,3).reshape(B,L,-1)
            ao = sa.o_proj(ao)
            h = h + (layer.post_attention_layernorm(ao) if hasattr(layer,'post_attention_layernorm') else ao)
            hf = layer.pre_feedforward_layernorm(h) if hasattr(layer,'pre_feedforward_layernorm') else h
            fo = layer.mlp(hf)
            h = h + (layer.post_feedforward_layernorm(fo) if hasattr(layer,'post_feedforward_layernorm') else fo)
            mx.eval(h)
        else:
            h = layer(h, mask=mask)
            mx.eval(h)
    return q_vec


# ---- Sub-centroid building + classification ----------------------------

def build_subclusters(model, tokenizer, target_layer, max_k=3, min_gain=0.05):
    """Build sub-centroids from diverse training data."""
    from sklearn.cluster import KMeans

    template_names = list(DIVERSE_TRAIN.keys())
    raw_vecs = {}
    n = 0
    total = sum(len(v) for v in DIVERSE_TRAIN.values())
    t0 = time.time()

    for name, prompts in DIVERSE_TRAIN.items():
        vecs = []
        for prompt in prompts:
            q = forward_capture_q_at_layer(model, tokenizer, prompt, target_layer)
            vecs.append(q.flatten())
            n += 1
            print(f"\r  Building centroids: {n}/{total} ({time.time()-t0:.0f}s)", end="", flush=True)
        raw_vecs[name] = np.stack(vecs)

    print()

    all_subclusters = {}
    for name in template_names:
        vecs_n = normalize(raw_vecs[name])
        c1 = vecs_n.mean(axis=0)
        c1 /= np.linalg.norm(c1) + 1e-10
        spread_1 = 1.0 - float((vecs_n @ c1).mean())

        best_centroids = [c1]
        best_spread = spread_1

        if len(vecs_n) >= 4:
            for k in range(2, min(max_k+1, len(vecs_n))):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(vecs_n)
                sub_c = []
                sub_spreads = []
                for ci in range(k):
                    m = labels == ci
                    if m.sum() == 0: continue
                    c = vecs_n[m].mean(axis=0)
                    c /= np.linalg.norm(c) + 1e-10
                    sub_c.append(c)
                    sub_spreads.append(1.0 - float((vecs_n[m] @ c).mean()))
                avg = np.mean(sub_spreads)
                if spread_1 - avg > min_gain and avg < best_spread:
                    best_centroids = sub_c
                    best_spread = avg

        all_subclusters[name] = best_centroids

    total_c = sum(len(v) for v in all_subclusters.values())
    print(f"  {total_c} sub-centroids across {len(template_names)} templates")
    return all_subclusters, template_names


def classify(q_vec, all_subclusters, template_names):
    """Classify a Q vector. Returns [(template, sub_id, cosine), ...] sorted."""
    q_flat = q_vec.flatten()
    q_n = q_flat / (np.linalg.norm(q_flat) + 1e-10)

    results = []
    for name in template_names:
        for sid, centroid in enumerate(all_subclusters[name]):
            sim = float(q_n @ centroid)
            results.append((name, sid, sim))

    results.sort(key=lambda x: x[2], reverse=True)

    # Deduplicate by template
    seen = set()
    deduped = []
    for name, sid, sim in results:
        if name not in seen:
            seen.add(name)
            deduped.append((name, sid, sim))

    return deduped


# ---- Threshold analysis -------------------------------------------------

def run_threshold_analysis(all_scores):
    """
    For each threshold, compute coverage (% routed) and precision (% correct among routed).
    """
    thresholds = np.arange(0.50, 1.00, 0.01)

    print(f"\n{'='*70}")
    print(f"CONFIDENCE THRESHOLD SWEEP")
    print(f"{'='*70}")
    print(f"\n  {'Threshold':>9s}  {'Coverage':>8s}  {'Precision':>9s}  {'Routed':>6s}  {'Correct':>7s}  {'Wrong':>5s}  {'Skipped':>7s}")
    print(f"  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*7}")

    results = []
    best_threshold = None
    best_coverage_at_100 = 0

    for t in thresholds:
        n_routed = 0
        n_correct = 0
        n_wrong = 0
        n_skipped = 0

        for score in all_scores:
            top1_sim = score["top1_sim"]
            top1_correct = score["top1_correct"]

            if top1_sim >= t:
                n_routed += 1
                if top1_correct:
                    n_correct += 1
                else:
                    n_wrong += 1
            else:
                n_skipped += 1

        total = len(all_scores)
        coverage = n_routed / total if total > 0 else 0
        precision = n_correct / n_routed if n_routed > 0 else 1.0

        results.append({
            "threshold": float(t),
            "coverage": coverage,
            "precision": precision,
            "n_routed": n_routed,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_skipped": n_skipped,
        })

        # Track best threshold with 100% precision
        if precision >= 1.0 and coverage > best_coverage_at_100:
            best_coverage_at_100 = coverage
            best_threshold = float(t)

        # Print at key points
        if t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] or \
           (precision >= 1.0 and coverage == best_coverage_at_100):
            marker = " <--" if precision >= 1.0 and coverage == best_coverage_at_100 else ""
            print(f"  {t:9.2f}  {coverage:7.0%}   {precision:8.0%}   {n_routed:6d}  {n_correct:7d}  {n_wrong:5d}  {n_skipped:7d}{marker}")

    return results, best_threshold, best_coverage_at_100


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Confidence-gated routing threshold analysis"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = args.layer

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    print(f"  Model: {args.model}, Layer: L{target_layer}")

    # Build sub-centroids
    print("\nBuilding sub-centroids...")
    subclusters, template_names = build_subclusters(model, tokenizer, target_layer)

    # ---- Score all prompt sets ----
    all_prompts = []

    # 1. Training prompts (self-check)
    for name, prompts in DIVERSE_TRAIN.items():
        for prompt in prompts:
            all_prompts.append({"text": prompt, "expected": name, "source": "train"})

    # 2. Stress prompts
    for p in STRESS_PROMPTS:
        all_prompts.append({**p, "source": "stress"})

    # 3. Realistic corpus
    for p in REALISTIC_CORPUS:
        all_prompts.append({**p, "source": "realistic"})

    total = len(all_prompts)
    print(f"\nScoring {total} prompts...")
    t0 = time.time()

    all_scores = []
    for i, p in enumerate(all_prompts):
        prompt = p["text"]
        expected = p["expected"]
        if isinstance(expected, str):
            expected = [expected]
        source = p["source"]

        q = forward_capture_q_at_layer(model, tokenizer, prompt, target_layer)
        ranked = classify(q, subclusters, template_names)

        top1_name = ranked[0][0]
        top1_sim = ranked[0][2]
        top1_correct = top1_name in expected

        top3_names = [r[0] for r in ranked[:3]]
        top3_correct = any(e in top3_names for e in expected)

        gap = ranked[0][2] - ranked[1][2] if len(ranked) > 1 else 0

        all_scores.append({
            "prompt": prompt,
            "expected": expected,
            "source": source,
            "top1": top1_name,
            "top1_sim": top1_sim,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "gap": gap,
            "difficulty": p.get("difficulty", ""),
        })

        print(f"\r  {i+1}/{total} ({time.time()-t0:.0f}s)", end="", flush=True)

    print(f"\n  Done in {time.time()-t0:.0f}s")

    # ---- Analysis per source ----
    for source in ["train", "stress", "realistic"]:
        scores = [s for s in all_scores if s["source"] == source]
        if not scores:
            continue

        print(f"\n{'='*70}")
        print(f"SOURCE: {source.upper()} ({len(scores)} prompts)")
        print(f"{'='*70}")

        # Basic accuracy
        n_top1 = sum(1 for s in scores if s["top1_correct"])
        n_top3 = sum(1 for s in scores if s["top3_correct"])
        print(f"  Top-1: {n_top1}/{len(scores)} ({n_top1/len(scores):.0%})")
        print(f"  Top-3: {n_top3}/{len(scores)} ({n_top3/len(scores):.0%})")

        # Score distribution
        correct_sims = [s["top1_sim"] for s in scores if s["top1_correct"]]
        wrong_sims = [s["top1_sim"] for s in scores if not s["top1_correct"]]

        if correct_sims:
            print(f"  Correct cosines: mean={np.mean(correct_sims):.3f}  "
                  f"min={np.min(correct_sims):.3f}  std={np.std(correct_sims):.3f}")
        if wrong_sims:
            print(f"  Wrong cosines:   mean={np.mean(wrong_sims):.3f}  "
                  f"max={np.max(wrong_sims):.3f}  std={np.std(wrong_sims):.3f}")

        if correct_sims and wrong_sims:
            sep = np.min(correct_sims) - np.max(wrong_sims)
            print(f"  Separation (min_correct - max_wrong): {sep:+.3f}")
            if sep > 0:
                print(f"  -> CLEANLY SEPARABLE at threshold {np.max(wrong_sims):.3f}")
            else:
                print(f"  -> OVERLAP region: {np.max(wrong_sims):.3f} to {np.min(correct_sims):.3f}")

        # Threshold sweep for this source
        results, best_t, best_cov = run_threshold_analysis(scores)

        if best_t is not None:
            print(f"\n  BEST THRESHOLD: {best_t:.2f}")
            print(f"  Coverage at 100% precision: {best_cov:.0%}")
            print(f"  -> {int(best_cov*len(scores))}/{len(scores)} prompts routed with ZERO error")
            print(f"  -> {int((1-best_cov)*len(scores))}/{len(scores)} fall back to neural")

        # Per-difficulty breakdown (for realistic corpus)
        if source == "realistic":
            print(f"\n  Per-difficulty:")
            for diff in ["easy", "medium", "hard", "edge"]:
                d_scores = [s for s in scores if s.get("difficulty") == diff]
                if not d_scores:
                    continue
                n_ok = sum(1 for s in d_scores if s["top1_correct"])
                avg_sim = np.mean([s["top1_sim"] for s in d_scores])
                print(f"    {diff:8s}: {n_ok}/{len(d_scores)} top-1  avg_cos={avg_sim:.3f}")

    # ---- Combined threshold ----
    print(f"\n{'='*70}")
    print(f"COMBINED THRESHOLD (all {len(all_scores)} prompts)")
    print(f"{'='*70}")

    results, best_t, best_cov = run_threshold_analysis(all_scores)

    # ---- Misses detail ----
    if best_t is not None:
        print(f"\n  Prompts that MISS at threshold {best_t:.2f}:")
        wrong_above = [s for s in all_scores if s["top1_sim"] >= best_t and not s["top1_correct"]]
        if wrong_above:
            print(f"  !!! {len(wrong_above)} WRONG ROUTES above threshold:")
            for s in wrong_above:
                print(f"    cos={s['top1_sim']:.3f} pred={s['top1']} expected={s['expected']}")
                print(f"      \"{s['prompt'][:60]}\"")
        else:
            print(f"  Zero wrong routes above threshold.")

        below = [s for s in all_scores if s["top1_sim"] < best_t]
        n_below_correct = sum(1 for s in below if s["top1_correct"])
        print(f"\n  Prompts below threshold (neural fallback): {len(below)}")
        print(f"    Of which correctly classified anyway: {n_below_correct}")
        print(f"    Actually need neural: {len(below) - n_below_correct}")

    # ---- Save ----
    save_data = {
        "layer": target_layer,
        "best_threshold": best_t,
        "best_coverage_at_100_precision": best_cov,
        "total_prompts": len(all_scores),
        "threshold_curve": results,
    }
    with open(output_dir / "confidence_threshold_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if best_t is not None and best_cov >= 0.80:
        print(f"\n  PRODUCTION-READY CONFIDENCE GATING")
        print(f"    Threshold: {best_t:.2f}")
        print(f"    Coverage: {best_cov:.0%} of prompts routed with ZERO error")
        print(f"    Fallback: {1-best_cov:.0%} use neural attention")
        print(f"    -> {best_cov:.0%} of attention computation eliminated")
    elif best_t is not None and best_cov >= 0.60:
        print(f"\n  VIABLE CONFIDENCE GATING")
        print(f"    Threshold: {best_t:.2f}")
        print(f"    Coverage: {best_cov:.0%} at 100% precision")
        print(f"    -> Significant but not dominant attention elimination")
    else:
        cov = best_cov if best_cov else 0
        print(f"\n  CONFIDENCE GATING LIMITED")
        print(f"    Only {cov:.0%} of prompts separable with 100% precision")
        print(f"    Need: better centroids, more training data, or accept some error")

    print()


if __name__ == "__main__":
    main()
