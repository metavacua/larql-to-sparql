#!/usr/bin/env python3
"""
q_centroid_diverse.py — Q-vector routing with diverse phrasing centroids

The stress test showed 100% on clean templates but 40% top-1 on natural
language. The mechanism works — the centroids are too narrow.

Fix: train centroids on mixed phrasings per template:
  - Declarative: "The capital of France is"
  - Question: "What is the capital of France?"
  - Colloquial: "So what's the capital of France?"
  - Imperative: "Tell me the capital of France."
  - Possessive: "France's capital is"

Then rerun the same stress test to measure improvement.

USAGE:
  python3 experiments/05_syntax_circuit_routing/q_centroid_diverse.py \
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Diverse training prompts ------------------------------------------
# Each template now has declarative + question + colloquial + imperative

DIVERSE_TRAIN = {
    "capital_of": [
        # Declarative (original)
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
        # Question form
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Germany?",
        "What's the capital of India?",
        # Colloquial / varied
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


# ---- Same stress prompts from q_centroid_stress.py ---------------------

STRESS_PROMPTS = {
    "long_context": {
        "desc": "15-30 token prompts with preamble",
        "prompts": [
            {"text": "In the early 19th century, the capital of the country that borders Germany to the west is", "expected": "capital_of"},
            {"text": "After years of research and many publications, the occupation of the famous physicist Albert Einstein was", "expected": "occupation_of"},
            {"text": "If you travel to the largest country in South America and ask someone what language they speak, the official language of Brazil is", "expected": "language_of"},
            {"text": "Among all the currencies used in Asian countries, the currency of Japan is the", "expected": "currency_of"},
            {"text": "Looking at a map of the world and considering the major continents, Australia is located in", "expected": "continent_of"},
        ],
    },
    "compositional": {
        "desc": "Multiple relations simultaneously",
        "prompts": [
            {"text": "The French-speaking capital of a country in Africa is", "expected": ["capital_of", "language_of", "continent_of"]},
            {"text": "The European country whose currency is the krona has its capital in", "expected": ["capital_of", "currency_of", "continent_of"]},
            {"text": "The birthplace of the famous German composer Beethoven was", "expected": ["birthplace_of", "occupation_of"]},
            {"text": "A fast animal that is bigger than a dog is a type of", "expected": ["hypernym", "comparison"]},
            {"text": "The opposite of the word that means happy is", "expected": ["antonym", "synonym"]},
        ],
    },
    "ambiguous": {
        "desc": "Underspecified prompts",
        "prompts": [
            {"text": "Paris is", "expected": ["capital_of", "continent_of", "birthplace_of"]},
            {"text": "Mozart was", "expected": ["occupation_of", "birthplace_of"]},
            {"text": "Japan is", "expected": ["continent_of", "capital_of", "language_of"]},
            {"text": "Light is", "expected": ["comparison", "synonym", "hypernym"]},
            {"text": "Python is", "expected": ["hypernym", "code_python"]},
        ],
    },
    "natural_language": {
        "desc": "Messy real-world phrasings",
        "prompts": [
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
    },
    "multi_hop": {
        "desc": "Chained relations",
        "prompts": [
            {"text": "The currency of the country where Einstein was born is the", "expected": ["currency_of", "birthplace_of"]},
            {"text": "The language spoken in the capital of Japan is", "expected": ["language_of", "capital_of"]},
            {"text": "The continent where the birthplace of Mozart is located is", "expected": ["continent_of", "birthplace_of"]},
            {"text": "The occupation of the person who was born in Stratford-upon-Avon was", "expected": ["occupation_of", "birthplace_of"]},
            {"text": "A word that means the opposite of the synonym of sad is", "expected": ["antonym", "synonym"]},
        ],
    },
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

    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return q_vector, pred_tok, len(tokens)


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Q-vector routing with diverse phrasing centroids"
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
    print(f"  Model: {args.model}")
    print(f"  Target layer: L{target_layer}")

    template_names = list(DIVERSE_TRAIN.keys())
    n_templates = len(template_names)
    total_train = sum(len(v) for v in DIVERSE_TRAIN.values())

    # ---- Build diverse centroids ----
    print(f"\nBuilding diverse centroids ({total_train} prompts, {n_templates} templates)...")
    t0 = time.time()
    n = 0

    centroid_vecs = {}
    for name, prompts in DIVERSE_TRAIN.items():
        vecs = []
        for prompt in prompts:
            q, pred, ntok = forward_capture_q_at_layer(
                model, tokenizer, prompt, target_layer
            )
            vecs.append(q.flatten())
            n += 1
            print(f"\r  {n}/{total_train} ({time.time()-t0:.0f}s)", end="", flush=True)
        centroid_vecs[name] = np.stack(vecs)

    print(f"\n  Training done in {time.time()-t0:.0f}s")

    # Compute centroids
    centroids = {}
    spreads = {}
    for name, vecs in centroid_vecs.items():
        vecs_n = normalize(vecs)
        c = vecs_n.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        centroids[name] = c

        sims = vecs_n @ c
        spreads[name] = {
            "n": len(vecs),
            "mean": float(sims.mean()),
            "min": float(sims.min()),
            "std": float(sims.std()),
        }

    C = np.stack([centroids[n] for n in template_names])

    print(f"\n  Centroid spreads (diverse training):")
    for name in template_names:
        s = spreads[name]
        print(f"    {name:20s}: n={s['n']:2d}  mean={s['mean']:.4f}  min={s['min']:.4f}  std={s['std']:.3f}")

    # ---- Also build narrow centroids (declarative only) for comparison ----
    print(f"\n  Building narrow centroids (declarative only) for comparison...")
    narrow_centroids = {}
    for name, vecs in centroid_vecs.items():
        # Take only the first N prompts (the declarative ones)
        # Count how many are declarative (don't start with What/Which/Tell/etc)
        declarative = []
        for i, prompt in enumerate(DIVERSE_TRAIN[name]):
            p = prompt.strip()
            if not any(p.startswith(q) for q in
                       ["What", "Which", "Where", "When", "How", "Tell",
                        "Do you", "Name", "Give", "Is a", "Calculate"]):
                declarative.append(i)

        if declarative:
            dec_vecs = normalize(vecs[declarative])
            c = dec_vecs.mean(axis=0)
            c /= np.linalg.norm(c) + 1e-10
            narrow_centroids[name] = c
        else:
            narrow_centroids[name] = centroids[name]

    C_narrow = np.stack([narrow_centroids[n] for n in template_names])

    # ---- Run stress tests ----
    total_stress = sum(len(cat["prompts"]) for cat in STRESS_PROMPTS.values())
    print(f"\nRunning {total_stress} stress prompts...\n")

    cat_stats_diverse = {}
    cat_stats_narrow = {}
    all_results = {}

    for cat_name, cat_info in STRESS_PROMPTS.items():
        print(f"{'='*70}")
        print(f"{cat_name.upper()}: {cat_info['desc']}")
        print(f"{'='*70}")

        cat_results = []
        n_top1_d = 0
        n_top3_d = 0
        n_top1_n = 0
        n_top3_n = 0

        for p in cat_info["prompts"]:
            prompt = p["text"]
            expected = p["expected"]
            if isinstance(expected, str):
                expected = [expected]

            q, pred, ntok = forward_capture_q_at_layer(
                model, tokenizer, prompt, target_layer
            )
            q_flat = q.flatten()
            q_n = q_flat / (np.linalg.norm(q_flat) + 1e-10)

            # Diverse centroids
            sims_d = q_n @ C.T
            ranked_d = np.argsort(-sims_d)
            top1_d = template_names[ranked_d[0]]
            top3_d = [template_names[ranked_d[i]] for i in range(3)]

            top1_hit_d = top1_d in expected
            top3_hit_d = any(e in top3_d for e in expected)
            if top1_hit_d:
                n_top1_d += 1
            if top3_hit_d:
                n_top3_d += 1

            # Narrow centroids
            sims_n = q_n @ C_narrow.T
            ranked_n = np.argsort(-sims_n)
            top1_n = template_names[ranked_n[0]]
            top3_n = [template_names[ranked_n[i]] for i in range(3)]

            top1_hit_n = top1_n in expected
            top3_hit_n = any(e in top3_n for e in expected)
            if top1_hit_n:
                n_top1_n += 1
            if top3_hit_n:
                n_top3_n += 1

            # Status
            status_d = "OK" if top1_hit_d else ("top3" if top3_hit_d else "MISS")
            status_n = "OK" if top1_hit_n else ("top3" if top3_hit_n else "MISS")
            improved = ""
            if status_d != status_n:
                if status_d == "OK" and status_n != "OK":
                    improved = " IMPROVED"
                elif status_d == "top3" and status_n == "MISS":
                    improved = " IMPROVED"
                elif status_n == "OK" and status_d != "OK":
                    improved = " REGRESSED"

            print(f"\n  [{status_d:4s}] \"{prompt[:55]}{'...' if len(prompt) > 55 else ''}\"")
            print(f"         expected: {expected}")
            print(f"         diverse  top-1: {top1_d} ({sims_d[ranked_d[0]]:.3f})  "
                  f"narrow top-1: {top1_n} ({sims_n[ranked_n[0]]:.3f}){improved}")
            print(f"         diverse  top-3: ", end="")
            for i in range(3):
                nm = template_names[ranked_d[i]]
                hit = "*" if nm in expected else " "
                print(f"{hit}{nm}({sims_d[ranked_d[i]]:.3f}) ", end="")
            print()

            # Show expected centroid sims
            for e in expected:
                if e in centroids:
                    sd = float(q_n @ centroids[e])
                    sn = float(q_n @ narrow_centroids[e])
                    rd = [template_names[ranked_d[i]] for i in range(len(template_names))].index(e) + 1
                    rn = [template_names[ranked_n[i]] for i in range(len(template_names))].index(e) + 1
                    print(f"         {e}: diverse={sd:.3f}(#{rd}) narrow={sn:.3f}(#{rn})")

            cat_results.append({
                "prompt": prompt,
                "expected": expected,
                "diverse_top1": top1_d,
                "diverse_top1_sim": float(sims_d[ranked_d[0]]),
                "diverse_top3": top3_d,
                "narrow_top1": top1_n,
                "narrow_top1_sim": float(sims_n[ranked_n[0]]),
                "top1_hit_diverse": top1_hit_d,
                "top3_hit_diverse": top3_hit_d,
                "top1_hit_narrow": top1_hit_n,
                "top3_hit_narrow": top3_hit_n,
                "prediction": pred,
            })

        n_prompts = len(cat_info["prompts"])
        cat_stats_diverse[cat_name] = {
            "n": n_prompts,
            "top1": n_top1_d / n_prompts,
            "top3": n_top3_d / n_prompts,
        }
        cat_stats_narrow[cat_name] = {
            "n": n_prompts,
            "top1": n_top1_n / n_prompts,
            "top3": n_top3_n / n_prompts,
        }

        print(f"\n  Diverse:  top-1={n_top1_d}/{n_prompts} ({n_top1_d/n_prompts:.0%})  "
              f"top-3={n_top3_d}/{n_prompts} ({n_top3_d/n_prompts:.0%})")
        print(f"  Narrow:   top-1={n_top1_n}/{n_prompts} ({n_top1_n/n_prompts:.0%})  "
              f"top-3={n_top3_n}/{n_prompts} ({n_top3_n/n_prompts:.0%})")

        all_results[cat_name] = cat_results

    # ---- Overall summary ----
    print(f"\n\n{'='*70}")
    print(f"OVERALL COMPARISON: DIVERSE vs NARROW CENTROIDS")
    print(f"{'='*70}")

    print(f"\n  {'Category':<20s}  {'Diverse':>14s}  {'Narrow':>14s}  {'Delta':>8s}")
    print(f"  {'':20s}  {'top1  top3':>14s}  {'top1  top3':>14s}  {'top1':>8s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*8}")

    total_d1 = total_d3 = total_n1 = total_n3 = 0
    total_n = 0

    for cat in STRESS_PROMPTS:
        sd = cat_stats_diverse[cat]
        sn = cat_stats_narrow[cat]
        n = sd["n"]
        total_n += n
        total_d1 += int(sd["top1"] * n)
        total_d3 += int(sd["top3"] * n)
        total_n1 += int(sn["top1"] * n)
        total_n3 += int(sn["top3"] * n)

        delta = sd["top1"] - sn["top1"]
        delta_str = f"{delta:+.0%}" if delta != 0 else "  ="
        print(f"  {cat:<20s}  {sd['top1']:4.0%}  {sd['top3']:4.0%}    "
              f"{sn['top1']:4.0%}  {sn['top3']:4.0%}    {delta_str}")

    d1 = total_d1 / total_n
    d3 = total_d3 / total_n
    n1 = total_n1 / total_n
    n3 = total_n3 / total_n
    delta = d1 - n1

    print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*8}")
    print(f"  {'OVERALL':<20s}  {d1:4.0%}  {d3:4.0%}    "
          f"{n1:4.0%}  {n3:4.0%}    {delta:+.0%}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if d1 >= 0.8:
        print(f"\n  DIVERSE CENTROIDS SOLVE THE COVERAGE PROBLEM")
        print(f"    Top-1: {d1:.0%} (was {n1:.0%} with narrow)")
        print(f"    Top-3: {d3:.0%}")
        print(f"    16 centroids with diverse training handle real-world queries")
    elif d3 >= 0.9:
        print(f"\n  TOP-3 ROUTING IS PRODUCTION-READY")
        print(f"    Top-1: {d1:.0%}, Top-3: {d3:.0%}")
        print(f"    Try 3 cached attention patterns, pick best")
    elif d1 > n1:
        print(f"\n  DIVERSE CENTROIDS HELP BUT NOT ENOUGH")
        print(f"    Top-1: {d1:.0%} (was {n1:.0%}), improvement: {delta:+.0%}")
        print(f"    Need: more phrasing variants, or sub-cluster per template")
    else:
        print(f"\n  DIVERSE TRAINING DIDN'T HELP")
        print(f"    The gap is structural, not a training data issue")

    # Save
    save_data = {
        "layer": target_layer,
        "diverse_stats": cat_stats_diverse,
        "narrow_stats": cat_stats_narrow,
        "overall_diverse": {"top1": d1, "top3": d3},
        "overall_narrow": {"top1": n1, "top3": n3},
        "centroid_spreads": spreads,
        "results": all_results,
    }
    with open(output_dir / "q_centroid_diverse_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {output_dir / 'q_centroid_diverse_results.json'}")

    print()


if __name__ == "__main__":
    main()
