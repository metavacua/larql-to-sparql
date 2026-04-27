#!/usr/bin/env python3
"""
q_centroid_stress.py — Stress-test Q-vector routing with hard prompts

The 16-template centroids get 100% on clean synthetic prompts.
Now test the edges:
  1. Long context (15-30 tokens vs 5-8)
  2. Compositional (multiple relations at once)
  3. Ambiguous frames (underspecified queries)
  4. Natural language (messy real-world phrasing)
  5. Multi-hop (chained relations)

Uses the L21 centroids from the clean training set.
For each stress prompt, reports: nearest centroid, distance, and
whether the correct template is in top-1/top-3.

USAGE:
  python3 experiments/05_syntax_circuit_routing/q_centroid_stress.py \
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


# ---- Clean training prompts (same as q_centroid_full.py) ---------------
# Used to build the 16 centroids

TRAIN_PROMPTS = {
    "capital_of": [
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
    "language_of": [
        "The official language of France is",
        "The official language of Japan is",
        "The official language of Brazil is",
        "The official language of China is",
        "The official language of Germany is",
        "The official language of Russia is",
        "The official language of Italy is",
    ],
    "currency_of": [
        "The currency of Japan is the",
        "The currency of India is the",
        "The currency of Brazil is the",
        "The currency of Mexico is the",
        "The currency of Sweden is the",
        "The currency of Poland is the",
        "The currency of Thailand is the",
    ],
    "continent_of": [
        "France is located in",
        "Japan is located in",
        "Brazil is located in",
        "Nigeria is located in",
        "Australia is located in",
        "Canada is located in",
        "Egypt is located in",
    ],
    "occupation_of": [
        "The occupation of Einstein was",
        "The occupation of Shakespeare was",
        "The occupation of Mozart was",
        "The occupation of Picasso was",
        "The occupation of Darwin was",
        "The occupation of Newton was",
        "The occupation of Beethoven was",
    ],
    "birthplace_of": [
        "Einstein was born in",
        "Shakespeare was born in",
        "Mozart was born in",
        "Picasso was born in",
        "Darwin was born in",
        "Newton was born in",
        "Beethoven was born in",
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
        "Poor means",
        "Strong means",
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
        "The opposite of hard is",
        "The opposite of wet is",
        "The opposite of loud is",
        "The opposite of brave is",
        "The opposite of smooth is",
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
        "Rain is to umbrella as sun is to",
        "North is to south as east is to",
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
        "A oak is a type of",
        "A python is a type of",
        "A hurricane is a type of",
        "A novel is a type of",
        "A sonnet is a type of",
        "A waltz is a type of",
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
        "1000 / 10 =",
        "7 + 8 =",
        "50 - 25 =",
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
    ],
    "causation": [
        "Plants grow because they need",
        "Ice melts because the temperature",
        "Birds fly because they have",
        "People sleep because the body",
        "Fire burns because of",
        "Metal rusts because of",
        "Rain falls because water",
    ],
    "temporal": [
        "World War II ended in",
        "The Roman Empire fell in",
        "The internet was invented in",
        "The first airplane flew in",
        "The moon landing happened in",
        "The Berlin Wall fell in",
        "The printing press was invented in",
    ],
}


# ---- Stress test prompts -----------------------------------------------

STRESS_PROMPTS = {
    "long_context": {
        "desc": "15-30 token prompts with preamble. Same relation, more context.",
        "prompts": [
            {
                "text": "In the early 19th century, the capital of the country that borders Germany to the west is",
                "expected": "capital_of",
                "note": "long preamble, indirect entity reference",
            },
            {
                "text": "After years of research and many publications, the occupation of the famous physicist Albert Einstein was",
                "expected": "occupation_of",
                "note": "20+ tokens, same relation",
            },
            {
                "text": "If you travel to the largest country in South America and ask someone what language they speak, the official language of Brazil is",
                "expected": "language_of",
                "note": "long narrative preamble",
            },
            {
                "text": "Among all the currencies used in Asian countries, the currency of Japan is the",
                "expected": "currency_of",
                "note": "context-setting preamble",
            },
            {
                "text": "Looking at a map of the world and considering the major continents, Australia is located in",
                "expected": "continent_of",
                "note": "descriptive preamble",
            },
        ],
    },
    "compositional": {
        "desc": "Queries involving multiple relations simultaneously.",
        "prompts": [
            {
                "text": "The French-speaking capital of a country in Africa is",
                "expected": ["capital_of", "language_of", "continent_of"],
                "note": "capital + language + continent",
            },
            {
                "text": "The European country whose currency is the krona has its capital in",
                "expected": ["capital_of", "currency_of", "continent_of"],
                "note": "currency -> country -> capital",
            },
            {
                "text": "The birthplace of the famous German composer Beethoven was",
                "expected": ["birthplace_of", "occupation_of"],
                "note": "birthplace + occupation + nationality",
            },
            {
                "text": "A fast animal that is bigger than a dog is a type of",
                "expected": ["hypernym", "comparison"],
                "note": "hypernym + comparison",
            },
            {
                "text": "The opposite of the word that means happy is",
                "expected": ["antonym", "synonym"],
                "note": "antonym of a synonym",
            },
        ],
    },
    "ambiguous": {
        "desc": "Underspecified prompts that could match multiple templates.",
        "prompts": [
            {
                "text": "Paris is",
                "expected": ["capital_of", "continent_of", "birthplace_of"],
                "note": "could be capital, location, or entity description",
            },
            {
                "text": "Mozart was",
                "expected": ["occupation_of", "birthplace_of"],
                "note": "could be occupation, nationality, birthplace",
            },
            {
                "text": "Japan is",
                "expected": ["continent_of", "capital_of", "language_of"],
                "note": "could be location, or start of many queries",
            },
            {
                "text": "Light is",
                "expected": ["comparison", "synonym", "hypernym"],
                "note": "comparison? definition? property?",
            },
            {
                "text": "Python is",
                "expected": ["hypernym", "code_python"],
                "note": "snake or programming language?",
            },
        ],
    },
    "natural_language": {
        "desc": "Messy real-world phrasings of clean template queries.",
        "prompts": [
            {
                "text": "So what money do they use in Japan anyway?",
                "expected": "currency_of",
                "note": "currency query, colloquial",
            },
            {
                "text": "What language do people speak in Brazil?",
                "expected": "language_of",
                "note": "language query, question form",
            },
            {
                "text": "Where was Einstein from originally?",
                "expected": "birthplace_of",
                "note": "birthplace query, question form",
            },
            {
                "text": "What did Picasso do for a living?",
                "expected": "occupation_of",
                "note": "occupation query, colloquial",
            },
            {
                "text": "What's another word for happy?",
                "expected": "synonym",
                "note": "synonym query, question form",
            },
            {
                "text": "What's the opposite of strong?",
                "expected": "antonym",
                "note": "antonym query, question form",
            },
            {
                "text": "Which continent is Nigeria on?",
                "expected": "continent_of",
                "note": "continent query, question form",
            },
            {
                "text": "Tell me the capital city of Thailand.",
                "expected": "capital_of",
                "note": "capital query, imperative",
            },
            {
                "text": "When did the French Revolution start?",
                "expected": "temporal",
                "note": "temporal query, question form",
            },
            {
                "text": "Is a whale a kind of fish or mammal?",
                "expected": "hypernym",
                "note": "hypernym query, conversational",
            },
        ],
    },
    "multi_hop": {
        "desc": "Chained relations requiring multi-step reasoning.",
        "prompts": [
            {
                "text": "The currency of the country where Einstein was born is the",
                "expected": ["currency_of", "birthplace_of"],
                "note": "birthplace -> country -> currency",
            },
            {
                "text": "The language spoken in the capital of Japan is",
                "expected": ["language_of", "capital_of"],
                "note": "capital -> city -> language",
            },
            {
                "text": "The continent where the birthplace of Mozart is located is",
                "expected": ["continent_of", "birthplace_of"],
                "note": "birthplace -> city -> continent",
            },
            {
                "text": "The occupation of the person who was born in Stratford-upon-Avon was",
                "expected": ["occupation_of", "birthplace_of"],
                "note": "birthplace(reverse) -> person -> occupation",
            },
            {
                "text": "A word that means the opposite of the synonym of sad is",
                "expected": ["antonym", "synonym"],
                "note": "synonym -> word -> antonym",
            },
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


def forward_capture_q_at_layer(model, tokenizer, prompt, target_layer):
    """Forward pass capturing Q vector at one specific layer."""
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

    # Prediction
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
        description="Stress-test Q-vector routing with hard prompts"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--layer", type=int, default=21,
                        help="Layer to extract Q from (default: 21)")
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

    # ---- Build centroids from clean training data ----
    print(f"\nBuilding centroids from clean training data...")
    template_names = list(TRAIN_PROMPTS.keys())
    template_to_id = {n: i for i, n in enumerate(template_names)}

    centroid_vecs = {}  # template_name -> list of Q vectors
    for name, prompts in TRAIN_PROMPTS.items():
        vecs = []
        for prompt in prompts:
            q, pred, ntok = forward_capture_q_at_layer(
                model, tokenizer, prompt, target_layer
            )
            vecs.append(q.flatten())
        centroid_vecs[name] = np.stack(vecs)

    # Compute centroids (L2-normalized)
    centroids = {}
    for name, vecs in centroid_vecs.items():
        vecs_n = normalize(vecs)
        c = vecs_n.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        centroids[name] = c

    C = np.stack([centroids[n] for n in template_names])  # [16, 2048]
    print(f"  Built {len(centroids)} centroids, dim={C.shape[1]}")

    # Also compute per-centroid spread (for threshold calibration)
    centroid_spreads = {}
    for name, vecs in centroid_vecs.items():
        vecs_n = normalize(vecs)
        sims = vecs_n @ centroids[name]
        centroid_spreads[name] = {
            "mean": float(sims.mean()),
            "min": float(sims.min()),
            "std": float(sims.std()),
        }

    # ---- Run stress tests ----
    total = sum(len(cat["prompts"]) for cat in STRESS_PROMPTS.values())
    print(f"\nRunning {total} stress prompts...")

    all_results = {}
    n = 0
    t0 = time.time()

    # Category-level stats
    cat_stats = {}

    for cat_name, cat_info in STRESS_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"{cat_name.upper()}: {cat_info['desc']}")
        print(f"{'='*70}")

        cat_results = []
        n_top1 = 0
        n_top3 = 0
        n_any_match = 0

        for p in cat_info["prompts"]:
            prompt = p["text"]
            expected = p["expected"]
            note = p["note"]

            # Normalize expected to list
            if isinstance(expected, str):
                expected = [expected]

            q, pred, ntok = forward_capture_q_at_layer(
                model, tokenizer, prompt, target_layer
            )
            q_flat = q.flatten()
            q_n = q_flat / (np.linalg.norm(q_flat) + 1e-10)

            # Cosine similarity to all centroids
            sims = q_n @ C.T  # [16]
            ranked = np.argsort(-sims)
            top1_name = template_names[ranked[0]]
            top3_names = [template_names[ranked[i]] for i in range(3)]
            top5_names = [template_names[ranked[i]] for i in range(5)]

            # Check matches
            top1_hit = top1_name in expected
            top3_hit = any(e in top3_names for e in expected)
            any_top5_hit = any(e in top5_names for e in expected)

            if top1_hit:
                n_top1 += 1
            if top3_hit:
                n_top3 += 1
            if any_top5_hit:
                n_any_match += 1

            # Cosine to each expected centroid
            expected_sims = {}
            for e in expected:
                if e in centroids:
                    expected_sims[e] = float(q_n @ centroids[e])

            result = {
                "prompt": prompt,
                "expected": expected,
                "note": note,
                "n_tokens": ntok,
                "prediction": pred,
                "top1": top1_name,
                "top1_sim": float(sims[ranked[0]]),
                "top3": top3_names,
                "top3_sims": [float(sims[ranked[i]]) for i in range(3)],
                "top5": top5_names,
                "expected_sims": expected_sims,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
            cat_results.append(result)

            # Print detail
            marker = "OK" if top1_hit else ("top3" if top3_hit else "MISS")
            print(f"\n  [{marker:4s}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
            print(f"         {ntok} tokens, predicts: {pred}")
            print(f"         expected: {expected}")
            print(f"         top-1: {top1_name} (cos={sims[ranked[0]]:.4f})")
            print(f"         top-3: ", end="")
            for i in range(3):
                name = template_names[ranked[i]]
                sim = sims[ranked[i]]
                hit = "*" if name in expected else " "
                print(f"{hit}{name}({sim:.3f})  ", end="")
            print()

            # Show expected centroids' sims
            for e, s in expected_sims.items():
                rank = list(template_names[r] for r in ranked).index(e) + 1 if e in template_names else "?"
                print(f"         {e}: cos={s:.4f} (rank {rank})")

            # Gap between top-1 and top-2
            gap = sims[ranked[0]] - sims[ranked[1]]
            print(f"         gap (top1-top2): {gap:.4f}")

            n += 1

        n_prompts = len(cat_info["prompts"])
        cat_stats[cat_name] = {
            "n": n_prompts,
            "top1_acc": n_top1 / n_prompts,
            "top3_acc": n_top3 / n_prompts,
            "any_top5": n_any_match / n_prompts,
        }

        print(f"\n  Summary: top-1={n_top1}/{n_prompts} ({n_top1/n_prompts:.0%})  "
              f"top-3={n_top3}/{n_prompts} ({n_top3/n_prompts:.0%})  "
              f"any-top-5={n_any_match}/{n_prompts} ({n_any_match/n_prompts:.0%})")

        all_results[cat_name] = cat_results

    print(f"\n\nDone in {time.time()-t0:.0f}s")

    # ---- Overall summary ----
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Category':<20s} {'N':>3s}  {'Top-1':>6s}  {'Top-3':>6s}  {'Any-5':>6s}")
    print(f"  {'-'*20} {'---':>3s}  {'------':>6s}  {'------':>6s}  {'------':>6s}")

    total_n = 0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0

    for cat, stats in cat_stats.items():
        n = stats["n"]
        t1 = stats["top1_acc"]
        t3 = stats["top3_acc"]
        t5 = stats["any_top5"]
        total_n += n
        total_top1 += int(t1 * n)
        total_top3 += int(t3 * n)
        total_top5 += int(t5 * n)
        print(f"  {cat:<20s} {n:3d}  {t1:5.0%}   {t3:5.0%}   {t5:5.0%}")

    overall_top1 = total_top1 / total_n if total_n else 0
    overall_top3 = total_top3 / total_n if total_n else 0
    overall_top5 = total_top5 / total_n if total_n else 0

    print(f"  {'OVERALL':<20s} {total_n:3d}  {overall_top1:5.0%}   {overall_top3:5.0%}   {overall_top5:5.0%}")

    # ---- Centroid spread context ----
    print(f"\n  Reference: clean training centroid spreads:")
    for name in template_names:
        s = centroid_spreads[name]
        print(f"    {name:20s}: mean={s['mean']:.4f}  min={s['min']:.4f}  std={s['std']:.3f}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if overall_top1 >= 0.8:
        print(f"\n  ROUTING ROBUST TO VARIATION")
        print(f"    {overall_top1:.0%} top-1 accuracy on stress prompts")
        print(f"    16 centroids handle context length, natural language, composition")
    elif overall_top3 >= 0.8:
        print(f"\n  ROUTING WORKS WITH TOP-3 FALLBACK")
        print(f"    Top-1: {overall_top1:.0%}, Top-3: {overall_top3:.0%}")
        print(f"    Need to check top-3 centroids, not just nearest")
        print(f"    Still viable: try 3 cached patterns, pick best match")
    elif overall_top5 >= 0.8:
        print(f"\n  ROUTING NEEDS MORE CENTROIDS")
        print(f"    Signal present but 16 centroids too coarse for real text")
        print(f"    Try: more templates, or cluster natural-language Q vectors separately")
    else:
        print(f"\n  ROUTING BREAKS ON HARD PROMPTS")
        print(f"    Clean templates work, real text doesn't")
        print(f"    The Q-vector signal is template-dependent, not relation-dependent")
        print(f"    Need: train centroids on diverse phrasings, not just templates")

    # Per-category verdict
    print(f"\n  Per-category breakdown:")
    for cat, stats in cat_stats.items():
        status = "OK" if stats["top1_acc"] >= 0.6 else "WEAK" if stats["top3_acc"] >= 0.6 else "FAIL"
        print(f"    {cat:<20s}: {status}")

    print()

    # ---- Save ----
    save_data = {
        "layer": target_layer,
        "cat_stats": cat_stats,
        "overall": {
            "top1": overall_top1,
            "top3": overall_top3,
            "top5": overall_top5,
            "n": total_n,
        },
        "centroid_spreads": centroid_spreads,
        "results": all_results,
    }
    with open(output_dir / "q_centroid_stress_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {output_dir / 'q_centroid_stress_results.json'}")


if __name__ == "__main__":
    main()
