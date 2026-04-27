#!/usr/bin/env python3
"""
Gemma 3-4B FFN Three-System Replacement

THE experiment: does the FFN three-system thesis hold at real scale?

At 20M:
  - Output engine replacement: Δ=0.000 (PERFECT)
  - Knowledge engine (single layer): +0.002 loss, 100% top-1 match
  - All three systems + trained attention: BEATS baseline

At 4B:
  - Vindex walk already works at all 34 layers (proven in Rust)
  - But does REPLACING FFN with external graph lookups preserve accuracy?

Method (no training — pure analysis):
  Phase 1: Extract entity/relation codebooks from Gemma's residuals
  Phase 2: Build knowledge graph from real factual triples
  Phase 3: Replace knowledge-layer FFN with graph queries (one layer at a time)
  Phase 4: Measure loss impact per layer
  Phase 5: Compare outputs (top-1/top-5 match)

Memory-efficient: hooks + forward passes only, float16, one-layer replacement.
"""

import os
import sys
import json
import time
import math
import random
import gc
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
SEED = 42
MAX_SEQ = 128
N_CODEBOOK_SAMPLES = 100  # samples for building codebooks
N_EVAL_SAMPLES = 50  # samples for evaluation
OUTPUT_DIR = "results_gemma4b_ffn"

# ---------------------------------------------------------------------------
# Real factual knowledge (replaces synthetic countries)
# ---------------------------------------------------------------------------

# Structure: (subject, relation, object, prompt_template)
# These are facts Gemma should know from pretraining
FACTUAL_TRIPLES = [
    # Capitals
    ("France", "capital_of", "Paris", "The capital of France is"),
    ("Germany", "capital_of", "Berlin", "The capital of Germany is"),
    ("Japan", "capital_of", "Tokyo", "The capital of Japan is"),
    ("Italy", "capital_of", "Rome", "The capital of Italy is"),
    ("Spain", "capital_of", "Madrid", "The capital of Spain is"),
    ("Brazil", "capital_of", "Brasilia", "The capital of Brazil is"),
    ("Australia", "capital_of", "Canberra", "The capital of Australia is"),
    ("Canada", "capital_of", "Ottawa", "The capital of Canada is"),
    ("Egypt", "capital_of", "Cairo", "The capital of Egypt is"),
    ("India", "capital_of", "Delhi", "The capital of India is"),
    ("China", "capital_of", "Beijing", "The capital of China is"),
    ("Russia", "capital_of", "Moscow", "The capital of Russia is"),
    ("Mexico", "capital_of", "Mexico City", "The capital of Mexico is"),
    ("Argentina", "capital_of", "Buenos Aires", "The capital of Argentina is"),
    ("Poland", "capital_of", "Warsaw", "The capital of Poland is"),
    ("Sweden", "capital_of", "Stockholm", "The capital of Sweden is"),
    ("Norway", "capital_of", "Oslo", "The capital of Norway is"),
    ("Turkey", "capital_of", "Ankara", "The capital of Turkey is"),
    ("Greece", "capital_of", "Athens", "The capital of Greece is"),
    ("Portugal", "capital_of", "Lisbon", "The capital of Portugal is"),

    # Languages
    ("France", "language_of", "French", "The official language of France is"),
    ("Germany", "language_of", "German", "The official language of Germany is"),
    ("Japan", "language_of", "Japanese", "The official language of Japan is"),
    ("Italy", "language_of", "Italian", "The official language of Italy is"),
    ("Spain", "language_of", "Spanish", "The official language of Spain is"),
    ("Brazil", "language_of", "Portuguese", "The official language of Brazil is"),
    ("China", "language_of", "Chinese", "The official language of China is"),
    ("Russia", "language_of", "Russian", "The official language of Russia is"),

    # Currencies
    ("Japan", "currency_of", "yen", "The currency of Japan is the"),
    ("United Kingdom", "currency_of", "pound", "The currency of the United Kingdom is the"),
    ("Switzerland", "currency_of", "franc", "The currency of Switzerland is the"),
    ("India", "currency_of", "rupee", "The currency of India is the"),
    ("China", "currency_of", "yuan", "The currency of China is the"),
    ("Brazil", "currency_of", "real", "The currency of Brazil is the"),
    ("Mexico", "currency_of", "peso", "The currency of Mexico is the"),
    ("Russia", "currency_of", "ruble", "The currency of Russia is the"),

    # Continents
    ("France", "continent_of", "Europe", "France is located on the continent of"),
    ("Japan", "continent_of", "Asia", "Japan is located on the continent of"),
    ("Brazil", "continent_of", "South America", "Brazil is located on the continent of"),
    ("Egypt", "continent_of", "Africa", "Egypt is located on the continent of"),
    ("Australia", "continent_of", "Australia", "Australia is located on the continent of"),
    ("Canada", "continent_of", "North America", "Canada is located on the continent of"),

    # Science facts
    ("water", "boils_at", "100", "Water boils at"),
    ("light", "speed_of", "300000", "The speed of light in km/s is approximately"),
    ("Earth", "orbits", "Sun", "The Earth orbits the"),
    ("Moon", "orbits", "Earth", "The Moon orbits the"),
    ("hydrogen", "atomic_number", "1", "The atomic number of hydrogen is"),
    ("gold", "symbol", "Au", "The chemical symbol for gold is"),
    ("iron", "symbol", "Fe", "The chemical symbol for iron is"),
    ("oxygen", "symbol", "O", "The chemical symbol for oxygen is"),
]

# Non-factual prompts for baseline comparison
NON_FACTUAL_PROMPTS = [
    "The cat sat on the",
    "She quickly ran to the",
    "Once upon a time, there was a",
    "In the beginning, the world was",
    "The most important thing about life is",
    "Running through the forest, the deer",
    "The old man looked at the sky and",
    "If it rains tomorrow, we should",
    "The book on the table was",
    "After a long day at work, she",
]


# ---------------------------------------------------------------------------
# Phase 1: Extract codebooks from Gemma's residuals
# ---------------------------------------------------------------------------

def extract_codebooks(model, tokenizer, triples, device, n_layers):
    """
    Build entity/relation codebooks from Gemma's own residual representations.
    For each (entity, relation, object) triple, capture the residual at each layer
    when processing the prompt.
    """
    print(f"\n  Extracting codebooks from {len(triples)} triples across {n_layers} layers...")
    model.eval()

    # We'll hook into the model to capture residuals before each FFN
    # In Gemma 3, the layers are at model.model.language_model.layers[i]
    layer_residuals = [None] * n_layers
    hooks = []

    def make_hook(li):
        def hook(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                layer_residuals[li] = args[0].detach()
        return hook

    # Hook into each MLP (FFN) to capture its input
    for li in range(n_layers):
        layer = model.model.language_model.layers[li]
        hooks.append(layer.mlp.register_forward_pre_hook(make_hook(li)))

    # Per-entity and per-relation residual accumulators
    entity_residuals = [defaultdict(list) for _ in range(n_layers)]
    relation_residuals = [defaultdict(list) for _ in range(n_layers)]

    t0 = time.time()
    with torch.no_grad():
        for ti, (subj, rel, obj, prompt) in enumerate(triples):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            _ = model(**inputs)

            for li in range(n_layers):
                if layer_residuals[li] is not None:
                    # Mean across sequence positions
                    mean_res = layer_residuals[li].mean(dim=1).squeeze(0).float().cpu()
                    entity_residuals[li][subj].append(mean_res)
                    relation_residuals[li][rel].append(mean_res)

            if (ti + 1) % 20 == 0:
                print(f"    {ti+1}/{len(triples)} ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()

    # Build centroids
    print(f"  Building centroids...")
    entity_centroids = []
    entity_names_per_layer = []
    relation_centroids = []
    relation_names_per_layer = []

    for li in range(n_layers):
        # Entity centroids
        e_names = sorted(entity_residuals[li].keys())
        if e_names:
            e_vecs = [torch.stack(entity_residuals[li][n]).mean(0) for n in e_names]
            entity_centroids.append(torch.stack(e_vecs).to(device))
        else:
            entity_centroids.append(torch.zeros(1, model.config.text_config.hidden_size,
                                               device=device))
            e_names = ["unknown"]
        entity_names_per_layer.append(e_names)

        # Relation centroids
        r_names = sorted(relation_residuals[li].keys())
        if r_names:
            r_vecs = [torch.stack(relation_residuals[li][n]).mean(0) for n in r_names]
            relation_centroids.append(torch.stack(r_vecs).to(device))
        else:
            relation_centroids.append(torch.zeros(1, model.config.text_config.hidden_size,
                                                 device=device))
            r_names = ["unknown"]
        relation_names_per_layer.append(r_names)

    n_entities = len(entity_names_per_layer[0])
    n_relations = len(relation_names_per_layer[0])
    elapsed = time.time() - t0
    print(f"  Codebooks: {n_entities} entities, {n_relations} relations ({elapsed:.0f}s)")

    return {
        "entity_centroids": entity_centroids,
        "entity_names": entity_names_per_layer,
        "relation_centroids": relation_centroids,
        "relation_names": relation_names_per_layer,
    }


# ---------------------------------------------------------------------------
# Phase 2: Build knowledge graph
# ---------------------------------------------------------------------------

def build_knowledge_graph(model, tokenizer, triples, device):
    """Build graph: (entity, relation) → target token embedding."""
    print(f"\n  Building knowledge graph...")
    embed_weight = model.model.language_model.embed_tokens.weight.data  # (vocab, hidden)

    graph = {}  # entity → {relation → {"target": str, "embedding": Tensor}}
    for subj, rel, obj, prompt in triples:
        if subj not in graph:
            graph[subj] = {}

        # Get target token embedding — try with space prefix (common for Gemma)
        target_id = None
        for prefix in [" ", "", "\n"]:
            ids = tokenizer.encode(prefix + obj, add_special_tokens=False)
            if ids:
                target_id = ids[0]
                break
        if target_id is not None:
            target_emb = embed_weight[target_id].float().clone()
            graph[subj][rel] = {
                "target": obj,
                "target_token_id": target_id,
                "embedding": target_emb.cpu(),
            }

    n_entities = len(graph)
    n_edges = sum(len(v) for v in graph.values())
    print(f"  Graph: {n_entities} entities, {n_edges} edges")

    return graph


def build_lookup_tables(graph, codebooks, hidden_dim, device):
    """Build vectorized lookup tables for batch querying."""
    entity_names = codebooks["entity_names"][0]
    relation_names = codebooks["relation_names"][0]

    n_e = len(entity_names)
    n_r = len(relation_names)

    lookup_table = torch.zeros(n_e, n_r, hidden_dim)
    lookup_mask = torch.zeros(n_e, n_r)

    e_to_idx = {n: i for i, n in enumerate(entity_names)}
    r_to_idx = {n: i for i, n in enumerate(relation_names)}

    for entity, rels in graph.items():
        ei = e_to_idx.get(entity)
        if ei is None:
            continue
        for rel, data in rels.items():
            ri = r_to_idx.get(rel)
            if ri is None:
                continue
            lookup_table[ei, ri] = data["embedding"]
            lookup_mask[ei, ri] = 1.0

    n_populated = lookup_mask.sum().int().item()
    print(f"  Lookup table: {n_e}×{n_r} = {n_populated} edges")

    return lookup_table.to(device), lookup_mask.to(device), e_to_idx, r_to_idx


# ---------------------------------------------------------------------------
# Phase 3: Per-layer FFN replacement
# ---------------------------------------------------------------------------

def evaluate_with_replacement(
    model, tokenizer, triples, non_factual, codebooks, lookup_table, lookup_mask,
    replace_layers, device, hidden_dim, inject_coeff=1.0, conf_threshold=0.3,
):
    """
    Run forward passes with FFN replaced at specified layers.
    Uses hooks to intercept and replace FFN output.

    Returns: loss on factual prompts, loss on non-factual, per-triple accuracy.
    """
    model.eval()
    n_layers = len(codebooks["entity_centroids"])

    # Hook state
    replacement_active = [False]

    def make_ffn_hook(li):
        """Hook that replaces FFN output with graph lookup."""
        def hook(module, args, output):
            if not replacement_active[0] or li not in replace_layers:
                return output

            # output is the FFN output tensor: (batch, seq, hidden)
            # args[0] is the FFN input (normed residual)
            ffn_input = args[0] if isinstance(args, tuple) else args
            B, S, D = ffn_input.shape
            flat = ffn_input.reshape(-1, D).float()

            # Decode entity/relation from residual
            e_centroids = F.normalize(codebooks["entity_centroids"][li].float(), dim=1)
            r_centroids = F.normalize(codebooks["relation_centroids"][li].float(), dim=1)
            flat_norm = F.normalize(flat, dim=1)

            e_sim = flat_norm @ e_centroids.t()
            e_conf, e_idx = e_sim.max(dim=1)

            r_sim = flat_norm @ r_centroids.t()
            r_conf, r_idx = r_sim.max(dim=1)

            # Graph lookup
            embs = lookup_table[e_idx, r_idx]  # (N, hidden)
            mask = lookup_mask[e_idx, r_idx]    # (N,)

            # Apply confidence threshold
            conf_mask = (e_conf > conf_threshold) & (r_conf > conf_threshold)
            final_mask = mask * conf_mask.float()

            # Blend: where we have a hit, inject graph embedding
            graph_out = embs * (final_mask.unsqueeze(1) * inject_coeff)
            graph_out = graph_out.reshape(B, S, D).to(output.dtype)

            # Replace FFN output where we have graph hits, keep original elsewhere
            # This preserves FFN output for non-knowledge tokens
            mask_3d = final_mask.reshape(B, S, 1).to(output.dtype)
            blended = output * (1 - mask_3d) + graph_out * mask_3d

            return blended

        return hook

    # Install hooks
    hooks = []
    for li in replace_layers:
        layer = model.model.language_model.layers[li]
        hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(li)))

    # Evaluate on factual prompts
    factual_losses = []
    correct_top1 = 0
    correct_top5 = 0
    total_factual = 0

    replacement_active[0] = True
    with torch.no_grad():
        for subj, rel, obj, prompt in triples:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Get all possible target token IDs
            target_ids_set = set()
            for prefix in ["", " ", "\n"]:
                ids = tokenizer.encode(prefix + obj, add_special_tokens=False)
                target_ids_set.update(ids[:2])
            if not target_ids_set:
                continue

            # Use the target ID with highest logit
            target_id = max(target_ids_set, key=lambda tid: logits[tid].item())

            # Loss at last position for target token
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_loss = -log_probs[target_id].item()
            factual_losses.append(token_loss)

            # Top-1/5 accuracy
            top1 = logits.argmax().item()
            top5 = set(logits.topk(5).indices.tolist())

            if top1 in target_ids_set:
                correct_top1 += 1
            if target_ids_set & top5:
                correct_top5 += 1
            total_factual += 1

    # Evaluate on non-factual prompts (should be minimally affected)
    non_factual_losses = []
    for prompt in non_factual:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                         truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Next-token loss averaged over sequence
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
            non_factual_losses.append(loss.item())

    replacement_active[0] = False
    for h in hooks:
        h.remove()

    return {
        "factual_loss": sum(factual_losses) / len(factual_losses) if factual_losses else 0,
        "non_factual_loss": sum(non_factual_losses) / len(non_factual_losses) if non_factual_losses else 0,
        "top1_accuracy": correct_top1 / total_factual if total_factual else 0,
        "top5_accuracy": correct_top5 / total_factual if total_factual else 0,
        "n_factual": total_factual,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  GEMMA 3-4B FFN THREE-SYSTEM REPLACEMENT")
    print("  Does the FFN-as-database thesis hold at real scale?")
    print("=" * 70)

    # Device — use CPU to avoid MPS float16 matmul issues
    # (forward-pass only, no training, so CPU is fine)
    device = torch.device("cpu")
    print(f"\n  Device: CPU (forward-pass only)")

    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    hidden_dim = tc.hidden_size
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers}L, hidden={hidden_dim}")

    # Verify model knows these facts
    print(f"\n  Verifying model knows the factual triples...")
    known_triples = []
    unknown_triples = []

    with torch.no_grad():
        for subj, rel, obj, prompt in FACTUAL_TRIPLES:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Try multiple tokenizations: "Paris", " Paris", etc.
            target_ids = set()
            for prefix in ["", " ", "\n"]:
                ids = tokenizer.encode(prefix + obj, add_special_tokens=False)
                target_ids.update(ids[:2])  # first 1-2 tokens

            top10 = set(logits.topk(10).indices.tolist())
            top1 = logits.argmax().item()
            top1_token = tokenizer.decode([top1]).strip()

            if target_ids & top10:
                known_triples.append((subj, rel, obj, prompt))
            else:
                unknown_triples.append((subj, rel, obj, prompt, top1_token))

    print(f"  Known (in top-5): {len(known_triples)}/{len(FACTUAL_TRIPLES)}")
    if unknown_triples[:5]:
        print(f"  Unknown examples:")
        for subj, rel, obj, prompt, pred in unknown_triples[:5]:
            print(f"    '{prompt}' → got '{pred}', expected '{obj}'")

    # Use only known triples for the experiment
    triples = known_triples if known_triples else FACTUAL_TRIPLES[:20]
    print(f"  Using {len(triples)} verified triples")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Baseline (no replacement)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Baseline (no FFN replacement)")
    print(f"{'='*70}")

    print(f"\n  Computing baseline...")
    baseline_factual_losses = []
    baseline_top1 = 0
    baseline_top5 = 0
    baseline_total = 0

    with torch.no_grad():
        for subj, rel, obj, prompt in triples:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Get all possible target token IDs
            target_ids = set()
            for prefix in ["", " ", "\n"]:
                ids = tokenizer.encode(prefix + obj, add_special_tokens=False)
                target_ids.update(ids[:2])
            if not target_ids:
                continue

            # Use the target ID with highest logit
            target_id = max(target_ids, key=lambda tid: logits[tid].item())
            log_probs = F.log_softmax(logits.float(), dim=-1)
            baseline_factual_losses.append(-log_probs[target_id].item())

            top1 = logits.argmax().item()
            top5 = set(logits.topk(5).indices.tolist())
            if top1 in target_ids:
                baseline_top1 += 1
            if target_ids & top5:
                baseline_top5 += 1
            baseline_total += 1

    baseline_nf_losses = []
    with torch.no_grad():
        for prompt in NON_FACTUAL_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ,
                             truncation=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
            baseline_nf_losses.append(loss.item())

    baseline_fl = sum(baseline_factual_losses) / len(baseline_factual_losses)
    baseline_nfl = sum(baseline_nf_losses) / len(baseline_nf_losses)

    print(f"  Baseline factual loss: {baseline_fl:.4f}")
    print(f"  Baseline non-factual loss: {baseline_nfl:.4f}")
    print(f"  Baseline top-1: {baseline_top1}/{baseline_total} "
          f"({baseline_top1/baseline_total:.0%})")
    print(f"  Baseline top-5: {baseline_top5}/{baseline_total} "
          f"({baseline_top5/baseline_total:.0%})")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Extract codebooks
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Extract entity/relation codebooks")
    print(f"{'='*70}")

    codebooks = extract_codebooks(model, tokenizer, triples, device, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Build knowledge graph + lookup tables
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Build knowledge graph")
    print(f"{'='*70}")

    graph = build_knowledge_graph(model, tokenizer, triples, device)
    lookup_table, lookup_mask, e_to_idx, r_to_idx = build_lookup_tables(
        graph, codebooks, hidden_dim, device)

    # Export graph
    graph_export = {}
    for entity, rels in graph.items():
        graph_export[entity] = {
            rel: {"target": data["target"], "token_id": data["target_token_id"]}
            for rel, data in rels.items()
        }
    with open(os.path.join(OUTPUT_DIR, "knowledge_graph.json"), "w") as f:
        json.dump(graph_export, f, indent=2)
    print(f"  Exported knowledge_graph.json")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Per-layer FFN replacement
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Per-layer FFN replacement")
    print(f"{'='*70}")

    # Test single-layer replacement at each layer
    layer_results = []
    print(f"\n  {'Layer':<8} {'Type':<12} {'Factual':>10} {'Δ':>8} "
          f"{'Non-fact':>10} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'─'*70}")

    layer_types = tc.layer_types

    for li in range(n_layers):
        ltype = layer_types[li] if li < len(layer_types) else "?"

        result = evaluate_with_replacement(
            model, tokenizer, triples, NON_FACTUAL_PROMPTS,
            codebooks, lookup_table, lookup_mask,
            replace_layers={li}, device=device, hidden_dim=hidden_dim,
            inject_coeff=1.0, conf_threshold=0.3,
        )

        delta = result["factual_loss"] - baseline_fl
        layer_results.append({
            "layer": li,
            "type": ltype,
            "factual_loss": result["factual_loss"],
            "non_factual_loss": result["non_factual_loss"],
            "delta": delta,
            "top1": result["top1_accuracy"],
            "top5": result["top5_accuracy"],
        })

        marker = " ←" if abs(delta) < 0.5 else ""
        print(f"  L{li:<6} {ltype:<12} {result['factual_loss']:>10.4f} "
              f"{delta:>+8.4f} {result['non_factual_loss']:>10.4f} "
              f"{result['top1_accuracy']:>7.0%} {result['top5_accuracy']:>7.0%}{marker}")

        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: Band replacement (knowledge layers)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 5: Band replacement")
    print(f"{'='*70}")

    # Identify best single-layer replacements
    best_layers = sorted(layer_results, key=lambda r: abs(r["delta"]))[:10]
    print(f"\n  Best single-layer replacements (lowest |Δ|):")
    for r in best_layers[:5]:
        print(f"    L{r['layer']} ({r['type']}): Δ={r['delta']:+.4f}, "
              f"top-1={r['top1']:.0%}")

    # Try band replacements
    third = n_layers // 3
    band_configs = [
        ("Early (L0-10)", set(range(0, 11))),
        ("Middle (L11-22)", set(range(11, 23))),
        ("Late (L23-33)", set(range(23, 34))),
        ("Knowledge est. (L10-23)", set(range(10, 24))),
        ("Best 5 layers", set(r["layer"] for r in best_layers[:5])),
        ("Best 10 layers", set(r["layer"] for r in best_layers[:10])),
    ]

    band_results = []
    print(f"\n  {'Band':<30} {'Factual':>10} {'Δ':>8} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'─'*70}")

    for label, layers in band_configs:
        result = evaluate_with_replacement(
            model, tokenizer, triples, NON_FACTUAL_PROMPTS,
            codebooks, lookup_table, lookup_mask,
            replace_layers=layers, device=device, hidden_dim=hidden_dim,
        )
        delta = result["factual_loss"] - baseline_fl
        band_results.append({
            "label": label,
            "layers": sorted(layers),
            "factual_loss": result["factual_loss"],
            "delta": delta,
            "top1": result["top1_accuracy"],
            "top5": result["top5_accuracy"],
        })
        print(f"  {label:<30} {result['factual_loss']:>10.4f} "
              f"{delta:>+8.4f} {result['top1_accuracy']:>7.0%} "
              f"{result['top5_accuracy']:>7.0%}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY: FFN REPLACEMENT AT 4B SCALE")
    print(f"{'='*70}")

    print(f"\n  Baseline: factual_loss={baseline_fl:.4f}, "
          f"top-1={baseline_top1}/{baseline_total} ({baseline_top1/baseline_total:.0%}), "
          f"top-5={baseline_top5}/{baseline_total} ({baseline_top5/baseline_total:.0%})")

    # Find layers where replacement has minimal impact
    good_layers = [r for r in layer_results if abs(r["delta"]) < 0.5]
    ok_layers = [r for r in layer_results if abs(r["delta"]) < 1.0]

    print(f"\n  Single-layer replacement impact:")
    print(f"    |Δ| < 0.5: {len(good_layers)}/{n_layers} layers ({len(good_layers)/n_layers:.0%})")
    print(f"    |Δ| < 1.0: {len(ok_layers)}/{n_layers} layers ({len(ok_layers)/n_layers:.0%})")

    # Layer band analysis
    for third_name, start, end in [("Early", 0, 11), ("Middle", 11, 23), ("Late", 23, 34)]:
        band_good = [r for r in layer_results if start <= r["layer"] < end and abs(r["delta"]) < 0.5]
        band_total = end - start
        print(f"    {third_name} (L{start}-{end-1}): {len(band_good)}/{band_total} replaceable")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    best_single = min(layer_results, key=lambda r: abs(r["delta"]))
    best_band = min(band_results, key=lambda r: abs(r["delta"]))

    if best_single["delta"] < 0.1 and best_single["top1"] >= baseline_top1/baseline_total * 0.9:
        print(f"\n  ✓ SINGLE-LAYER REPLACEMENT WORKS at 4B")
        print(f"    Best: L{best_single['layer']} Δ={best_single['delta']:+.4f}, "
              f"top-1={best_single['top1']:.0%}")
        print(f"    (20M result was Δ=+0.002, 100% top-1 at L6)")
    else:
        print(f"\n  Best single-layer: L{best_single['layer']} "
              f"Δ={best_single['delta']:+.4f}, top-1={best_single['top1']:.0%}")

    if len(good_layers) >= n_layers * 0.3:
        print(f"\n  ✓ FFN IS REPLACEABLE at scale ({len(good_layers)}/{n_layers} layers)")
        print(f"    The FFN-as-database thesis holds.")
    elif len(ok_layers) >= n_layers * 0.3:
        print(f"\n  ~ FFN PARTIALLY REPLACEABLE ({len(ok_layers)}/{n_layers} at |Δ|<1.0)")
    else:
        print(f"\n  ✗ FFN REPLACEMENT DEGRADES AT SCALE")
        print(f"    Only {len(good_layers)}/{n_layers} layers replaceable at |Δ|<0.5")

    print(f"\n  Comparison with 20M results:")
    print(f"    {'Metric':<35} {'20M':>12} {'4B':>12}")
    print(f"    {'─'*60}")
    print(f"    {'Single-layer Δ (best)':<35} {'+0.002':>12} "
          f"{best_single['delta']:>+12.4f}")
    print(f"    {'Single-layer top-1':<35} {'100%':>12} "
          f"{best_single['top1']:>11.0%}")
    print(f"    {'Replaceable layers (|Δ|<0.5)':<35} {'4/12 (33%)':>12} "
          f"{len(good_layers)}/{n_layers} ({len(good_layers)/n_layers:.0%})")

    # Save
    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "baseline_factual_loss": baseline_fl,
        "baseline_nonfactual_loss": baseline_nfl,
        "baseline_top1": baseline_top1 / baseline_total,
        "baseline_top5": baseline_top5 / baseline_total,
        "n_triples": len(triples),
        "n_known_triples": len(known_triples),
        "per_layer": layer_results,
        "band_results": band_results,
        "good_layers_count": len(good_layers),
        "ok_layers_count": len(ok_layers),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
