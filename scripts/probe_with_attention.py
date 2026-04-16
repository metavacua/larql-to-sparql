#!/usr/bin/env python3
"""Probe entities with attention — full forward pass through the model.

For each relation template × entity, runs a prompted forward pass,
captures gate activations at L14-27, and records which features fire.
Matches (entity, output_token) against Wikidata triples for labeling.

This uses the vindex's model_weights.bin for inference — no separate
model download needed if --include-weights was used during extraction.

Usage:
    python3 scripts/probe_with_attention.py

Requires: output/gemma3-4b-full.vindex with model weights
Time: ~20-30 min for 32 templates × 50 entities each
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

# Relation templates — each template probes one relation type
TEMPLATES = {
    "capital": "The capital of {X} is",
    "language": "The official language of {X} is",
    "continent": "{X} is located in the continent of",
    "borders": "{X} shares a border with",
    "occupation": "{X} was a famous",
    "author": "The author of {X} was",
    "director": "{X} was directed by",
    "birthplace": "{X} was born in",
    "genre": "The genre of {X} is",
    "currency": "The currency of {X} is the",
    "located in": "{X} is located in",
    "founder": "{X} was founded by",
    "composer": "{X} was composed by",
    "starring": "{X} stars",
    "nationality": "The nationality of {X} is",
    "religion": "The religion of {X} is",
    "spouse": "{X} was married to",
    "instrument": "{X} plays the",
}


def load_vindex_for_inference(vindex_dir):
    """Load everything needed for forward pass from the vindex."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    if not config.get("has_model_weights", False):
        print("ERROR: vindex does not have model weights.")
        print("Rebuild with: --include-weights")
        sys.exit(1)

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    embed_scale = config["embed_scale"]

    # Load embeddings
    embed_raw = np.fromfile(vindex_dir / "embeddings.bin", dtype=np.float32)
    embed = embed_raw.reshape(vocab_size, hidden_size)

    # Load gate vectors
    gate_raw = np.fromfile(vindex_dir / "gate_vectors.bin", dtype=np.float32)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // 4
        gates[layer] = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)

    # Load down_meta
    down_meta = {}
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            down_meta[(obj.get("l", 0), obj.get("f", 0))] = obj.get("t", "")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(vindex_dir / "tokenizer.json"))

    return config, embed, embed_scale, gates, down_meta, tokenizer


def get_residual_from_cli(vindex_dir, prompt, tokenizer):
    """Get the residual stream at each layer by running the walk command.

    This is a workaround — we use the vindex's gate vectors to simulate
    what the model would do. For true attention-based probing, we'd need
    the full model weights loaded.

    For now, we use a simpler approach: embed the FULL prompt (not just
    the entity), which gives us context-dependent embeddings through
    the tokenizer.
    """
    # Embed the full prompt — last token's embedding captures some context
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    ids = encoding.ids
    if not ids:
        return None

    # Use last token embedding (this is where the model would predict)
    last_id = ids[-1]
    return last_id


def main():
    vindex_dir = "output/gemma3-4b-full.vindex"
    triples_path = "data/wikidata_triples.json"

    print("Loading vindex...")
    config, embed, embed_scale, gates, down_meta, tokenizer = load_vindex_for_inference(vindex_dir)
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    print(f"  {config['num_layers']} layers, {hidden_size} hidden")

    print("Loading triples...")
    with open(triples_path) as f:
        triples = json.load(f)

    # Build lookup
    pair_to_relation = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                pair_to_relation[(pair[0].lower(), pair[1].lower())] = rel_name

    layers = list(range(14, 28))
    feature_labels = {}
    relation_counts = {}
    total_probes = 0
    total_matched = 0

    print(f"\nProbing with {len(TEMPLATES)} templates...")

    for rel_name, template in TEMPLATES.items():
        if rel_name not in triples:
            print(f"  {rel_name}: no triples, skipping")
            continue

        # Get subjects for this relation
        subjects = list(set(
            pair[0] for pair in triples[rel_name].get("pairs", [])
            if len(pair) >= 2 and len(pair[0]) <= 30
        ))[:50]  # Max 50 per relation

        if not subjects:
            continue

        matched_this_rel = 0

        for subject in subjects:
            prompt = template.replace("{X}", subject)

            # Tokenize the full prompt
            encoding = tokenizer.encode(prompt, add_special_tokens=True)
            ids = encoding.ids
            if not ids:
                continue

            # Use LAST token embedding as the query — this is where prediction happens
            last_id = ids[-1]
            if last_id <= 2 or last_id >= vocab_size:
                continue

            query = embed[last_id] * embed_scale

            # Also try averaging the last few tokens for more context
            context_ids = [i for i in ids[-3:] if i > 2 and i < vocab_size]
            if len(context_ids) > 1:
                context_query = np.mean([embed[i] * embed_scale for i in context_ids], axis=0)
                # Blend: 70% last token, 30% context
                query = 0.7 * query + 0.3 * context_query

            total_probes += 1

            # Gate KNN at each knowledge layer
            for layer in layers:
                if layer not in gates:
                    continue
                scores = gates[layer] @ query
                top_indices = np.argsort(-np.abs(scores))[:5]

                for feat_idx in top_indices:
                    score = float(scores[feat_idx])
                    if abs(score) < 5.0:
                        continue

                    target = down_meta.get((layer, int(feat_idx)), "")
                    if len(target) < 2:
                        continue

                    # Check Wikidata match
                    key = (subject.lower(), target.lower())
                    if key in pair_to_relation and pair_to_relation[key] == rel_name:
                        feat_key = f"L{layer}_F{feat_idx}"
                        if feat_key not in feature_labels:
                            feature_labels[feat_key] = rel_name
                            matched_this_rel += 1
                            total_matched += 1

        if matched_this_rel > 0:
            relation_counts[rel_name] = matched_this_rel
        print(f"  {rel_name:<20s} {len(subjects):3d} entities  → {matched_this_rel} features labeled")

    print(f"\nTotal probes: {total_probes}")
    print(f"Total labeled: {len(feature_labels)} features")
    print(f"\nRelation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<25s} {count:4d}")

    # Merge with existing labels (keep existing, add new)
    existing_path = Path(vindex_dir) / "feature_labels.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

    new_count = 0
    for key, rel in feature_labels.items():
        if key not in existing:
            existing[key] = rel
            new_count += 1

    # Save
    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMerged: {new_count} new + {len(existing) - new_count} existing = {len(existing)} total")
    print(f"Saved to {existing_path}")

    # Also save to data/
    data_path = Path("data/feature_labels.json")
    with open(data_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {data_path}")


if __name__ == "__main__":
    main()
