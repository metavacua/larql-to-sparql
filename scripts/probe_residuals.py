#!/usr/bin/env python3
"""Probe with REAL residuals — capture post-attention residuals via larql,
then project through gate matrices to find actual feature activations.

Step 1: Use `larql residuals` to capture residual vectors at each layer
Step 2: Project residuals through gate vectors (from the vindex)
Step 3: Find top-K activated features per layer
Step 4: Match (entity, feature_output) against Wikidata

This is ground truth — the model processed the prompt with full attention,
and we're reading which features actually fire on the resulting residual.

Usage:
    python3 scripts/probe_residuals.py
"""

import json
import subprocess
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


TEMPLATES = {
    "capital": "The capital of {X} is",
    "language": "The official language of {X} is",
    "continent": "{X} is a country in",
    "borders": "{X} shares a border with",
    "occupation": "{X} was a",
    "birthplace": "{X} was born in",
    "currency": "The currency of {X} is",
    "located in": "{X} is located in",
    "author": "{X} was written by",
    "director": "{X} was directed by",
    "genre": "The genre of {X} is",
    "founder": "{X} was founded by",
    "nationality": "{X} has the nationality of",
}

VINDEX = "output/gemma3-4b-full.vindex"


def load_vindex_gates_and_meta(vindex_dir):
    """Load gate vectors and down_meta from vindex."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]

    gate_raw = np.fromfile(vindex_dir / "gate_vectors.bin", dtype=np.float32)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // 4
        gates[layer] = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)

    down_meta = {}
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            down_meta[(obj.get("l", 0), obj.get("f", 0))] = obj.get("t", "")

    return config, gates, down_meta


def capture_residuals(prompt, vindex_dir):
    """Run larql residuals command to get per-layer residual vectors."""
    try:
        result = subprocess.run(
            ["target/release/larql", "residuals",
             "--model", vindex_dir,
             "--prompts", prompt,
             "--output", "/dev/stdout",
             "--format", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def project_residual_through_gates(residual_vec, gate_matrix, top_k=10):
    """Project a residual vector through gate matrix to find top-K features."""
    residual = np.array(residual_vec, dtype=np.float32)
    scores = gate_matrix @ residual
    top_indices = np.argsort(-np.abs(scores))[:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if abs(scores[i]) > 3.0]


def main():
    triples_path = "data/wikidata_triples.json"

    print("Loading vindex gates and metadata...")
    config, gates, down_meta = load_vindex_gates_and_meta(VINDEX)
    hidden_size = config["hidden_size"]

    print("Loading triples...")
    with open(triples_path) as f:
        triples = json.load(f)

    pair_to_relation = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                pair_to_relation[(pair[0].lower(), pair[1].lower())] = rel_name

    # First test: can we capture residuals?
    print("\nTesting residual capture...")
    test_prompt = "The capital of France is"
    residuals = capture_residuals(test_prompt, VINDEX)

    if residuals is None:
        print("ERROR: Could not capture residuals.")
        print("The `larql residuals` command may not support this vindex format.")
        print("\nFalling back to direct gate projection from model weights...")
        run_direct_probe()
        return

    print(f"  Captured residuals at {len(residuals)} layers")

    # Full probe
    feature_labels = {}
    relation_counts = defaultdict(int)
    total_probes = 0

    for rel_name, template in TEMPLATES.items():
        if rel_name not in triples:
            continue

        subjects = list(set(
            pair[0] for pair in triples[rel_name].get("pairs", [])
            if len(pair) >= 2 and len(pair[0]) <= 25
        ))[:20]

        matched = 0
        for subject in subjects:
            prompt = template.replace("{X}", subject)
            residuals = capture_residuals(prompt, VINDEX)
            if residuals is None:
                continue

            total_probes += 1

            for layer_data in residuals:
                layer = layer_data.get("layer", 0)
                if layer < 14 or layer > 27:
                    continue
                if layer not in gates:
                    continue

                vec = layer_data.get("residual", [])
                if len(vec) != hidden_size:
                    continue

                hits = project_residual_through_gates(vec, gates[layer], top_k=10)
                for feat, score in hits:
                    target = down_meta.get((layer, feat), "")
                    if len(target) < 2:
                        continue
                    key = (subject.lower(), target.lower())
                    if key in pair_to_relation and pair_to_relation[key] == rel_name:
                        feat_key = f"L{layer}_F{feat}"
                        if feat_key not in feature_labels:
                            feature_labels[feat_key] = rel_name
                            relation_counts[rel_name] += 1
                            matched += 1

        print(f"  {rel_name:<20s} {len(subjects):3d} entities → {matched} features")

    # Save
    existing_path = Path(VINDEX) / "feature_labels.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

    new_count = 0
    for key, rel in feature_labels.items():
        if key not in existing:
            existing[key] = rel
            new_count += 1

    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)

    data_path = Path("data/feature_labels.json")
    with open(data_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMerged: {new_count} new + {len(existing) - new_count} existing = {len(existing)} total")


def run_direct_probe():
    """Fallback: use trace_forward via a Rust helper to get real residuals,
    then project through gates. This requires adding a new CLI command."""
    print("\nDirect probe requires a `larql probe` CLI command.")
    print("TODO: Add `larql probe --template 'The capital of {X} is' --entities France,Germany,Japan`")
    print("      that runs trace_forward and dumps per-layer gate activations.")


if __name__ == "__main__":
    main()
