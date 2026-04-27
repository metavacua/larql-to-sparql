#!/usr/bin/env python3
"""
Backpropagation-is-INSERT: Feature-Level Analysis (v4)

Tracks WHICH specific FFN features receive gradient writes from each
relation type, and whether those assignments are stable across training.

Key measurements:
  1. Feature stability: does the same feature always receive "plural" writes?
  2. Feature→relation mapping: address table (layer, feature) → relation
  3. Feature birth: when does each feature claim its relation?
  4. Feature exclusivity: is each feature owned by one relation or shared?
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model import TinyGemma
from synth_data_v2 import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_LAYERS = 12
DIM = 256
FFN_DIM = 1024
EPOCHS = 40
BATCH_SIZE = 8
LR = 3e-4
MAX_SEQ = 64
SEED = 42

# Track features every N steps
TRACK_EVERY = 25
# Full anatomy at these steps
ANATOMY_STEPS = {0, 50, 100, 250, 500, 1000, 2000, 4000}

OUTPUT_DIR = "results_v4"
TOP_K = 10  # top features per gradient


# ---------------------------------------------------------------------------
# Dataset (same as v3)
# ---------------------------------------------------------------------------

class ClampedTokenizer:
    def __init__(self, tok, vocab):
        self.tok = tok
        self.vocab = vocab
        self.pad_token_id = tok.pad_token_id or 0
    def encode(self, text, **kwargs):
        ids = self.tok.encode(text, **kwargs)
        return [min(i, self.vocab - 1) for i in ids]


class TokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   max_length=max_len, truncation=True)
            self.encodings.append(ids)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def collate_fn(batch, pad_id=0):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


# ---------------------------------------------------------------------------
# Feature Registry: tracks which features receive writes from each relation
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    Tracks per-feature gradient accumulation, indexed by relation type.
    This is the core data structure: it builds the (layer, feature) → relation map.
    """

    def __init__(self, n_layers: int, ffn_dim: int):
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim

        # Per-relation cumulative gradient energy: relation → (n_layers, ffn_dim)
        self.relation_energy = defaultdict(
            lambda: torch.zeros(n_layers, ffn_dim)
        )
        # Per-band cumulative
        self.band_energy = defaultdict(
            lambda: torch.zeros(n_layers, ffn_dim)
        )
        # Total energy (all relations combined)
        self.total_energy = torch.zeros(n_layers, ffn_dim)

        # Snapshots for tracking stability over time
        # step → { relation → top-K features per layer }
        self.snapshots = {}

        self.n_updates = defaultdict(int)
        self.total_updates = 0

    def record_gradient(self, model: TinyGemma, relation: str, band: str):
        """Record which features received gradient from this relation."""
        for li, layer in enumerate(model.layers):
            g = layer.ffn.gate.weight.grad
            if g is None:
                continue
            feat_norms = g.norm(dim=1).cpu()  # (ffn_dim,)
            self.relation_energy[relation][li] += feat_norms
            self.band_energy[band][li] += feat_norms
            self.total_energy[li] += feat_norms

        self.n_updates[relation] += 1
        self.total_updates += 1

    def snapshot(self, step: int):
        """Save current top-K features per relation per layer."""
        snap = {}
        for relation, energy in self.relation_energy.items():
            rel_snap = {}
            for li in range(self.n_layers):
                topk = energy[li].topk(min(TOP_K, self.ffn_dim))
                rel_snap[li] = topk.indices.tolist()
            snap[relation] = rel_snap
        self.snapshots[step] = snap

    def compute_stability(self) -> Dict:
        """
        Measure how stable feature assignments are across snapshots.
        For each relation, compute Jaccard similarity of top-K features
        between consecutive snapshots.
        """
        steps = sorted(self.snapshots.keys())
        if len(steps) < 2:
            return {}

        stability = {}
        for relation in self.relation_energy.keys():
            rel_stability = []
            for i in range(1, len(steps)):
                prev = self.snapshots[steps[i-1]].get(relation, {})
                curr = self.snapshots[steps[i]].get(relation, {})

                layer_jaccards = []
                for li in range(self.n_layers):
                    prev_set = set(prev.get(li, []))
                    curr_set = set(curr.get(li, []))
                    if prev_set or curr_set:
                        jaccard = len(prev_set & curr_set) / len(prev_set | curr_set)
                    else:
                        jaccard = 1.0
                    layer_jaccards.append(jaccard)

                rel_stability.append({
                    "step_from": steps[i-1],
                    "step_to": steps[i],
                    "mean_jaccard": sum(layer_jaccards) / len(layer_jaccards),
                    "per_layer": layer_jaccards,
                })

            stability[relation] = rel_stability

        return stability

    def compute_exclusivity(self) -> Dict:
        """
        For each feature, measure how exclusively it belongs to one relation.
        Exclusivity = max_relation_energy / total_energy for that feature.
        A feature with exclusivity 1.0 only receives writes from one relation.
        """
        results = {}
        for li in range(self.n_layers):
            total = self.total_energy[li]  # (ffn_dim,)

            # For each feature, find which relation contributes most
            feat_exclusivity = []
            feat_owner = []

            for fi in range(self.ffn_dim):
                if total[fi] < 1e-10:
                    feat_exclusivity.append(0.0)
                    feat_owner.append("none")
                    continue

                max_energy = 0.0
                owner = "none"
                for rel, energy in self.relation_energy.items():
                    if energy[li][fi] > max_energy:
                        max_energy = energy[li][fi]
                        owner = rel

                excl = max_energy / total[fi].item()
                feat_exclusivity.append(excl)
                feat_owner.append(owner)

            # Stats
            excl_tensor = torch.tensor(feat_exclusivity)
            active = (total > total.median() * 0.1)
            active_excl = excl_tensor[active]

            results[f"L{li}"] = {
                "mean_exclusivity": active_excl.mean().item() if len(active_excl) > 0 else 0,
                "median_exclusivity": active_excl.median().item() if len(active_excl) > 0 else 0,
                "pct_exclusive_gt80": (active_excl > 0.8).float().mean().item() if len(active_excl) > 0 else 0,
                "pct_exclusive_gt50": (active_excl > 0.5).float().mean().item() if len(active_excl) > 0 else 0,
                "n_active_features": active.sum().item(),
            }

            # Top-5 most exclusive features per layer
            top_excl_idx = excl_tensor.topk(min(5, len(excl_tensor))).indices.tolist()
            results[f"L{li}"]["top_exclusive"] = [
                {"feature": fi, "owner": feat_owner[fi],
                 "exclusivity": feat_exclusivity[fi]}
                for fi in top_excl_idx
            ]

        return results

    def compute_address_table(self) -> List[Dict]:
        """
        Build the (layer, feature) → relation address table.
        Only includes features with > 50% exclusivity.
        """
        table = []
        for li in range(self.n_layers):
            total = self.total_energy[li]

            for fi in range(self.ffn_dim):
                if total[fi] < total.median() * 0.5:
                    continue

                max_energy = 0.0
                owner = "none"
                for rel, energy in self.relation_energy.items():
                    if energy[li][fi] > max_energy:
                        max_energy = energy[li][fi]
                        owner = rel

                excl = max_energy / total[fi].item() if total[fi] > 0 else 0
                if excl > 0.5:
                    table.append({
                        "layer": li,
                        "feature": fi,
                        "relation": owner,
                        "exclusivity": round(excl, 3),
                        "energy": round(max_energy.item(), 4),
                    })

        # Sort by exclusivity descending
        table.sort(key=lambda x: -x["exclusivity"])
        return table

    def compute_relation_overlap(self) -> Dict:
        """
        For each pair of relations, compute feature overlap per layer.
        Low overlap = clean separation. High overlap = shared features.
        """
        relations = list(self.relation_energy.keys())
        overlaps = {}

        for i, rel_a in enumerate(relations):
            for rel_b in relations[i+1:]:
                pair_key = f"{rel_a} vs {rel_b}"
                layer_overlaps = []

                for li in range(self.n_layers):
                    a = self.relation_energy[rel_a][li]
                    b = self.relation_energy[rel_b][li]

                    # Normalise
                    a_n = a / (a.sum() + 1e-10)
                    b_n = b / (b.sum() + 1e-10)

                    # Top-K overlap
                    a_top = set(a.topk(TOP_K).indices.tolist())
                    b_top = set(b.topk(TOP_K).indices.tolist())

                    if a_top or b_top:
                        jaccard = len(a_top & b_top) / len(a_top | b_top)
                    else:
                        jaccard = 0.0

                    # Distribution overlap (min of normalised)
                    dist_overlap = torch.min(a_n, b_n).sum().item()

                    layer_overlaps.append({
                        "jaccard": jaccard,
                        "dist_overlap": dist_overlap,
                    })

                overlaps[pair_key] = {
                    "mean_jaccard": sum(l["jaccard"] for l in layer_overlaps) / len(layer_overlaps),
                    "mean_dist_overlap": sum(l["dist_overlap"] for l in layer_overlaps) / len(layer_overlaps),
                    "per_layer_jaccard": [l["jaccard"] for l in layer_overlaps],
                }

        return overlaps


# ---------------------------------------------------------------------------
# Single-gradient anatomy with feature tracking
# ---------------------------------------------------------------------------

def gradient_step_and_track(
    model: TinyGemma,
    sample: GroundTruth,
    tokenizer: ClampedTokenizer,
    registry: FeatureRegistry,
    device: torch.device,
) -> Dict:
    """Run a single gradient step and record features."""
    model.train()
    model.zero_grad()

    ids = tokenizer.encode(sample.text, add_special_tokens=True,
                           max_length=MAX_SEQ, truncation=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    loss.backward()

    # Record in registry
    registry.record_gradient(model, sample.relation, sample.band)

    # Collect per-layer top features for this gradient
    top_features_per_layer = {}
    for li, layer in enumerate(model.layers):
        g = layer.ffn.gate.weight.grad
        if g is not None:
            feat_norms = g.norm(dim=1)
            topk = feat_norms.topk(min(TOP_K, len(feat_norms)))
            top_features_per_layer[li] = {
                "indices": topk.indices.tolist(),
                "norms": [round(v.item(), 4) for v in topk.values],
            }

    model.zero_grad()

    return {
        "relation": sample.relation,
        "band": sample.band,
        "loss": loss.item(),
        "top_features": top_features_per_layer,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_stability(stability: Dict, step: int):
    """Show feature stability for each relation."""
    print(f"\n{'━'*70}")
    print(f"  FEATURE STABILITY @ step {step}")
    print(f"  (Jaccard similarity of top-{TOP_K} features between snapshots)")
    print(f"{'━'*70}")

    # Get latest stability for each relation
    for relation, steps_data in sorted(stability.items()):
        if not steps_data:
            continue
        latest = steps_data[-1]
        j = latest["mean_jaccard"]
        bar = '█' * int(30 * j)
        signal = "STABLE" if j > 0.6 else "settling" if j > 0.3 else "volatile"
        print(f"  {relation:<20} {bar:30s} {j:.3f}  [{signal}]")

    sys.stdout.flush()


def display_exclusivity(exclusivity: Dict):
    """Show per-layer feature exclusivity."""
    print(f"\n{'━'*70}")
    print(f"  FEATURE EXCLUSIVITY")
    print(f"  (1.0 = feature exclusively owned by one relation)")
    print(f"{'━'*70}")

    print(f"\n  {'Layer':<6} {'Mean':>6} {'Median':>7} {'>80%':>6} {'>50%':>6} {'Active':>7}")
    for li in range(N_LAYERS):
        d = exclusivity.get(f"L{li}", {})
        mean = d.get("mean_exclusivity", 0)
        median = d.get("median_exclusivity", 0)
        gt80 = d.get("pct_exclusive_gt80", 0)
        gt50 = d.get("pct_exclusive_gt50", 0)
        active = d.get("n_active_features", 0)
        print(f"  L{li:<4} {mean:>6.3f} {median:>7.3f} {gt80:>6.1%} {gt50:>6.1%} {active:>7}")

    # Show top exclusive features
    print(f"\n  Top exclusive features:")
    for li in range(N_LAYERS):
        d = exclusivity.get(f"L{li}", {})
        tops = d.get("top_exclusive", [])
        if tops:
            best = tops[0]
            print(f"    L{li}: feature #{best['feature']:4d} → "
                  f"{best['owner']:<20s} (excl={best['exclusivity']:.3f})")

    sys.stdout.flush()


def display_address_table(table: List[Dict]):
    """Show the feature→relation address table."""
    print(f"\n{'━'*70}")
    print(f"  ADDRESS TABLE: (layer, feature) → relation")
    print(f"  ({len(table)} entries with exclusivity > 50%)")
    print(f"{'━'*70}")

    # Group by relation
    by_rel = defaultdict(list)
    for entry in table:
        by_rel[entry["relation"]].append(entry)

    print(f"\n  {'Relation':<20} {'Entries':>8} {'Avg Excl':>9} {'Layers':>20}")
    for rel in sorted(by_rel.keys()):
        entries = by_rel[rel]
        avg_excl = sum(e["exclusivity"] for e in entries) / len(entries)
        layers = sorted(set(e["layer"] for e in entries))
        layer_str = ",".join(str(l) for l in layers[:8])
        if len(layers) > 8:
            layer_str += "..."
        print(f"  {rel:<20} {len(entries):>8} {avg_excl:>9.3f} {layer_str:>20}")

    # Show a few example entries
    print(f"\n  Sample entries (highest exclusivity):")
    for entry in table[:15]:
        print(f"    L{entry['layer']:2d} feat#{entry['feature']:4d} → "
              f"{entry['relation']:<20s} excl={entry['exclusivity']:.3f}")

    sys.stdout.flush()


def display_overlap(overlaps: Dict):
    """Show relation-pair overlap."""
    print(f"\n{'━'*70}")
    print(f"  RELATION OVERLAP (feature sharing between relation pairs)")
    print(f"{'━'*70}")

    # Sort by overlap (ascending = most separated first)
    sorted_pairs = sorted(overlaps.items(), key=lambda x: x[1]["mean_jaccard"])

    print(f"\n  {'Pair':<45} {'Jaccard':>8} {'DistOvl':>8}")
    for pair, data in sorted_pairs[:20]:
        j = data["mean_jaccard"]
        d = data["mean_dist_overlap"]
        signal = "SEPARATED" if j < 0.05 else "low" if j < 0.15 else "shared"
        print(f"  {pair:<45} {j:>8.3f} {d:>8.3f}  [{signal}]")

    sys.stdout.flush()


def display_stability_over_time(stability: Dict):
    """Show stability trajectory for key relations."""
    print(f"\n{'━'*70}")
    print(f"  STABILITY TRAJECTORY (feature assignment over training)")
    print(f"{'━'*70}")

    key_relations = ["capital_of", "president_of", "synonym", "hypernym",
                     "plural", "past_tense", "python:def", "rust:fn"]

    for rel in key_relations:
        if rel not in stability or not stability[rel]:
            continue

        print(f"\n  {rel}:")
        for entry in stability[rel]:
            step = entry["step_to"]
            j = entry["mean_jaccard"]
            bar = '▓' * int(25 * j) + '░' * (25 - int(25 * j))
            print(f"    step {step:>5}: {bar} {j:.3f}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  BACKPROPAGATION IS INSERT — v4: Feature-Level Analysis")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"\n  Device: CPU")

    # Tokenizer
    print("  Loading Gemma tokenizer...")
    raw_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if raw_tok.pad_token_id is None:
        raw_tok.pad_token_id = 0
    VOCAB = 32000
    tokenizer = ClampedTokenizer(raw_tok, VOCAB)
    print(f"  Vocab: {VOCAB}")

    # Data
    print("  Building corpus...")
    samples, ground_truth = build_mixed_corpus(n_countries=50, seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    # Model
    model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)
    print(f"  Params: {model.param_count():,}")

    # Dataset + loader
    dataset = TokenDataset([s.text for s in samples], tokenizer, MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Feature registry
    registry = FeatureRegistry(N_LAYERS, FFN_DIM)

    # Fixed anatomy samples
    rng = random.Random(SEED)
    band_map = defaultdict(list)
    for s in samples:
        band_map[s.band].append(s)

    anatomy_samples = {}
    for band in ["syntax", "knowledge", "code"]:
        pool = band_map[band]
        anatomy_samples[band] = rng.sample(pool, min(10, len(pool)))

    # Also group by relation for per-relation tracking
    relation_samples = defaultdict(list)
    for s in samples:
        relation_samples[s.relation].append(s)

    # Select fixed samples per relation for tracking
    track_samples = {}
    for rel, pool in relation_samples.items():
        track_samples[rel] = rng.sample(pool, min(5, len(pool)))

    # ─── Training ───
    step = 0
    loss_history = []
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  TRAINING — {EPOCHS} epochs, tracking features every {TRACK_EVERY} steps")
    print(f"{'='*70}")
    sys.stdout.flush()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_batch = 0

        for batch in loader:
            batch = batch.to(device)

            # --- Feature tracking at scheduled intervals ---
            if step % TRACK_EVERY == 0:
                # Run single-gradient on samples from each relation
                for rel, samps in track_samples.items():
                    for s in samps:
                        gradient_step_and_track(
                            model, s, tokenizer, registry, device
                        )

            # --- Full anatomy at milestone steps ---
            if step in ANATOMY_STEPS:
                registry.snapshot(step)

                # Show stability so far
                if step > 0:
                    stability = registry.compute_stability()
                    display_stability(stability, step)

                # Show exclusivity
                exclusivity = registry.compute_exclusivity()
                display_exclusivity(exclusivity)

                sys.stdout.flush()

            # --- Normal training step ---
            model.train()
            optimizer.zero_grad()

            logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batch += 1
            loss_history.append({"step": step, "loss": loss_val})
            step += 1

        avg_loss = epoch_loss / max(n_batch, 1)
        elapsed = time.time() - t0
        # Compact epoch line
        print(f"  E{epoch+1:2d}/{EPOCHS} loss={avg_loss:.4f} step={step} {elapsed:.0f}s")
        sys.stdout.flush()

    # ─── Final Analysis ───
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS @ step {step}")
    print(f"{'='*70}")

    # Final snapshot
    # Run one more round of tracking to ensure final state captured
    for rel, samps in track_samples.items():
        for s in samps:
            gradient_step_and_track(model, s, tokenizer, registry, device)
    registry.snapshot(step)

    # 1. Stability
    stability = registry.compute_stability()
    display_stability(stability, step)
    display_stability_over_time(stability)

    # 2. Exclusivity
    exclusivity = registry.compute_exclusivity()
    display_exclusivity(exclusivity)

    # 3. Address table
    address_table = registry.compute_address_table()
    display_address_table(address_table)

    # 4. Relation overlap
    overlaps = registry.compute_relation_overlap()
    display_overlap(overlaps)

    # ─── Save ───
    with open(os.path.join(OUTPUT_DIR, "stability.json"), "w") as f:
        json.dump(stability, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "exclusivity.json"), "w") as f:
        json.dump(exclusivity, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "address_table.json"), "w") as f:
        json.dump(address_table, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "overlaps.json"), "w") as f:
        json.dump(overlaps, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "loss.json"), "w") as f:
        json.dump(loss_history, f)

    with open(os.path.join(OUTPUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Summary snapshot counts
    print(f"\n  Registry: {registry.total_updates} gradient recordings")
    print(f"  Relations tracked: {len(registry.relation_energy)}")
    print(f"  Snapshots: {len(registry.snapshots)}")
    print(f"  Address table entries: {len(address_table)}")
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"  Results: {OUTPUT_DIR}/")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
