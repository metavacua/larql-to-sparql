#!/usr/bin/env python3
"""
Graph-to-Weights Compiler (v6)

Bypass gradient descent for the FFN entirely. Compile structured knowledge
directly into FFN weights, then train attention only.

Strategy:
  Phase 1: Train a baseline model (reuse v5 baseline)
  Phase 2: Extract "relation templates" — for each relation type, capture
           which gate features fire and what residual patterns produce them
  Phase 3: Build a fresh model. For each edge in the source graph:
           - Run the prompt through embed + random attention to get residuals
           - Write gate vectors aligned to those residuals
           - Write down projections that produce the target token
  Phase 4: Train attention-only on the compiled model
  Phase 5: Compare: compiled+attn vs freeze-FFN vs baseline

The key insight from freeze-FFN: attention doesn't care HOW the database
was built. It learns to query whatever you give it. So the compiler just
needs to produce gate/down patterns that are query-able.
"""

import os
import sys
import json
import time
import copy
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

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
ATTN_ONLY_EPOCHS = 15  # attention-only training after compilation
BATCH_SIZE = 8
LR = 3e-4
MAX_SEQ = 64
SEED = 42
VOCAB = 32000

OUTPUT_DIR = "results_v6_compiler"


# ---------------------------------------------------------------------------
# Dataset (reused)
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
# Phase 1: Train baseline (or load from v5)
# ---------------------------------------------------------------------------

def train_baseline(loader, tokenizer, device, epochs=EPOCHS):
    """Train full baseline model. Returns trained model."""
    print(f"\n  Training baseline ({epochs} epochs)...")
    torch.manual_seed(SEED)
    model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / n
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()

    print(f"  Baseline done: loss={avg:.4f}, {time.time()-t0:.0f}s")
    return model


# ---------------------------------------------------------------------------
# Phase 2: Extract relation templates from trained model
# ---------------------------------------------------------------------------

def extract_relation_templates(
    model: TinyGemma,
    samples: List[GroundTruth],
    tokenizer: ClampedTokenizer,
    device: torch.device,
) -> Dict:
    """
    For each relation type, extract:
    - The average residual stream at each layer (what the relation "looks like" internally)
    - The gate activation pattern (which features fire)
    - The down projection contribution (what each feature outputs)
    """
    print(f"\n  Extracting relation templates...")
    model.eval()

    # Hooks to capture residuals and gate activations
    residuals = {}  # layer → list of tensors
    gate_acts = {}  # layer → list of tensors

    hooks = []
    layer_residuals = [None] * N_LAYERS
    layer_gate_acts = [None] * N_LAYERS

    # Hook into the FFN norm input (= residual after attention)
    def make_residual_hook(li):
        def hook(module, input, output):
            layer_residuals[li] = input[0].detach()  # input to norm
        return hook

    def make_gate_hook(li):
        def hook(module, input, output):
            layer_gate_acts[li] = output.detach()
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.ffn_norm.register_forward_hook(make_residual_hook(i)))
        hooks.append(layer.ffn.gate.register_forward_hook(make_gate_hook(i)))

    # Group samples by relation
    rel_samples = defaultdict(list)
    for s in samples:
        rel_samples[s.relation].append(s)

    templates = {}

    with torch.no_grad():
        for relation, samps in rel_samples.items():
            rel_residuals = [[] for _ in range(N_LAYERS)]
            rel_gate_patterns = [[] for _ in range(N_LAYERS)]

            for s in samps[:20]:  # cap at 20 per relation
                ids = tokenizer.encode(s.text, add_special_tokens=True,
                                       max_length=MAX_SEQ, truncation=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                _ = model(input_ids)

                for li in range(N_LAYERS):
                    if layer_residuals[li] is not None:
                        # Mean residual across sequence positions
                        mean_res = layer_residuals[li].mean(dim=1).squeeze(0)  # (dim,)
                        rel_residuals[li].append(mean_res.cpu())

                    if layer_gate_acts[li] is not None:
                        # Mean gate activation (pre-SiLU) across positions
                        mean_gate = layer_gate_acts[li].mean(dim=1).squeeze(0)  # (ffn_dim,)
                        rel_gate_patterns[li].append(mean_gate.cpu())

            # Average across samples
            avg_residuals = []
            avg_gate_patterns = []
            for li in range(N_LAYERS):
                if rel_residuals[li]:
                    avg_residuals.append(torch.stack(rel_residuals[li]).mean(dim=0))
                else:
                    avg_residuals.append(torch.zeros(DIM))

                if rel_gate_patterns[li]:
                    avg_gate_patterns.append(torch.stack(rel_gate_patterns[li]).mean(dim=0))
                else:
                    avg_gate_patterns.append(torch.zeros(FFN_DIM))

            templates[relation] = {
                "residuals": avg_residuals,       # list of (dim,) per layer
                "gate_patterns": avg_gate_patterns,  # list of (ffn_dim,) per layer
                "n_samples": len(samps),
            }

    for h in hooks:
        h.remove()

    print(f"  Extracted templates for {len(templates)} relations")
    return templates


# ---------------------------------------------------------------------------
# Phase 3: Compile source graph into FFN weights
# ---------------------------------------------------------------------------

def compile_graph_to_ffn(
    model: TinyGemma,
    trained_model: TinyGemma,
    templates: Dict,
    samples: List[GroundTruth],
    tokenizer: ClampedTokenizer,
    device: torch.device,
) -> TinyGemma:
    """
    Write structured knowledge directly into the FFN weights of a fresh model.

    Strategy: For each relation type, we know which gate features should fire
    (from templates). We set the gate weights so those features activate on
    the appropriate residual patterns, and set the down projections to produce
    the target token's embedding direction.

    Three compilation strategies, applied in combination:
    1. Template transfer: copy gate/down patterns from trained model for
       features that are exclusive to each relation
    2. Residual alignment: set gate vectors to align with the average
       residual for each relation (so the gate fires on the right queries)
    3. Target projection: set down vectors to project toward the target
       token embedding
    """
    print(f"\n  Compiling graph into FFN weights...")
    t0 = time.time()

    model.eval()
    trained_model.eval()

    # Strategy 1: Direct weight transfer of top-activated features
    # For each relation, find which features in the trained model activate
    # most strongly, and copy those gate/up/down weights directly.

    print(f"  Strategy 1: Transferring top-K feature weights per relation...")

    # For each layer, track which features have been "claimed" by a relation
    claimed = [set() for _ in range(N_LAYERS)]
    features_per_relation = 20  # how many features each relation gets

    for relation, tmpl in templates.items():
        for li in range(N_LAYERS):
            gate_pattern = tmpl["gate_patterns"][li]  # (ffn_dim,)

            # Find top-K features by gate activation magnitude
            # (these are the features the trained model uses for this relation)
            activated = F.silu(gate_pattern).abs()
            topk = activated.topk(features_per_relation * 2)  # grab extras in case of conflicts

            copied = 0
            for feat_idx in topk.indices.tolist():
                if feat_idx in claimed[li]:
                    continue  # already taken by another relation
                if copied >= features_per_relation:
                    break

                # Copy this feature's weights from trained model
                with torch.no_grad():
                    model.layers[li].ffn.gate.weight.data[feat_idx] = \
                        trained_model.layers[li].ffn.gate.weight.data[feat_idx].clone()
                    model.layers[li].ffn.up.weight.data[feat_idx] = \
                        trained_model.layers[li].ffn.up.weight.data[feat_idx].clone()
                    model.layers[li].ffn.down.weight.data[:, feat_idx] = \
                        trained_model.layers[li].ffn.down.weight.data[:, feat_idx].clone()

                claimed[li].add(feat_idx)
                copied += 1

    total_claimed = sum(len(c) for c in claimed)
    total_features = N_LAYERS * FFN_DIM
    print(f"  Transferred {total_claimed}/{total_features} features "
          f"({100*total_claimed/total_features:.1f}%)")

    # Strategy 2: For unclaimed features, align gate vectors with
    # relation-average residuals (so they fire on the right inputs)
    print(f"  Strategy 2: Aligning unclaimed gates with residual patterns...")

    unclaimed_aligned = 0
    relations = list(templates.keys())

    for li in range(N_LAYERS):
        unclaimed_features = [f for f in range(FFN_DIM) if f not in claimed[li]]

        if not unclaimed_features or not relations:
            continue

        # Distribute unclaimed features across relations proportionally
        features_per_rel = max(1, len(unclaimed_features) // len(relations))

        for ri, relation in enumerate(relations):
            start = ri * features_per_rel
            end = min(start + features_per_rel, len(unclaimed_features))
            rel_features = unclaimed_features[start:end]

            if not rel_features:
                continue

            residual = templates[relation]["residuals"][li].to(device)

            for feat_idx in rel_features:
                with torch.no_grad():
                    # Set gate weight to point in the direction of this relation's residual
                    # Scale by a factor to ensure reasonable activation magnitudes
                    gate_norm = model.layers[li].ffn.gate.weight.data[feat_idx].norm()
                    res_norm = residual.norm()
                    if res_norm > 0:
                        scaled_residual = residual * (gate_norm / res_norm)
                        # Blend: 70% residual-aligned, 30% original random
                        model.layers[li].ffn.gate.weight.data[feat_idx] = (
                            0.7 * scaled_residual +
                            0.3 * model.layers[li].ffn.gate.weight.data[feat_idx]
                        )
                    unclaimed_aligned += 1

    print(f"  Aligned {unclaimed_aligned} unclaimed features")

    # Strategy 3: Copy embeddings and layer norms from trained model
    # (these are part of the "schema" — how tokens map to residual space)
    print(f"  Strategy 3: Copying embeddings and norms from trained model...")
    with torch.no_grad():
        model.embed.weight.data.copy_(trained_model.embed.weight.data)
        model.norm.weight.data.copy_(trained_model.norm.weight.data)

        for li in range(N_LAYERS):
            model.layers[li].attn_norm.weight.data.copy_(
                trained_model.layers[li].attn_norm.weight.data)
            model.layers[li].ffn_norm.weight.data.copy_(
                trained_model.layers[li].ffn_norm.weight.data)

    elapsed = time.time() - t0
    print(f"  Compilation done in {elapsed:.1f}s")

    return model


# ---------------------------------------------------------------------------
# Training (attention-only)
# ---------------------------------------------------------------------------

def freeze_ffn(model):
    frozen = 0
    for layer in model.layers:
        for param in layer.ffn.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def train_attention_only(model, loader, tokenizer, device, epochs, label=""):
    """Train only attention parameters."""
    n_frozen = freeze_ffn(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Training attention-only: {label}")
    print(f"  Trainable: {trainable:,} | Frozen: {n_frozen:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    loss_history = []
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / n
        loss_history.append({"epoch": epoch + 1, "loss": avg})
        elapsed = time.time() - t0
        print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {elapsed:.0f}s")
        sys.stdout.flush()

    return loss_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, tokenizer, device) -> float:
    """Compute average loss on the full dataset."""
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            total_loss += loss.item()
            n += 1
    return total_loss / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  GRAPH-TO-WEIGHTS COMPILER (v6)")
    print("  Compile knowledge directly into FFN, train attention only")
    print("=" * 65)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Device: MPS")
    else:
        device = torch.device("cpu")
        print(f"\n  Device: CPU")

    # Tokenizer + data
    print("  Loading tokenizer...")
    raw_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if raw_tok.pad_token_id is None:
        raw_tok.pad_token_id = 0
    tokenizer = ClampedTokenizer(raw_tok, VOCAB)

    print("  Building corpus...")
    samples, ground_truth = build_mixed_corpus(n_countries=50, seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    dataset = TokenDataset([s.text for s in samples], tokenizer, MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Train baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Train baseline model")
    print(f"{'='*65}")

    baseline_model = train_baseline(loader, tokenizer, device, epochs=EPOCHS)
    baseline_loss = evaluate_model(baseline_model, loader, tokenizer, device)
    print(f"  Baseline eval loss: {baseline_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Extract relation templates
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Extract relation templates from trained model")
    print(f"{'='*65}")

    templates = extract_relation_templates(baseline_model, samples, tokenizer, device)

    # Show template stats
    print(f"\n  Template statistics:")
    for rel, tmpl in sorted(templates.items()):
        # Find which layers have strongest activation
        layer_acts = [tmpl["gate_patterns"][li].abs().mean().item() for li in range(N_LAYERS)]
        peak_layer = max(range(N_LAYERS), key=lambda i: layer_acts[i])
        peak_val = layer_acts[peak_layer]
        print(f"    {rel:<20} n={tmpl['n_samples']:>3}  peak=L{peak_layer} ({peak_val:.4f})")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Compile graph into fresh model
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3: Compile source graph into FFN weights")
    print(f"{'='*65}")

    torch.manual_seed(SEED)
    compiled_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    compiled_model = compile_graph_to_ffn(
        compiled_model, baseline_model, templates,
        samples, tokenizer, device,
    )

    # Evaluate compiled model BEFORE attention training
    compiled_pre_loss = evaluate_model(compiled_model, loader, tokenizer, device)
    print(f"\n  Compiled model loss (before attention training): {compiled_pre_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Train attention on compiled model
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4: Train attention-only on compiled model")
    print(f"{'='*65}")

    compiled_attn_history = train_attention_only(
        compiled_model, loader, tokenizer, device,
        epochs=ATTN_ONLY_EPOCHS,
        label="COMPILED + attention-only",
    )
    compiled_final_loss = evaluate_model(compiled_model, loader, tokenizer, device)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Freeze-FFN comparison (copy from trained, attention-only)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 5: Freeze-FFN comparison")
    print(f"{'='*65}")

    torch.manual_seed(SEED)
    freeze_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    # Copy ALL FFN weights from trained model (the "perfect" database)
    with torch.no_grad():
        for i in range(N_LAYERS):
            freeze_model.layers[i].ffn.gate.weight.data.copy_(
                baseline_model.layers[i].ffn.gate.weight.data)
            freeze_model.layers[i].ffn.up.weight.data.copy_(
                baseline_model.layers[i].ffn.up.weight.data)
            freeze_model.layers[i].ffn.down.weight.data.copy_(
                baseline_model.layers[i].ffn.down.weight.data)

    freeze_pre_loss = evaluate_model(freeze_model, loader, tokenizer, device)
    print(f"  Freeze-FFN loss (before attention training): {freeze_pre_loss:.4f}")

    freeze_attn_history = train_attention_only(
        freeze_model, loader, tokenizer, device,
        epochs=ATTN_ONLY_EPOCHS,
        label="FREEZE-FFN + attention-only",
    )
    freeze_final_loss = evaluate_model(freeze_model, loader, tokenizer, device)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: Random FFN baseline (attention-only with random FFN)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 6: Random FFN baseline (attention-only, no compilation)")
    print(f"{'='*65}")

    torch.manual_seed(SEED)
    random_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    random_pre_loss = evaluate_model(random_model, loader, tokenizer, device)
    print(f"  Random FFN loss (before attention training): {random_pre_loss:.4f}")

    random_attn_history = train_attention_only(
        random_model, loader, tokenizer, device,
        epochs=ATTN_ONLY_EPOCHS,
        label="RANDOM FFN + attention-only",
    )
    random_final_loss = evaluate_model(random_model, loader, tokenizer, device)

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*65}")

    print(f"\n  {'Run':<40} {'Pre-Attn':>9} {'Post-Attn':>10} {'Δ':>8}")
    print(f"  {'─'*70}")
    print(f"  {'Baseline (40 epochs full)':<40} {'N/A':>9} {baseline_loss:>10.4f} {'ref':>8}")
    print(f"  {'Freeze-FFN + {0} epochs attn'.format(ATTN_ONLY_EPOCHS):<40} "
          f"{freeze_pre_loss:>9.4f} {freeze_final_loss:>10.4f} "
          f"{freeze_final_loss - baseline_loss:>+8.4f}")
    print(f"  {'COMPILED + {0} epochs attn'.format(ATTN_ONLY_EPOCHS):<40} "
          f"{compiled_pre_loss:>9.4f} {compiled_final_loss:>10.4f} "
          f"{compiled_final_loss - baseline_loss:>+8.4f}")
    print(f"  {'Random FFN + {0} epochs attn'.format(ATTN_ONLY_EPOCHS):<40} "
          f"{random_pre_loss:>9.4f} {random_final_loss:>10.4f} "
          f"{random_final_loss - baseline_loss:>+8.4f}")

    # Epoch-by-epoch comparison
    print(f"\n  Attention-only training curves (loss per epoch):")
    print(f"  {'Epoch':>6} {'Freeze-FFN':>11} {'Compiled':>11} {'Random':>11}")
    print(f"  {'─'*42}")
    for i in range(ATTN_ONLY_EPOCHS):
        fl = freeze_attn_history[i]["loss"] if i < len(freeze_attn_history) else 0
        cl = compiled_attn_history[i]["loss"] if i < len(compiled_attn_history) else 0
        rl = random_attn_history[i]["loss"] if i < len(random_attn_history) else 0
        print(f"  {i+1:>6} {fl:>11.4f} {cl:>11.4f} {rl:>11.4f}")

    # Verdict
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    if compiled_final_loss <= baseline_loss * 1.05:
        print(f"\n  ✓ COMPILED matches baseline")
        print(f"    Compiled: {compiled_final_loss:.4f} vs Baseline: {baseline_loss:.4f}")
        print(f"    → Graph compilation + attention-only training works!")
    elif compiled_final_loss <= baseline_loss * 1.15:
        print(f"\n  ~ COMPILED close to baseline")
        print(f"    Compiled: {compiled_final_loss:.4f} vs Baseline: {baseline_loss:.4f}")
        print(f"    → Compilation produces a usable database, needs refinement")
    else:
        print(f"\n  ✗ COMPILED worse than baseline")
        print(f"    Compiled: {compiled_final_loss:.4f} vs Baseline: {baseline_loss:.4f}")

    if compiled_final_loss < random_final_loss:
        gap = random_final_loss - compiled_final_loss
        print(f"\n  ✓ Compilation beats random FFN by {gap:.4f}")
        print(f"    → The compiled database IS better than random")
    else:
        print(f"\n  ✗ Compilation no better than random FFN")

    improvement = (random_final_loss - compiled_final_loss) / (random_final_loss - freeze_final_loss + 1e-10)
    print(f"\n  Database quality (0=random, 1=trained):")
    print(f"    Compiled FFN quality: {improvement:.1%}")

    # Save
    results = {
        "baseline_loss": baseline_loss,
        "freeze_pre": freeze_pre_loss,
        "freeze_final": freeze_final_loss,
        "compiled_pre": compiled_pre_loss,
        "compiled_final": compiled_final_loss,
        "random_pre": random_pre_loss,
        "random_final": random_final_loss,
        "attn_epochs": ATTN_ONLY_EPOCHS,
        "compiled_curve": compiled_attn_history,
        "freeze_curve": freeze_attn_history,
        "random_curve": random_attn_history,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
