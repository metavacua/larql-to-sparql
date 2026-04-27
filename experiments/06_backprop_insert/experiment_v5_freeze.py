#!/usr/bin/env python3
"""
Freeze-FFN Experiment: Is the FFN database separable from the attention query engine?

Three runs:
  Run 1: BASELINE    — full training (all params)
  Run 2: FREEZE-FFN  — copy FFN weights from trained model, freeze them, train attention only
  Run 3: PROGRESSIVE — train all params for 5 epochs, then freeze FFN for remaining 35

If Freeze-FFN converges fast → FFN and attention are separable.
If Progressive matches baseline → the volatile phase (5 epochs) is all the FFN needs.
"""

import os
import sys
import json
import time
import copy
import random
from collections import defaultdict
from typing import List, Dict

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
VOCAB = 32000
PROGRESSIVE_FREEZE_EPOCH = 5  # freeze FFN after this epoch

OUTPUT_DIR = "results_v5_freeze"

# Feature tracking
BANDS = {"syntax": (0, 3), "knowledge": (4, 7), "output": (8, 11)}
TRACK_EVERY = 50
TOP_K = 10


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
# Feature stability tracker (lightweight)
# ---------------------------------------------------------------------------

class StabilityTracker:
    """Track top-K feature indices per relation over training."""

    def __init__(self, n_layers, ffn_dim):
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.relation_energy = defaultdict(
            lambda: torch.zeros(n_layers, ffn_dim)
        )
        self.prev_topk = {}  # relation → {layer → set of indices}
        self.stability_history = []  # list of (step, {relation → jaccard})

    def record_gradient(self, model, relation):
        for li, layer in enumerate(model.layers):
            g = layer.ffn.gate.weight.grad
            if g is not None:
                self.relation_energy[relation][li] += g.norm(dim=1).cpu()

    def snapshot(self, step):
        """Compute stability vs previous snapshot."""
        curr_topk = {}
        for rel, energy in self.relation_energy.items():
            curr_topk[rel] = {}
            for li in range(self.n_layers):
                topk_idx = energy[li].topk(min(TOP_K, self.ffn_dim)).indices
                curr_topk[rel][li] = set(topk_idx.tolist())

        if self.prev_topk:
            jaccards = {}
            for rel in curr_topk:
                if rel not in self.prev_topk:
                    continue
                layer_j = []
                for li in range(self.n_layers):
                    prev = self.prev_topk[rel].get(li, set())
                    curr = curr_topk[rel].get(li, set())
                    if prev or curr:
                        j = len(prev & curr) / len(prev | curr)
                    else:
                        j = 1.0
                    layer_j.append(j)
                jaccards[rel] = sum(layer_j) / len(layer_j)
            self.stability_history.append({"step": step, "jaccards": jaccards})

        self.prev_topk = curr_topk


def gradient_anatomy_batch(model, samples, tokenizer, device):
    """Quick gradient anatomy: measure band energy distribution."""
    results = []
    for sample in samples:
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
            ignore_index=0,
        )
        loss.backward()

        total = []
        for layer in model.layers:
            g = layer.ffn.gate.weight.grad
            d = layer.ffn.down.weight.grad
            gn = g.norm().item() if g is not None else 0
            dn = d.norm().item() if d is not None else 0
            total.append(gn + dn)

        total_sum = sum(total) or 1e-10
        band_energy = {}
        for bn, (lo, hi) in BANDS.items():
            band_energy[bn] = sum(total[lo:hi+1]) / total_sum

        target = sample.band if sample.band != "code" else "syntax"
        results.append({
            "band": sample.band,
            "relation": sample.relation,
            "target_band_fraction": band_energy.get(target, 0),
            "band_energy": band_energy,
        })
        model.zero_grad()

    return results


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_ffn(model):
    """Freeze all FFN parameters."""
    frozen = 0
    for layer in model.layers:
        for param in layer.ffn.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def train_run(
    run_name: str,
    model: TinyGemma,
    loader: DataLoader,
    tokenizer: ClampedTokenizer,
    samples: List[GroundTruth],
    device: torch.device,
    freeze_mode: str = "none",  # "none", "from_start", "progressive"
    progressive_epoch: int = 5,
) -> Dict:
    """
    Run one training experiment.
    freeze_mode:
      "none" — baseline, all params train
      "from_start" — FFN frozen from epoch 0
      "progressive" — FFN frozen after progressive_epoch
    """
    print(f"\n{'='*65}")
    print(f"  RUN: {run_name}")
    print(f"  freeze_mode={freeze_mode}, trainable={count_trainable(model):,}, "
          f"frozen={count_frozen(model):,}")
    print(f"{'='*65}")
    sys.stdout.flush()

    if freeze_mode == "from_start":
        n_frozen = freeze_ffn(model)
        print(f"  FFN frozen: {n_frozen:,} params")
        print(f"  Trainable: {count_trainable(model):,} params")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    # Anatomy samples
    rng = random.Random(SEED)
    band_map = defaultdict(list)
    for s in samples:
        band_map[s.band].append(s)
    anatomy_samps = []
    for band in ["syntax", "knowledge", "code"]:
        anatomy_samps.extend(rng.sample(band_map[band], min(5, len(band_map[band]))))

    # Relation tracking samples
    rel_map = defaultdict(list)
    for s in samples:
        rel_map[s.relation].append(s)
    track_samples = {}
    for rel, pool in rel_map.items():
        track_samples[rel] = rng.sample(pool, min(3, len(pool)))

    tracker = StabilityTracker(N_LAYERS, FFN_DIM)

    loss_history = []
    anatomy_history = []
    step = 0
    t0 = time.time()
    ffn_frozen = (freeze_mode == "from_start")

    for epoch in range(EPOCHS):
        # Progressive freeze check
        if freeze_mode == "progressive" and epoch == progressive_epoch and not ffn_frozen:
            n_frozen = freeze_ffn(model)
            ffn_frozen = True
            # Rebuild optimizer with only trainable params
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LR, weight_decay=0.01,
            )
            print(f"\n  *** FFN FROZEN at epoch {epoch} ***")
            print(f"  Frozen: {n_frozen:,} | Trainable: {count_trainable(model):,}")

        epoch_loss = 0
        n_batch = 0

        for batch in loader:
            batch = batch.to(device)

            # Feature tracking
            if step % TRACK_EVERY == 0 and not ffn_frozen:
                for rel, samps in track_samples.items():
                    for s in samps:
                        model.train()
                        model.zero_grad()
                        ids = tokenizer.encode(s.text, add_special_tokens=True,
                                               max_length=MAX_SEQ, truncation=True)
                        inp = torch.tensor([ids], dtype=torch.long, device=device)
                        logits = model(inp)
                        loss_tmp = F.cross_entropy(
                            logits[:, :-1, :].contiguous().view(-1, VOCAB),
                            inp[:, 1:].contiguous().view(-1),
                            ignore_index=0,
                        )
                        loss_tmp.backward()
                        tracker.record_gradient(model, rel)
                        model.zero_grad()
                tracker.snapshot(step)

            # Training step
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

        # Anatomy at epoch boundaries
        anatomy = gradient_anatomy_batch(model, anatomy_samps, tokenizer, device)
        by_band = defaultdict(list)
        for a in anatomy:
            by_band[a["band"]].append(a["target_band_fraction"])
        syn_tf = sum(by_band.get("syntax", [0])) / max(len(by_band.get("syntax", [1])), 1)
        kn_tf = sum(by_band.get("knowledge", [0])) / max(len(by_band.get("knowledge", [1])), 1)

        anatomy_history.append({
            "epoch": epoch + 1,
            "step": step,
            "loss": avg_loss,
            "syn_target": syn_tf,
            "kn_target": kn_tf,
        })

        freeze_marker = " [FFN FROZEN]" if ffn_frozen else ""
        print(f"  E{epoch+1:2d}/{EPOCHS} loss={avg_loss:.4f} "
              f"syn→syn={syn_tf:.3f} kn→kn={kn_tf:.3f} "
              f"{elapsed:.0f}s{freeze_marker}")
        sys.stdout.flush()

    # Stability summary
    stability_summary = {}
    if tracker.stability_history:
        last = tracker.stability_history[-1]["jaccards"]
        stability_summary = last

    return {
        "run_name": run_name,
        "freeze_mode": freeze_mode,
        "final_loss": loss_history[-1]["loss"] if loss_history else 0,
        "avg_final_loss": anatomy_history[-1]["loss"] if anatomy_history else 0,
        "loss_history": loss_history,
        "anatomy_history": anatomy_history,
        "stability": stability_summary,
        "trainable_params": count_trainable(model),
        "total_time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  FREEZE-FFN EXPERIMENT")
    print("  Is the FFN database separable from the attention query engine?")
    print("=" * 65)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Device: MPS")
    else:
        device = torch.device("cpu")
        print(f"\n  Device: CPU")

    # Tokenizer
    print("  Loading tokenizer...")
    raw_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if raw_tok.pad_token_id is None:
        raw_tok.pad_token_id = 0
    tokenizer = ClampedTokenizer(raw_tok, VOCAB)

    # Data
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
    # RUN 1: BASELINE (full training)
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(SEED)
    baseline_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)
    print(f"\n  Model params: {baseline_model.param_count():,}")

    baseline_result = train_run(
        "BASELINE (full training)",
        baseline_model, loader, tokenizer, samples, device,
        freeze_mode="none",
    )

    # Save trained FFN weights for the freeze experiment
    trained_ffn_state = {}
    for i, layer in enumerate(baseline_model.layers):
        trained_ffn_state[f"layer_{i}_gate"] = layer.ffn.gate.weight.data.clone()
        trained_ffn_state[f"layer_{i}_up"] = layer.ffn.up.weight.data.clone()
        trained_ffn_state[f"layer_{i}_down"] = layer.ffn.down.weight.data.clone()

    # ═══════════════════════════════════════════════════════════════
    # RUN 2: FREEZE-FFN (copy FFN from trained, freeze, train attention only)
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(SEED)
    freeze_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    # Copy trained FFN weights into fresh model
    for i, layer in enumerate(freeze_model.layers):
        layer.ffn.gate.weight.data.copy_(trained_ffn_state[f"layer_{i}_gate"])
        layer.ffn.up.weight.data.copy_(trained_ffn_state[f"layer_{i}_up"])
        layer.ffn.down.weight.data.copy_(trained_ffn_state[f"layer_{i}_down"])

    freeze_result = train_run(
        "FREEZE-FFN (trained FFN, attention-only training)",
        freeze_model, loader, tokenizer, samples, device,
        freeze_mode="from_start",
    )

    # ═══════════════════════════════════════════════════════════════
    # RUN 3: PROGRESSIVE FREEZE (5 epochs full, then freeze FFN)
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(SEED)
    progressive_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)

    progressive_result = train_run(
        f"PROGRESSIVE (freeze FFN after epoch {PROGRESSIVE_FREEZE_EPOCH})",
        progressive_model, loader, tokenizer, samples, device,
        freeze_mode="progressive",
        progressive_epoch=PROGRESSIVE_FREEZE_EPOCH,
    )

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  COMPARISON")
    print(f"{'='*65}")

    results = [baseline_result, freeze_result, progressive_result]

    print(f"\n  {'Run':<45} {'Final Loss':>10} {'Time':>7} {'Trainable':>12}")
    print(f"  {'─'*75}")
    for r in results:
        print(f"  {r['run_name']:<45} {r['avg_final_loss']:>10.4f} "
              f"{r['total_time']:>6.0f}s {r['trainable_params']:>12,}")

    # Loss curve comparison at key epochs
    print(f"\n  Loss at key epochs:")
    print(f"  {'Epoch':>6} {'Baseline':>10} {'Freeze-FFN':>11} {'Progressive':>12}")
    print(f"  {'─'*42}")

    for epoch_idx in [0, 4, 9, 14, 19, 24, 29, 34, 39]:
        if epoch_idx >= EPOCHS:
            continue
        vals = []
        for r in results:
            if epoch_idx < len(r["anatomy_history"]):
                vals.append(f"{r['anatomy_history'][epoch_idx]['loss']:>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {epoch_idx+1:>6} {'  '.join(vals)}")

    # Convergence speed: epochs to reach baseline's epoch-10 loss
    baseline_e10_loss = baseline_result["anatomy_history"][9]["loss"] if len(baseline_result["anatomy_history"]) > 9 else None
    if baseline_e10_loss:
        print(f"\n  Epochs to reach baseline epoch-10 loss ({baseline_e10_loss:.4f}):")
        for r in results:
            reached = None
            for ah in r["anatomy_history"]:
                if ah["loss"] <= baseline_e10_loss:
                    reached = ah["epoch"]
                    break
            if reached:
                print(f"    {r['run_name']:<45} epoch {reached}")
            else:
                print(f"    {r['run_name']:<45} NOT REACHED")

    # Band targeting comparison
    print(f"\n  Band targeting at convergence:")
    print(f"  {'Run':<45} {'Syn→Syn':>8} {'Kn→Kn':>8}")
    print(f"  {'─'*62}")
    for r in results:
        if r["anatomy_history"]:
            last = r["anatomy_history"][-1]
            print(f"  {r['run_name']:<45} {last['syn_target']:>8.3f} {last['kn_target']:>8.3f}")

    # Verdict
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    bl = baseline_result["avg_final_loss"]
    fr = freeze_result["avg_final_loss"]
    pr = progressive_result["avg_final_loss"]

    if fr <= bl * 1.05:
        print(f"\n  ✓ FREEZE-FFN matches baseline ({fr:.4f} vs {bl:.4f})")
        print(f"    → FFN and attention ARE separable")
        print(f"    → Training time is dominated by FFN construction")
    elif fr <= bl * 1.2:
        print(f"\n  ~ FREEZE-FFN close to baseline ({fr:.4f} vs {bl:.4f})")
        print(f"    → FFN and attention are partially separable")
    else:
        print(f"\n  ✗ FREEZE-FFN worse than baseline ({fr:.4f} vs {bl:.4f})")
        print(f"    → FFN and attention co-adapt")

    if pr <= bl * 1.05:
        print(f"\n  ✓ PROGRESSIVE matches baseline ({pr:.4f} vs {bl:.4f})")
        print(f"    → {PROGRESSIVE_FREEZE_EPOCH} epochs is enough for FFN")
    elif pr <= bl * 1.2:
        print(f"\n  ~ PROGRESSIVE close ({pr:.4f} vs {bl:.4f})")
    else:
        print(f"\n  ✗ PROGRESSIVE worse ({pr:.4f} vs {bl:.4f})")

    # Save
    for r in results:
        name = r["run_name"].split("(")[0].strip().lower().replace(" ", "_").replace("-", "_")
        with open(os.path.join(OUTPUT_DIR, f"{name}_loss.json"), "w") as f:
            json.dump(r["loss_history"], f)
        with open(os.path.join(OUTPUT_DIR, f"{name}_anatomy.json"), "w") as f:
            json.dump(r["anatomy_history"], f, indent=2)

    summary = {
        "baseline_final_loss": bl,
        "freeze_ffn_final_loss": fr,
        "progressive_final_loss": pr,
        "baseline_time": baseline_result["total_time"],
        "freeze_time": freeze_result["total_time"],
        "progressive_time": progressive_result["total_time"],
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR}/")
    print(f"  Total time: {sum(r['total_time'] for r in results):.0f}s")


if __name__ == "__main__":
    main()
