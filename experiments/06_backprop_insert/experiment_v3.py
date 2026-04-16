#!/usr/bin/env python3
"""
Backpropagation-is-INSERT: Experiment v3

Uses real Gemma tokenizer + WordNet syntax ground truth.
Streams results live during training.

Key measurements:
  1. Gradient band targeting: do syntax examples write to early layers,
     knowledge examples to middle layers?
  2. Contrastive: same gradient step, syntax vs knowledge — different layers?
  3. Feature exclusivity: do syntax and knowledge use different FFN features?
  4. Phase transition: when does band separation emerge during training?
"""

import os
import sys
import json
import time
import math
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
N_HEADS = 4
N_KV_HEADS = 2
MAX_SEQ = 64
EPOCHS = 40
BATCH_SIZE = 8
LR = 3e-4
N_COUNTRIES = 50
SEED = 42

# Band definitions
BANDS = {"syntax": (0, 3), "knowledge": (4, 7), "output": (8, 11)}

# When to measure
ANATOMY_STEPS = {0, 10, 25, 50, 100, 200, 500, 1000, 1500, 2000, 3000}
SAMPLES_PER_BAND = 10

OUTPUT_DIR = "results_v3"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int):
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
# Gradient Anatomy
# ---------------------------------------------------------------------------

def gradient_anatomy(model, sample, tokenizer, device):
    """Single-gradient anatomy: measure where one example writes."""
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
        ignore_index=tokenizer.pad_token_id or 0,
    )
    loss.backward()

    gate_disp = []
    down_disp = []
    top_features = []

    for layer in model.layers:
        g = layer.ffn.gate.weight.grad
        d = layer.ffn.down.weight.grad

        gn = g.norm().item() if g is not None else 0.0
        dn = d.norm().item() if d is not None else 0.0
        gate_disp.append(gn)
        down_disp.append(dn)

        if g is not None:
            feat_norms = g.norm(dim=1)
            topk = feat_norms.topk(min(5, len(feat_norms)))
            top_features.append(topk.indices.tolist())
        else:
            top_features.append([])

    model.zero_grad()

    total = [g + d for g, d in zip(gate_disp, down_disp)]
    total_sum = sum(total) or 1e-10

    band_energy = {}
    for bname, (lo, hi) in BANDS.items():
        band_energy[bname] = sum(total[lo:hi+1]) / total_sum

    target = sample.band if sample.band != "code" else "syntax"
    target_frac = band_energy.get(target, 0)

    return {
        "band": sample.band,
        "relation": sample.relation,
        "text": sample.text[:80],
        "loss": loss.item(),
        "total_displacement": total,
        "band_energy": band_energy,
        "target_band": target,
        "target_band_fraction": target_frac,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Contrastive Test
# ---------------------------------------------------------------------------

def contrastive_test(model, syn_sample, kn_sample, tokenizer, device):
    """Compare gradient profiles of syntax vs knowledge examples."""
    sr = gradient_anatomy(model, syn_sample, tokenizer, device)
    kr = gradient_anatomy(model, kn_sample, tokenizer, device)

    n = len(sr["total_displacement"])
    preference = []
    for i in range(n):
        s = sr["total_displacement"][i]
        k = kr["total_displacement"][i]
        t = s + k
        preference.append((s - k) / t if t > 0 else 0)

    # Feature overlap: are different features activated?
    overlap_per_layer = []
    for i in range(n):
        sf = set(sr["top_features"][i])
        kf = set(kr["top_features"][i])
        if sf and kf:
            overlap_per_layer.append(len(sf & kf) / len(sf | kf))
        else:
            overlap_per_layer.append(0)

    return {
        "preference": preference,
        "feature_overlap": overlap_per_layer,
        "syntax_bands": sr["band_energy"],
        "knowledge_bands": kr["band_energy"],
    }


# ---------------------------------------------------------------------------
# Live Display
# ---------------------------------------------------------------------------

def display_anatomy(results, step):
    """Print gradient anatomy results with visual bars."""
    W = 25  # bar width

    print(f"\n{'━'*65}")
    print(f"  ⚡ GRADIENT ANATOMY @ step {step}")
    print(f"{'━'*65}")

    for band in ["syntax", "knowledge", "code"]:
        band_r = [r for r in results if r["band"] == band]
        if not band_r:
            continue

        avg_disp = [0.0] * N_LAYERS
        for r in band_r:
            for i, d in enumerate(r["total_displacement"]):
                avg_disp[i] += d
        avg_disp = [d / len(band_r) for d in avg_disp]
        max_d = max(avg_disp) or 1.0

        target = band if band != "code" else "syntax"
        target_frac = sum(r["target_band_fraction"] for r in band_r) / len(band_r)

        print(f"\n  [{band.upper()}] → target: {target} band ({target_frac:.0%})")
        for i, d in enumerate(avg_disp):
            bar = int(W * d / max_d)
            bl = ""
            for bn, (lo, hi) in BANDS.items():
                if lo <= i <= hi:
                    bl = f" {bn[:3].upper()}"
            print(f"    L{i:2d} {'█' * bar}{'░' * (W - bar)} {d:.4f}{bl}")

    # Summary table
    print(f"\n  {'Input':<10} {'→Syntax':>8} {'→Knowl':>8} {'→Output':>8} {'Target%':>8}")
    print(f"  {'─'*42}")
    for band in ["syntax", "knowledge", "code"]:
        band_r = [r for r in results if r["band"] == band]
        if not band_r:
            continue
        avg_be = defaultdict(float)
        for r in band_r:
            for k, v in r["band_energy"].items():
                avg_be[k] += v
        for k in avg_be:
            avg_be[k] /= len(band_r)
        tf = sum(r["target_band_fraction"] for r in band_r) / len(band_r)
        print(f"  {band:<10} {avg_be['syntax']:>8.1%} {avg_be['knowledge']:>8.1%} "
              f"{avg_be['output']:>8.1%} {tf:>8.1%}")

    sys.stdout.flush()


def display_contrastive(results, step):
    """Print contrastive test results."""
    if not results:
        return

    avg_pref = [0.0] * N_LAYERS
    avg_overlap = [0.0] * N_LAYERS
    for r in results:
        for i in range(N_LAYERS):
            avg_pref[i] += r["preference"][i]
            avg_overlap[i] += r["feature_overlap"][i]
    n = len(results)
    avg_pref = [p / n for p in avg_pref]
    avg_overlap = [o / n for o in avg_overlap]

    print(f"\n  CONTRASTIVE (syntax+ / knowledge-) @ step {step}")
    for i in range(N_LAYERS):
        p = avg_pref[i]
        o = avg_overlap[i]
        bar_len = int(12 * abs(p))
        if p >= 0:
            bar = ' ' * 12 + '│' + '▶' * bar_len + ' ' * (12 - bar_len)
        else:
            bar = ' ' * (12 - bar_len) + '◀' * bar_len + '│' + ' ' * 12

        bl = ""
        for bn, (lo, hi) in BANDS.items():
            if lo <= i <= hi:
                bl = f" {bn[:3].upper()}"
        print(f"    L{i:2d} {bar} {p:+.3f}  overlap={o:.2f}{bl}")

    sys.stdout.flush()


def display_phase_table(all_anatomy):
    """Print phase transition table."""
    steps = sorted(set(r["step"] for r in all_anatomy))

    print(f"\n{'━'*65}")
    print(f"  PHASE TRANSITION")
    print(f"{'━'*65}")
    print(f"  {'Step':>6} {'Syn→Syn':>8} {'Kn→Kn':>8} {'Code→Syn':>9} {'Δ(S-K)':>8} {'Signal':>10}")

    for step in steps:
        by_band = defaultdict(list)
        for r in all_anatomy:
            if r["step"] == step:
                by_band[r["band"]].append(r["target_band_fraction"])

        sf = sum(by_band.get("syntax", [0])) / max(len(by_band.get("syntax", [1])), 1)
        kf = sum(by_band.get("knowledge", [0])) / max(len(by_band.get("knowledge", [1])), 1)
        cf = sum(by_band.get("code", [0])) / max(len(by_band.get("code", [1])), 1)
        delta = sf - kf

        sig = ""
        if delta > 0.15:
            sig = "SEPARATED"
        elif delta > 0.08:
            sig = "emerging"
        elif delta > 0.03:
            sig = "weak"

        print(f"  {step:>6} {sf:>8.3f} {kf:>8.3f} {cf:>9.3f} {delta:>+8.3f} {sig:>10}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  BACKPROPAGATION IS INSERT — v3")
    print("  Real Gemma tokenizer + WordNet ground truth")
    print("=" * 65)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n  Device: CPU")

    # Tokenizer
    print("  Loading Gemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Data
    print("  Building corpus (WordNet + synthetic KG + code)...")
    samples, ground_truth = build_mixed_corpus(n_countries=N_COUNTRIES, seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    with open(os.path.join(OUTPUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Check token counts
    token_lens = [len(tokenizer.encode(s.text)) for s in samples[:100]]
    print(f"  Avg tokens/sample: {sum(token_lens)/len(token_lens):.1f}")

    # Model — use Gemma's actual vocab size (262144) but that's too large
    # for a tiny model. Use a smaller embedding + hash trick.
    # Actually, let's just use a reasonable vocab size and map through it.
    VOCAB = min(tokenizer.vocab_size, 32000)  # Cap at 32K for memory
    print(f"  Using vocab size: {VOCAB} (capped from {tokenizer.vocab_size})")

    model = TinyGemma(
        vocab_size=VOCAB,
        dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq=MAX_SEQ,
    ).to(device)
    print(f"  Model params: {model.param_count():,}")

    # We need to clamp token IDs to vocab range
    class ClampedDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len, vocab_size):
            self.encodings = []
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=True,
                                       max_length=max_len, truncation=True)
                # Clamp to vocab range
                ids = [min(i, vocab_size - 1) for i in ids]
                self.encodings.append(ids)

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return torch.tensor(self.encodings[idx], dtype=torch.long)

    dataset = ClampedDataset([s.text for s in samples], tokenizer, MAX_SEQ, VOCAB)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Wrap tokenizer encode to clamp
    class ClampedTokenizer:
        def __init__(self, tok, vocab):
            self.tok = tok
            self.vocab = vocab
            self.pad_token_id = tok.pad_token_id
        def encode(self, text, **kwargs):
            ids = self.tok.encode(text, **kwargs)
            return [min(i, self.vocab - 1) for i in ids]

    clamped_tok = ClampedTokenizer(tokenizer, VOCAB)

    # Anatomy samples (fixed)
    rng = random.Random(SEED)
    band_map = defaultdict(list)
    for s in samples:
        band_map[s.band].append(s)

    anatomy_samps = {}
    for band in ["syntax", "knowledge", "code"]:
        pool = band_map[band]
        anatomy_samps[band] = rng.sample(pool, min(SAMPLES_PER_BAND, len(pool)))

    # ─── Training ───
    all_anatomy = []
    all_contrastive = []
    loss_history = []
    step = 0

    print(f"\n{'='*65}")
    print(f"  TRAINING — {EPOCHS} epochs, {len(loader)} batches/epoch")
    print(f"{'='*65}")
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_batch = 0

        for batch in loader:
            batch = batch.to(device)

            # Anatomy checkpoint
            if step in ANATOMY_STEPS:
                step_results = []
                for band, samps in anatomy_samps.items():
                    for s in samps:
                        r = gradient_anatomy(model, s, clamped_tok, device)
                        r["step"] = step
                        step_results.append(r)
                        all_anatomy.append(r)

                display_anatomy(step_results, step)

                # Contrastive
                n_pairs = min(5, len(anatomy_samps["syntax"]), len(anatomy_samps["knowledge"]))
                step_contrastive = []
                for i in range(n_pairs):
                    cr = contrastive_test(
                        model,
                        anatomy_samps["syntax"][i],
                        anatomy_samps["knowledge"][i],
                        clamped_tok, device,
                    )
                    cr["step"] = step
                    step_contrastive.append(cr)
                    all_contrastive.append(cr)

                display_contrastive(step_contrastive, step)

            # Training step
            model.train()
            optimizer.zero_grad()

            logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id or 0,
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
        print(f"\n  Epoch {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  step={step}  {elapsed:.0f}s")
        sys.stdout.flush()

    # Final anatomy
    print(f"\n{'='*65}")
    print(f"  FINAL MEASUREMENTS @ step {step}")
    print(f"{'='*65}")

    final_results = []
    for band, samps in anatomy_samps.items():
        for s in samps:
            r = gradient_anatomy(model, s, clamped_tok, device)
            r["step"] = step
            final_results.append(r)
            all_anatomy.append(r)

    display_anatomy(final_results, step)

    # Phase transition
    display_phase_table(all_anatomy)

    # Cross-band leakage summary
    print(f"\n{'━'*65}")
    print(f"  CROSS-BAND LEAKAGE (final step)")
    print(f"{'━'*65}")

    for band in ["syntax", "knowledge", "code"]:
        br = [r for r in all_anatomy if r["step"] == step and r["band"] == band]
        if not br:
            continue
        tf = sum(r["target_band_fraction"] for r in br) / len(br)
        leak = 1 - tf
        status = "PASS" if leak < 0.4 else "PARTIAL" if leak < 0.6 else "FAIL"
        print(f"  {band:<10} target_band={tf:.1%}  leakage={leak:.1%}  [{status}]")

    # Save
    with open(os.path.join(OUTPUT_DIR, "anatomy.json"), "w") as f:
        json.dump(all_anatomy, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "contrastive.json"), "w") as f:
        json.dump(all_contrastive, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "loss.json"), "w") as f:
        json.dump(loss_history, f)

    total_time = time.time() - t0
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
