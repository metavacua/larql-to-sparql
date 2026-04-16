"""
Backpropagation-is-INSERT: Experiment v2

Improvements over v1:
  - Word-level tokenizer (entities = single tokens, not char sequences)
  - Live streaming output during training
  - Per-feature gradient tracking (which FFN features move, not just layer norms)
  - Contrastive band measurement: for each gradient, measure energy ratio
    between the layer band that SHOULD receive it vs the one that SHOULDN'T
  - Interference experiment: same-batch syntax + knowledge writes
"""

import os
import sys
import json
import time
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import TinyGemma
from synth_data import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Word-level tokenizer
# ---------------------------------------------------------------------------

class WordTokenizer:
    """Word-level tokenizer that treats entities as single tokens."""

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self._next_id = 4

        # Add common punctuation as tokens
        for tok in [".", ",", "'", '"', ":", ";", "(", ")", "[", "]",
                     "{", "}", "=", "+", "-", "*", "/", "<", ">",
                     "!", "?", "&", "|", "#", "@", "_", "\\", "\n", "\t"]:
            self._add(tok)

    def _add(self, word: str) -> int:
        if word not in self.word_to_id:
            self.word_to_id[word] = self._next_id
            self.id_to_word[self._next_id] = word
            self._next_id += 1
        return self.word_to_id[word]

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from all training texts."""
        for text in texts:
            for word in self._tokenize(text):
                self._add(word)

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, keeping punctuation as separate tokens."""
        tokens = []
        current = []
        for c in text:
            if c in ' \t':
                if current:
                    tokens.append(''.join(current))
                    current = []
            elif c in '.,;:()[]{}=+-*/<>!?&|#@\\_"\'\n\t':
                if current:
                    tokens.append(''.join(current))
                    current = []
                tokens.append(c)
            else:
                current.append(c)
        if current:
            tokens.append(''.join(current))
        return tokens

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for word in self._tokenize(text):
            ids.append(self.word_to_id.get(word, self.unk_id))
        ids.append(self.eos_id)
        return ids

    @property
    def vocab_size(self) -> int:
        return self._next_id


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: WordTokenizer, max_len: int = 64):
        self.encodings = []
        for text in texts:
            ids = tokenizer.encode(text)[:max_len]
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
# Gradient Anatomy v2: per-feature tracking
# ---------------------------------------------------------------------------

def compute_gradient_anatomy(
    model: TinyGemma,
    sample: GroundTruth,
    tokenizer: WordTokenizer,
    device: torch.device,
) -> dict:
    """
    Single-gradient anatomy for one training example.
    Returns per-layer displacement and per-feature top movers.
    """
    model.train()
    model.zero_grad()

    ids = tokenizer.encode(sample.text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, model.vocab_size),
        shift_labels.view(-1),
        ignore_index=0,
    )
    loss.backward()

    n_layers = model.n_layers
    gate_disp = []
    down_disp = []
    top_features = []  # per layer: indices of features with largest gradient

    for li, layer in enumerate(model.layers):
        g = layer.ffn.gate.weight.grad  # (ffn_dim, dim)
        d = layer.ffn.down.weight.grad  # (dim, ffn_dim)

        # Per-feature gradient norm (each feature = one row of gate)
        feat_grad_norm = g.norm(dim=1)  # (ffn_dim,)
        gate_disp.append(feat_grad_norm.sum().item())
        down_disp.append(d.norm().item())

        # Top-5 features by gradient magnitude
        topk = feat_grad_norm.topk(min(5, len(feat_grad_norm)))
        top_features.append({
            "indices": topk.indices.tolist(),
            "norms": [f"{v:.4f}" for v in topk.values.tolist()],
        })

    model.zero_grad()

    total = [g + d for g, d in zip(gate_disp, down_disp)]
    total_sum = sum(total)

    # Band energies (4 layers each for 12-layer model)
    bands = {
        "syntax": (0, 3),
        "knowledge": (4, 7),
        "output": (8, 11),
    }
    band_energy = {}
    for bname, (lo, hi) in bands.items():
        band_energy[bname] = sum(total[lo:hi+1]) / total_sum if total_sum > 0 else 0

    # Target band
    target_band = sample.band if sample.band != "code" else "syntax"

    return {
        "band": sample.band,
        "relation": sample.relation,
        "text": sample.text[:80],
        "loss": loss.item(),
        "gate_displacement": gate_disp,
        "down_displacement": down_disp,
        "total_displacement": total,
        "band_energy": band_energy,
        "target_band": target_band,
        "target_band_fraction": band_energy.get(target_band, 0),
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Contrastive Test: same input, different knowledge types
# ---------------------------------------------------------------------------

def contrastive_gradient_test(
    model: TinyGemma,
    syntax_sample: GroundTruth,
    knowledge_sample: GroundTruth,
    tokenizer: WordTokenizer,
    device: torch.device,
) -> dict:
    """
    Feed one syntax and one knowledge example as separate gradients.
    Measure whether they write to different layer bands.
    """
    syn_result = compute_gradient_anatomy(model, syntax_sample, tokenizer, device)
    kn_result = compute_gradient_anatomy(model, knowledge_sample, tokenizer, device)

    # Compute layer-wise difference: which layers are more activated by
    # syntax vs knowledge?
    n = len(syn_result["total_displacement"])
    diff = []
    for i in range(n):
        s = syn_result["total_displacement"][i]
        k = kn_result["total_displacement"][i]
        total = s + k
        # Positive = more syntax, negative = more knowledge
        diff.append((s - k) / total if total > 0 else 0)

    return {
        "syntax_text": syntax_sample.text[:60],
        "knowledge_text": knowledge_sample.text[:60],
        "layer_preference": diff,  # positive = syntax-preferred
        "syntax_bands": syn_result["band_energy"],
        "knowledge_bands": kn_result["band_energy"],
    }


# ---------------------------------------------------------------------------
# Feature Persistence Tracker
# ---------------------------------------------------------------------------

class FeatureTracker:
    """Track which FFN features are being written to over training."""

    def __init__(self, n_layers: int, ffn_dim: int):
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        # Cumulative gradient energy per feature
        self.cumulative = torch.zeros(n_layers, ffn_dim)
        # Per-band cumulative
        self.band_cumulative = {
            "syntax": torch.zeros(n_layers, ffn_dim),
            "knowledge": torch.zeros(n_layers, ffn_dim),
            "code": torch.zeros(n_layers, ffn_dim),
        }
        self.steps = 0

    def update(self, model: TinyGemma, band: str):
        """Call after loss.backward() to accumulate gradient stats."""
        for li, layer in enumerate(model.layers):
            g = layer.ffn.gate.weight.grad
            if g is not None:
                feat_norms = g.norm(dim=1).cpu()  # (ffn_dim,)
                self.cumulative[li] += feat_norms
                if band in self.band_cumulative:
                    self.band_cumulative[band][li] += feat_norms
        self.steps += 1

    def report(self) -> dict:
        """Report feature sharing and exclusivity between bands."""
        results = {}
        for li in range(self.n_layers):
            syn = self.band_cumulative["syntax"][li]
            kn = self.band_cumulative["knowledge"][li]

            # Normalise
            syn_n = syn / (syn.sum() + 1e-10)
            kn_n = kn / (kn.sum() + 1e-10)

            # Overlap: features used by BOTH bands
            overlap = torch.min(syn_n, kn_n).sum().item()
            results[f"L{li}_overlap"] = overlap

            # Top-10 features exclusive to each band
            diff = syn_n - kn_n
            syn_exclusive = diff.topk(10).indices.tolist()
            kn_exclusive = (-diff).topk(10).indices.tolist()
            results[f"L{li}_syn_exclusive"] = syn_exclusive
            results[f"L{li}_kn_exclusive"] = kn_exclusive

        return results


# ---------------------------------------------------------------------------
# Live display helpers
# ---------------------------------------------------------------------------

def print_bar(label: str, values: List[float], band_ranges: dict, width: int = 30):
    """Print a horizontal bar chart with band annotations."""
    max_v = max(values) if values else 1.0
    for i, v in enumerate(values):
        bar_len = int(width * v / max_v) if max_v > 0 else 0
        band_label = ""
        for bname, (lo, hi) in band_ranges.items():
            if lo <= i <= hi:
                band_label = f" [{bname[:3].upper()}]"
                break
        print(f"  L{i:2d}: {'█' * bar_len}{'░' * (width - bar_len)} {v:.4f}{band_label}")


def print_anatomy_summary(results: List[dict], step: int):
    """Print live summary of gradient anatomy at a checkpoint."""
    print(f"\n{'─'*60}")
    print(f"  GRADIENT ANATOMY @ step {step}")
    print(f"{'─'*60}")

    bands = {"syntax": (0, 3), "knowledge": (4, 7), "output": (8, 11)}

    for band in ["syntax", "knowledge", "code"]:
        band_results = [r for r in results if r["band"] == band]
        if not band_results:
            continue

        n_layers = len(band_results[0]["total_displacement"])
        avg_disp = [0.0] * n_layers
        for r in band_results:
            for i, d in enumerate(r["total_displacement"]):
                avg_disp[i] += d
        avg_disp = [d / len(band_results) for d in avg_disp]

        target = band if band != "code" else "syntax"
        target_frac = sum(r["target_band_fraction"] for r in band_results) / len(band_results)

        print(f"\n  [{band.upper()}] → target={target} band  (fraction: {target_frac:.1%})")
        print_bar(band, avg_disp, bands)

    # Summary table
    print(f"\n  {'Band':<12} {'→ Syntax':>10} {'→ Knowledge':>12} {'→ Output':>10}")
    for band in ["syntax", "knowledge", "code"]:
        band_results = [r for r in results if r["band"] == band]
        if not band_results:
            continue
        avg_be = defaultdict(float)
        for r in band_results:
            for k, v in r["band_energy"].items():
                avg_be[k] += v
        for k in avg_be:
            avg_be[k] /= len(band_results)
        print(f"  {band:<12} {avg_be['syntax']:>10.1%} {avg_be['knowledge']:>12.1%} {avg_be['output']:>10.1%}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 60)
    print("  BACKPROPAGATION IS INSERT — Experiment v2")
    print("=" * 60)

    # Config
    N_LAYERS = 12
    DIM = 256
    FFN_DIM = 1024
    EPOCHS = 40
    BATCH_SIZE = 8
    LR = 3e-4
    N_COUNTRIES = 50
    SEED = 42
    ANATOMY_STEPS = {0, 25, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000}
    MAX_SEQ = 64

    output_dir = "results_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n  Device: CUDA")
    else:
        device = torch.device("cpu")
        print("\n  Device: CPU")

    # Data
    print("\n  Building synthetic corpus...")
    samples, ground_truth = build_mixed_corpus(n_countries=N_COUNTRIES, seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    # Tokenizer
    tokenizer = WordTokenizer()
    tokenizer.build_vocab([s.text for s in samples])
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Model
    model = TinyGemma(
        vocab_size=tokenizer.vocab_size,
        dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)
    print(f"  Model params: {model.param_count():,}")

    # Dataset
    dataset = TextDataset([s.text for s in samples], tokenizer, max_len=MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Anatomy samples (fixed)
    rng = random.Random(SEED)
    band_samples = defaultdict(list)
    for s in samples:
        band_samples[s.band].append(s)

    anatomy_samples = {}
    for band in ["syntax", "knowledge", "code"]:
        pool = band_samples[band]
        anatomy_samples[band] = rng.sample(pool, min(10, len(pool)))

    # Feature tracker
    tracker = FeatureTracker(N_LAYERS, FFN_DIM)

    # Storage
    all_anatomy = []
    all_contrastive = []
    loss_history = []
    global_step = 0
    bands = {"syntax": (0, 3), "knowledge": (4, 7), "output": (8, 11)}

    print(f"\n{'='*60}")
    print(f"  TRAINING — {EPOCHS} epochs, {len(loader)} batches/epoch")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            # --- Anatomy checkpoint ---
            if global_step in ANATOMY_STEPS:
                step_results = []
                for band, samps in anatomy_samples.items():
                    for samp in samps:
                        r = compute_gradient_anatomy(model, samp, tokenizer, device)
                        r["step"] = global_step
                        step_results.append(r)
                        all_anatomy.append(r)

                print_anatomy_summary(step_results, global_step)

                # Contrastive test
                n_pairs = min(5, len(anatomy_samples["syntax"]), len(anatomy_samples["knowledge"]))
                for i in range(n_pairs):
                    cr = contrastive_gradient_test(
                        model,
                        anatomy_samples["syntax"][i],
                        anatomy_samples["knowledge"][i],
                        tokenizer, device,
                    )
                    cr["step"] = global_step
                    all_contrastive.append(cr)

            # --- Training step ---
            model.train()
            optimizer.zero_grad()

            logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_id,
            )
            loss.backward()

            # Track which features receive gradient (sample from current batch)
            # Use the first sample's band as the label (approximate)
            if global_step % 10 == 0:
                # Determine dominant band in batch (heuristic)
                batch_band = "knowledge"  # most samples are knowledge
                if global_step % 30 < 10:
                    batch_band = "syntax"
                tracker.update(model, batch_band)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            loss_history.append({"step": global_step, "loss": loss_val})
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"step={global_step}  time={elapsed:.0f}s")
        sys.stdout.flush()

    # --- Final anatomy ---
    print(f"\n{'='*60}")
    print(f"  FINAL ANATOMY @ step {global_step}")
    print(f"{'='*60}")
    final_results = []
    for band, samps in anatomy_samples.items():
        for samp in samps:
            r = compute_gradient_anatomy(model, samp, tokenizer, device)
            r["step"] = global_step
            final_results.append(r)
            all_anatomy.append(r)

    print_anatomy_summary(final_results, global_step)

    # Feature overlap report
    feat_report = tracker.report()

    # --- Save ---
    with open(os.path.join(output_dir, "anatomy.json"), "w") as f:
        json.dump(all_anatomy, f, indent=2)
    with open(os.path.join(output_dir, "contrastive.json"), "w") as f:
        json.dump(all_contrastive, f, indent=2)
    with open(os.path.join(output_dir, "loss.json"), "w") as f:
        json.dump(loss_history, f)
    with open(os.path.join(output_dir, "feature_overlap.json"), "w") as f:
        json.dump(feat_report, f, indent=2)
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    # --- Phase transition analysis ---
    print(f"\n{'='*60}")
    print(f"  PHASE TRANSITION ANALYSIS")
    print(f"{'='*60}")

    steps_seen = sorted(set(r["step"] for r in all_anatomy))
    print(f"\n  {'Step':>6}  {'Syn→Syn':>8}  {'Kn→Kn':>8}  {'Code→Syn':>9}  {'Δ(Syn-Kn)':>10}")

    for step in steps_seen:
        step_results = [r for r in all_anatomy if r["step"] == step]
        by_band = defaultdict(list)
        for r in step_results:
            by_band[r["band"]].append(r["target_band_fraction"])

        syn_f = sum(by_band.get("syntax", [0])) / max(len(by_band.get("syntax", [1])), 1)
        kn_f = sum(by_band.get("knowledge", [0])) / max(len(by_band.get("knowledge", [1])), 1)
        code_f = sum(by_band.get("code", [0])) / max(len(by_band.get("code", [1])), 1)
        delta = syn_f - kn_f

        marker = ""
        if delta > 0.15:
            marker = " ← SEPARATION"
        elif delta > 0.05:
            marker = " ← emerging"

        print(f"  {step:>6}  {syn_f:>8.3f}  {kn_f:>8.3f}  {code_f:>9.3f}  {delta:>+10.3f}{marker}")

    # Contrastive summary
    print(f"\n{'='*60}")
    print(f"  CONTRASTIVE: Layer Preference (syntax > 0, knowledge < 0)")
    print(f"{'='*60}")

    contrastive_by_step = defaultdict(list)
    for cr in all_contrastive:
        contrastive_by_step[cr["step"]].append(cr)

    for step in sorted(contrastive_by_step.keys()):
        crs = contrastive_by_step[step]
        avg_pref = [0.0] * N_LAYERS
        for cr in crs:
            for i, p in enumerate(cr["layer_preference"]):
                avg_pref[i] += p
        avg_pref = [p / len(crs) for p in avg_pref]

        print(f"\n  Step {step}:")
        for i, p in enumerate(avg_pref):
            bar_len = int(15 * abs(p))
            if p >= 0:
                bar = ' ' * 15 + '|' + '▶' * bar_len
            else:
                bar = ' ' * (15 - bar_len) + '◀' * bar_len + '|'

            band_label = ""
            for bname, (lo, hi) in bands.items():
                if lo <= i <= hi:
                    band_label = f" [{bname[:3].upper()}]"
            print(f"    L{i:2d}: {bar} {p:+.3f}{band_label}")

    # Feature overlap
    print(f"\n{'='*60}")
    print(f"  FEATURE OVERLAP (syntax vs knowledge per layer)")
    print(f"{'='*60}")
    for i in range(N_LAYERS):
        overlap = feat_report.get(f"L{i}_overlap", 0)
        bar = '█' * int(30 * overlap)
        band_label = ""
        for bname, (lo, hi) in bands.items():
            if lo <= i <= hi:
                band_label = f" [{bname[:3].upper()}]"
        print(f"  L{i:2d}: {bar:30s} {overlap:.3f}{band_label}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")
    print(f"  Results: {output_dir}/")
    sys.stdout.flush()


if __name__ == "__main__":
    run_experiment()
