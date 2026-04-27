"""
Backpropagation-is-INSERT: Core Experiment

Trains a tiny Gemma-like transformer on mixed synthetic data (facts + syntax + code),
then performs gradient anatomy to measure whether gradients write to the correct
FFN layer bands:
  - Syntax band: L0-3   (morphology, synonyms, hypernyms)
  - Knowledge band: L4-7  (factual relations)
  - Output band: L8-11   (formatting, output shaping)

The key measurement: does a factual training example produce FFN weight displacement
concentrated in the knowledge band, and a syntactic example in the syntax band?
"""

import os
import json
import time
import math
import copy
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import TinyGemma
from synth_data import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Model
    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 12
    ffn_dim: int = 1024
    n_heads: int = 4
    n_kv_heads: int = 2
    max_seq: int = 128

    # Training
    lr: float = 3e-4
    batch_size: int = 8
    epochs: int = 30
    warmup_steps: int = 100

    # Checkpointing
    checkpoint_every: int = 50  # steps

    # Gradient anatomy
    anatomy_steps: List[int] = None  # which steps to run anatomy
    anatomy_samples_per_band: int = 10

    # Data
    n_countries: int = 50
    seed: int = 42

    # Band definitions (layer ranges, inclusive)
    syntax_band: Tuple[int, int] = (0, 3)
    knowledge_band: Tuple[int, int] = (4, 7)
    output_band: Tuple[int, int] = (8, 11)

    # Output
    output_dir: str = "results"

    def __post_init__(self):
        if self.anatomy_steps is None:
            self.anatomy_steps = [0, 50, 100, 200, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int = 128):
        self.encodings = []
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.encodings.append(ids)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


class SimpleTokenizer:
    """Character-level tokenizer with a few special tokens.
    Good enough for this experiment — we care about FFN structure, not tokenization."""

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        # Reserve special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        next_id = 4

        # Map printable ASCII + common chars
        for c in range(32, 127):
            self.char_to_id[chr(c)] = next_id
            self.id_to_char[next_id] = chr(c)
            next_id += 1

        # Add newline, tab
        for c in ['\n', '\t']:
            self.char_to_id[c] = next_id
            self.id_to_char[next_id] = c
            next_id += 1

        self.actual_vocab_size = next_id

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for c in text:
            ids.append(self.char_to_id.get(c, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids if i > 3)


def collate_fn(batch: List[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


# ---------------------------------------------------------------------------
# Gradient Anatomy
# ---------------------------------------------------------------------------

@dataclass
class GradientAnatomyResult:
    """Results from a single-gradient anatomy measurement."""
    step: int
    band: str              # ground truth band of the input
    relation: str          # ground truth relation
    text: str

    # Per-layer FFN gate displacement (L2 norm of gradient)
    gate_displacement: List[float]     # [n_layers]
    # Per-layer FFN down displacement
    down_displacement: List[float]     # [n_layers]
    # Per-layer FFN up displacement
    up_displacement: List[float]       # [n_layers]
    # Total displacement per layer (gate + down + up)
    total_displacement: List[float]    # [n_layers]

    # Band-level aggregates
    syntax_band_energy: float = 0.0
    knowledge_band_energy: float = 0.0
    output_band_energy: float = 0.0

    # Fraction in target band
    target_band_fraction: float = 0.0

    # Sparsity: fraction of layers with > 10% of max displacement
    layer_sparsity: float = 0.0


def compute_gradient_anatomy(
    model: TinyGemma,
    sample: GroundTruth,
    tokenizer: SimpleTokenizer,
    config: ExperimentConfig,
    step: int,
    device: torch.device,
) -> GradientAnatomyResult:
    """
    Compute single-gradient anatomy for one training example.
    Measures where in the FFN layers the gradient concentrates.
    """
    model.train()
    model.zero_grad()

    # Encode and forward
    ids = tokenizer.encode(sample.text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    logits = model(input_ids)

    # Next-token prediction loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, model.vocab_size),
        shift_labels.view(-1),
        ignore_index=0,
    )

    loss.backward()

    # Measure per-layer FFN gradient displacement
    gate_disp = []
    down_disp = []
    up_disp = []
    total_disp = []

    for layer in model.layers:
        g = layer.ffn.gate.weight.grad
        d = layer.ffn.down.weight.grad
        u = layer.ffn.up.weight.grad

        gn = g.norm().item() if g is not None else 0.0
        dn = d.norm().item() if d is not None else 0.0
        un = u.norm().item() if u is not None else 0.0

        gate_disp.append(gn)
        down_disp.append(dn)
        up_disp.append(un)
        total_disp.append(gn + dn + un)

    # Compute band energies
    def band_energy(disps, lo, hi):
        return sum(disps[lo:hi+1])

    syntax_e = band_energy(total_disp, *config.syntax_band)
    knowledge_e = band_energy(total_disp, *config.knowledge_band)
    output_e = band_energy(total_disp, *config.output_band)
    total_e = syntax_e + knowledge_e + output_e

    # Target band fraction
    if sample.band == "syntax" or sample.band == "code":
        target_e = syntax_e
    elif sample.band == "knowledge":
        target_e = knowledge_e
    else:
        target_e = output_e

    target_frac = target_e / total_e if total_e > 0 else 0.0

    # Sparsity: how many layers have > 10% of max
    max_disp = max(total_disp) if total_disp else 1.0
    active_layers = sum(1 for d in total_disp if d > 0.1 * max_disp)
    sparsity = active_layers / len(total_disp)

    model.zero_grad()

    return GradientAnatomyResult(
        step=step,
        band=sample.band,
        relation=sample.relation,
        text=sample.text[:100],
        gate_displacement=gate_disp,
        down_displacement=down_disp,
        up_displacement=up_disp,
        total_displacement=total_disp,
        syntax_band_energy=syntax_e / total_e if total_e > 0 else 0,
        knowledge_band_energy=knowledge_e / total_e if total_e > 0 else 0,
        output_band_energy=output_e / total_e if total_e > 0 else 0,
        target_band_fraction=target_frac,
        layer_sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# Graph Reconstruction (Gate Vector KNN)
# ---------------------------------------------------------------------------

@dataclass
class GraphReconstructionResult:
    step: int
    loss: float

    # Per-band metrics
    syntax_cluster_purity: float = 0.0
    knowledge_cluster_purity: float = 0.0

    # Gate vector statistics
    gate_norm_per_layer: List[float] = None
    gate_cosine_similarity_within_band: float = 0.0

    # Overall
    n_active_features_per_layer: List[int] = None


def extract_gate_stats(model: TinyGemma) -> GraphReconstructionResult:
    """Extract gate vector statistics from the model at current state."""
    gate_norms = []
    active_features = []

    for layer in model.layers:
        W = layer.ffn.gate.weight.data  # (ffn_dim, dim)
        norms = W.norm(dim=1)  # per-feature norm
        gate_norms.append(norms.mean().item())
        # Count features with above-median norm as "active"
        median_norm = norms.median()
        active = (norms > median_norm * 1.5).sum().item()
        active_features.append(active)

    return GraphReconstructionResult(
        step=0, loss=0.0,
        gate_norm_per_layer=gate_norms,
        n_active_features_per_layer=active_features,
    )


def measure_band_specialisation(
    model: TinyGemma,
    samples: List[GroundTruth],
    tokenizer: SimpleTokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Measure how specialised each FFN layer is for different input types.
    Feed syntax vs knowledge samples and measure activation differences per layer.
    """
    model.eval()

    layer_activations = {
        "syntax": [[] for _ in range(config.n_layers)],
        "knowledge": [[] for _ in range(config.n_layers)],
        "code": [[] for _ in range(config.n_layers)],
    }

    # Register hooks to capture FFN gate activations
    hooks = []
    gate_outputs = [None] * config.n_layers

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            gate_outputs[layer_idx] = output.detach()
        return hook_fn

    for i, layer in enumerate(model.layers):
        h = layer.ffn.gate.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for sample in samples:
            ids = tokenizer.encode(sample.text)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _ = model(input_ids)

            band = sample.band
            for i in range(config.n_layers):
                if gate_outputs[i] is not None:
                    # Mean activation magnitude across sequence
                    act = F.silu(gate_outputs[i]).abs().mean().item()
                    layer_activations[band][i].append(act)

    for h in hooks:
        h.remove()

    # Compute per-layer specialisation: ratio of knowledge vs syntax activation
    results = {}
    for layer_idx in range(config.n_layers):
        syn_acts = layer_activations["syntax"][layer_idx]
        kn_acts = layer_activations["knowledge"][layer_idx]

        syn_mean = sum(syn_acts) / len(syn_acts) if syn_acts else 0
        kn_mean = sum(kn_acts) / len(kn_acts) if kn_acts else 0

        # Specialisation ratio: how much more one band activates than the other
        total = syn_mean + kn_mean
        if total > 0:
            results[f"L{layer_idx}_syntax_ratio"] = syn_mean / total
            results[f"L{layer_idx}_knowledge_ratio"] = kn_mean / total
        else:
            results[f"L{layer_idx}_syntax_ratio"] = 0.5
            results[f"L{layer_idx}_knowledge_ratio"] = 0.5

    return results


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(config: ExperimentConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Build data
    print("\n=== Building synthetic corpus ===")
    samples, ground_truth = build_mixed_corpus(n_countries=config.n_countries, seed=config.seed)
    print(f"Total samples: {ground_truth['counts']}")

    # Save ground truth
    with open(os.path.join(config.output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    actual_vocab = tokenizer.actual_vocab_size
    print(f"Actual vocab size: {actual_vocab}")

    # Model
    model = TinyGemma(
        vocab_size=actual_vocab,
        dim=config.dim,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq=config.max_seq,
    ).to(device)
    print(f"Model params: {model.param_count():,}")

    # Dataset
    texts = [s.text for s in samples]
    dataset = TextDataset(texts, tokenizer, max_len=config.max_seq)

    def my_collate(batch):
        return collate_fn(batch, pad_id=tokenizer.pad_id)

    loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=my_collate,
        drop_last=True,
    )

    # Optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    # Separate samples by band for anatomy
    band_samples = defaultdict(list)
    for s in samples:
        band_samples[s.band].append(s)

    # Select anatomy samples (fixed across all steps for consistency)
    rng = random.Random(config.seed)
    anatomy_samples = {}
    for band in ["syntax", "knowledge", "code"]:
        pool = band_samples[band]
        n = min(config.anatomy_samples_per_band, len(pool))
        anatomy_samples[band] = rng.sample(pool, n)

    # --------------- Training ---------------
    all_anatomy_results = []
    all_reconstruction_results = []
    all_specialisation_results = []
    loss_history = []
    global_step = 0

    print("\n=== Training ===")
    t0 = time.time()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            # --- Gradient anatomy at scheduled steps ---
            if global_step in config.anatomy_steps:
                print(f"\n  [Step {global_step}] Running gradient anatomy...")
                for band, band_samps in anatomy_samples.items():
                    for samp in band_samps:
                        result = compute_gradient_anatomy(
                            model, samp, tokenizer, config, global_step, device,
                        )
                        all_anatomy_results.append(asdict(result))

                # Also measure band specialisation
                spec_samples = []
                for band in ["syntax", "knowledge"]:
                    spec_samples.extend(band_samples[band][:30])
                spec = measure_band_specialisation(model, spec_samples, tokenizer, config, device)
                spec["step"] = global_step
                all_specialisation_results.append(spec)

                # Gate stats
                gate_stats = extract_gate_stats(model)
                gate_stats.step = global_step
                all_reconstruction_results.append(asdict(gate_stats))

            # --- Normal training step ---
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            loss_history.append({"step": global_step, "loss": loss_val})

            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{config.epochs}  loss={avg_loss:.4f}  "
              f"step={global_step}  time={elapsed:.1f}s")

    # --- Final anatomy ---
    print(f"\n  [Step {global_step}] Running FINAL gradient anatomy...")
    for band, band_samps in anatomy_samples.items():
        for samp in band_samps:
            result = compute_gradient_anatomy(
                model, samp, tokenizer, config, global_step, device,
            )
            all_anatomy_results.append(asdict(result))

    spec_samples = []
    for band in ["syntax", "knowledge"]:
        spec_samples.extend(band_samples[band][:30])
    spec = measure_band_specialisation(model, spec_samples, tokenizer, config, device)
    spec["step"] = global_step
    all_specialisation_results.append(spec)

    # --------------- Save Results ---------------
    print("\n=== Saving results ===")

    with open(os.path.join(config.output_dir, "anatomy_results.json"), "w") as f:
        json.dump(all_anatomy_results, f, indent=2)

    with open(os.path.join(config.output_dir, "reconstruction_results.json"), "w") as f:
        json.dump(all_reconstruction_results, f, indent=2)

    with open(os.path.join(config.output_dir, "specialisation_results.json"), "w") as f:
        json.dump(all_specialisation_results, f, indent=2)

    with open(os.path.join(config.output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)

    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # --------------- Print Summary ---------------
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Aggregate anatomy by band and step
    print("\n--- Gradient Band Targeting (target_band_fraction) ---")
    print(f"{'Step':>6}  {'Syntax→SynBand':>15}  {'Knowledge→KnBand':>17}  {'Code→SynBand':>13}")

    anatomy_by_step = defaultdict(lambda: defaultdict(list))
    for r in all_anatomy_results:
        anatomy_by_step[r["step"]][r["band"]].append(r["target_band_fraction"])

    for step in sorted(anatomy_by_step.keys()):
        syn = anatomy_by_step[step].get("syntax", [])
        kn = anatomy_by_step[step].get("knowledge", [])
        code = anatomy_by_step[step].get("code", [])

        syn_mean = sum(syn) / len(syn) if syn else 0
        kn_mean = sum(kn) / len(kn) if kn else 0
        code_mean = sum(code) / len(code) if code else 0

        print(f"{step:>6}  {syn_mean:>15.3f}  {kn_mean:>17.3f}  {code_mean:>13.3f}")

    # Layer displacement heatmap (text)
    print("\n--- Average FFN Displacement Per Layer (final step) ---")
    final_step = max(anatomy_by_step.keys())
    for band in ["syntax", "knowledge", "code"]:
        results_at_final = [r for r in all_anatomy_results
                           if r["step"] == final_step and r["band"] == band]
        if not results_at_final:
            continue

        avg_disp = [0.0] * config.n_layers
        for r in results_at_final:
            for i, d in enumerate(r["total_displacement"]):
                avg_disp[i] += d
        avg_disp = [d / len(results_at_final) for d in avg_disp]
        max_d = max(avg_disp) if avg_disp else 1.0

        # Normalise to bars
        bars = ""
        for i, d in enumerate(avg_disp):
            bar_len = int(20 * d / max_d)
            band_label = ""
            if config.syntax_band[0] <= i <= config.syntax_band[1]:
                band_label = " [SYN]"
            elif config.knowledge_band[0] <= i <= config.knowledge_band[1]:
                band_label = " [KN]"
            else:
                band_label = " [OUT]"
            bars += f"  L{i:2d}: {'█' * bar_len}{'░' * (20 - bar_len)} {d:.4f}{band_label}\n"

        print(f"\n  [{band.upper()}] samples:")
        print(bars)

    # Specialisation
    print("--- Band Specialisation (knowledge ratio per layer) ---")
    if all_specialisation_results:
        final_spec = all_specialisation_results[-1]
        for i in range(config.n_layers):
            kn_ratio = final_spec.get(f"L{i}_knowledge_ratio", 0.5)
            bar = int(40 * kn_ratio)
            band_label = ""
            if config.syntax_band[0] <= i <= config.syntax_band[1]:
                band_label = " [SYN]"
            elif config.knowledge_band[0] <= i <= config.knowledge_band[1]:
                band_label = " [KN]"
            else:
                band_label = " [OUT]"
            print(f"  L{i:2d}: {'◀' * (20 - bar)}|{'▶' * (20 - (40 - bar))} "
                  f"syn={1-kn_ratio:.2f} kn={kn_ratio:.2f}{band_label}")

    # Cross-band leakage
    print("\n--- Cross-Band Leakage ---")
    for band in ["syntax", "knowledge"]:
        results_at_final = [r for r in all_anatomy_results
                           if r["step"] == final_step and r["band"] == band]
        if not results_at_final:
            continue
        mean_target = sum(r["target_band_fraction"] for r in results_at_final) / len(results_at_final)
        leakage = 1.0 - mean_target
        status = "PASS" if leakage < 0.5 else "PARTIAL" if leakage < 0.7 else "FAIL"
        print(f"  {band}: target_band={mean_target:.1%}, leakage={leakage:.1%} [{status}]")

    print(f"\nFinal loss: {loss_history[-1]['loss']:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Results saved to: {config.output_dir}/")

    return all_anatomy_results, all_specialisation_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ExperimentConfig(
        epochs=30,
        batch_size=8,
        lr=3e-4,
        n_countries=50,
        checkpoint_every=50,
        anatomy_steps=[0, 50, 100, 200, 500, 1000, 2000],
        anatomy_samples_per_band=10,
        output_dir="results",
    )

    print("Backpropagation-is-INSERT Experiment")
    print("=" * 50)
    print(f"Config: {json.dumps(asdict(config), indent=2)}")

    train(config)
