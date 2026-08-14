#!/usr/bin/env python3
"""Singular spectrum of a MoE expert bank, taken over the **expert axis**.

Asks how much of an expert bank is shared structure, in the one orientation
that can actually answer it, and pairs it with the independent question of
whether each expert is individually low-rank.

## Why the orientation is the whole measurement

The tempting version — stack `E` experts of shape `[d_out, d_in]` into a
`[E·d_out, d_in]` matrix and compare its effective rank to `E ×` the individual
effective rank — cannot return "independent". That matrix's rank is capped at
`d_in` by the ambient dimension, so with a per-expert effective rank of order
`d_in/2` the observable ratio is boxed into roughly `[1, 2]` no matter what the
weights contain. It reports "massively redundant" by construction. That is the
same failure as reading a near-zero cosine between two flattened weight
matrices as evidence of orthogonality: a statistic dominated by the ambient
dimension rather than by the property under test.

Reshaping instead to `[E, d_out·d_in]` — one row per flattened expert — caps
the rank at `E`, which *is* the quantity in question. A flat spectrum means `E`
genuinely independent experts; a concentrated one means the bank is a
low-dimensional family.

## The null is flat, and it is measured rather than assumed

For `E` independent random experts the Gram matrix is `≈ D·σ²·I` with
`D = d_out·d_in >> E`, so every singular value is equal and the cumulative
energy at rank `M` is exactly `M/E`. Deviations are `O(sqrt(E/D))` — under 1%
at these shapes. `--random-control` measures it anyway, because a null that is
only derived is a null that can be derived wrongly.

## What rank M means if you ship it

A rank-`M` truncation writes each expert as a linear combination of `M` shared
atom matrices: `W_i ≈ Σ_j c_ij A_j`. Storage is `M·D + E·M` against `E·D`, and
since `E << D` the coefficients are free — the compression ratio is `E/M`. So
"5× compression" is exactly "rank 26 of 128", and "how many distinct expert
functions are there" and "how far does this compress" are one question in two
sets of units.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Reported statistics ────────────────────────────────────────────────────

# Energy thresholds reported for every (layer, projection) pair.
ENERGY_LEVELS = (0.90, 0.95, 0.99, 0.999)

# Compression ratios whose retained energy is reported directly, because these
# are the budgets a serving decision is actually made at.
COMPRESSION_BUDGETS = (2, 5, 10, 20)

# ── Checkpoint layout (HF convention, shared by Qwen3-MoE, OLMoE, Mixtral) ──

SAFETENSORS_INDEX = "model.safetensors.index.json"
CONFIG_FILE = "config.json"
WEIGHT_MAP_FIELD = "weight_map"
NUM_LAYERS_FIELD = "num_hidden_layers"
EXPERT_PREFIX = "model.layers.{layer}.mlp.experts."
EXPERT_SUFFIX = ".{projection}.weight"

# Torch, not numpy: these checkpoints are bf16 and numpy has no bf16 dtype.
SAFETENSORS_FRAMEWORK = "pt"

# ── Defaults ───────────────────────────────────────────────────────────────

# Layers are sampled evenly from the model's actual depth rather than named,
# so the default is not silently wrong on a model of a different height.
DEFAULT_LAYER_SAMPLES = 5
DEFAULT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Per-expert SVD is the expensive axis; a sample is enough to place the mean.
DEFAULT_EXPERT_SAMPLE = 8
RANDOM_CONTROL_SEED = 20260731

# ── Report layout ──────────────────────────────────────────────────────────

VARIANTS = ("raw", "centred")
LABEL_COLUMN = 10
RANK_COLUMN = 8
ENERGY_COLUMN = 11
ENERGY_DECIMALS = 4
MISSING = "—"


def sample_layers(snapshot, count):
    """`count` layer indices spread evenly over the model's depth."""
    config = json.loads((snapshot / CONFIG_FILE).read_text())
    depth = config[NUM_LAYERS_FIELD]
    return sorted({int(round(i)) for i in np.linspace(0, depth - 1, min(count, depth))})


def load_expert_bank(snapshot, layer, projection):
    """`([E, d_out*d_in] float32, (d_out, d_in))` — one flattened expert per row."""
    import torch
    from safetensors import safe_open

    index = json.loads((snapshot / SAFETENSORS_INDEX).read_text())
    prefix = EXPERT_PREFIX.format(layer=layer)
    suffix = EXPERT_SUFFIX.format(projection=projection)
    keys = [k for k in index[WEIGHT_MAP_FIELD] if k.startswith(prefix) and k.endswith(suffix)]
    if not keys:
        raise SystemExit(f"no tensors matching {prefix}*{suffix}")
    keys.sort(key=lambda k: int(k[len(prefix) : -len(suffix)]))

    # Group by shard so each file is opened once, not once per expert.
    by_shard = {}
    for key in keys:
        by_shard.setdefault(index[WEIGHT_MAP_FIELD][key], []).append(key)

    rows, shape = {}, None
    for shard, shard_keys in by_shard.items():
        with safe_open(snapshot / shard, framework=SAFETENSORS_FRAMEWORK) as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key).to(torch.float32).numpy()
                shape = tensor.shape
                rows[key] = tensor.ravel()
    return np.stack([rows[k] for k in keys]), shape


def spectrum(bank):
    """Descending eigenvalues of `bank @ bank.T` — the squared singular values.

    `E` is ~128 and `D` ~1.5M, so forming the Gram matrix and eigendecomposing
    it is orders cheaper than an SVD of the bank and numerically equivalent for
    the energy fractions reported here. float64 because squaring doubles the
    dynamic range, and the small eigenvalues decide where the spectrum flattens.
    """
    wide = bank.astype(np.float64)
    return np.clip(np.linalg.eigvalsh(wide @ wide.T)[::-1], 0.0, None)


def energy_curve(eigenvalues):
    """Cumulative energy fraction at each rank; index 0 is rank 1."""
    total = eigenvalues.sum()
    if total <= 0:
        return np.zeros_like(eigenvalues)
    return np.cumsum(eigenvalues) / total


def rank_for_energy(curve, level):
    """Smallest rank retaining `level` of the total energy."""
    return int(np.searchsorted(curve, level) + 1)


def rank_at_compression(experts, ratio):
    """Retained rank for a target cross-expert compression `ratio`."""
    return min(max(experts // ratio, 1), experts)


def per_expert_ranks(bank, shape, sample=DEFAULT_EXPERT_SAMPLE):
    """Within-expert low-rank structure — the *other* compression route.

    Cross-expert redundancy and per-expert low rank are independent questions
    that fail independently: a bank of `E` mutually unrelated experts still
    compresses if each expert is individually low-rank, and a bank of
    near-identical experts compresses even if each is full rank. Measuring only
    the first and reporting "incompressible" would overclaim.

    Storage for a rank-`r` factorisation of one `[d_out, d_in]` expert is
    `r·(d_out + d_in)` against `d_out·d_in`, so it only pays below a **breakeven
    rank** of `d_out·d_in / (d_out + d_in)`. Above that the factorisation is
    larger than the matrix it replaces. That is a worse exchange rate than the
    shared-atom form, which pays nothing per expert.
    """
    d_out, d_in = shape
    step = max(len(bank) // sample, 1)
    curves = [
        energy_curve(np.linalg.svd(row.reshape(shape).astype(np.float64), compute_uv=False) ** 2)
        for row in bank[::step][:sample]
    ]
    return {
        "sampled_experts": len(curves),
        "full_rank": int(min(shape)),
        "breakeven_rank": d_out * d_in / (d_out + d_in),
        "rank_at_energy": {
            f"{level:.3f}": float(np.mean([rank_for_energy(c, level) for c in curves]))
            for level in ENERGY_LEVELS
        },
    }


def summarise(bank, label, expert_shape=None):
    """Spectrum summary for one bank, raw and with the shared mean removed."""
    experts = bank.shape[0]
    out = {"label": label, "experts": experts, "dim": int(bank.shape[1])}
    if expert_shape is not None:
        out["per_expert"] = per_expert_ranks(bank, expert_shape)

    for variant, matrix in (("raw", bank), ("centred", bank - bank.mean(axis=0))):
        curve = energy_curve(spectrum(matrix))
        out[variant] = {
            "rank_at_energy": {
                f"{level:.3f}": rank_for_energy(curve, level) for level in ENERGY_LEVELS
            },
            "energy_at_compression": {
                str(ratio): float(curve[rank_at_compression(experts, ratio) - 1])
                for ratio in COMPRESSION_BUDGETS
            },
        }
    return out


def random_control(shape, seed=RANDOM_CONTROL_SEED):
    """Matched-shape Gaussian bank — the measured version of the flat null."""
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


def format_report(entry):
    experts = entry["experts"]
    lines = [f"\n── {entry['label']} — {experts} experts, D={entry['dim']:,} ──"]

    header = "  ".join(f"r@{level:.1%}".rjust(RANK_COLUMN) for level in ENERGY_LEVELS)
    lines.append(f"{'variant':>{LABEL_COLUMN}}  {header}")
    for variant in VARIANTS:
        ranks = entry[variant]["rank_at_energy"]
        cells = "  ".join(
            f"{ranks[f'{level:.3f}']:>{RANK_COLUMN}}" for level in ENERGY_LEVELS
        )
        lines.append(f"{variant:>{LABEL_COLUMN}}  {cells}")

    budgets = "  ".join(
        f"{ratio}x (r={rank_at_compression(experts, ratio)})".rjust(ENERGY_COLUMN)
        for ratio in COMPRESSION_BUDGETS
    )
    lines.append(f"\n{'':>{LABEL_COLUMN}}  {budgets}")
    for variant in VARIANTS:
        energies = entry[variant]["energy_at_compression"]
        cells = "  ".join(
            f"{energies[str(ratio)]:>{ENERGY_COLUMN}.{ENERGY_DECIMALS}f}"
            for ratio in COMPRESSION_BUDGETS
        )
        lines.append(f"{variant:>{LABEL_COLUMN}}  {cells}")
    flat = "  ".join(
        f"{rank_at_compression(experts, ratio) / experts:>{ENERGY_COLUMN}.{ENERGY_DECIMALS}f}"
        for ratio in COMPRESSION_BUDGETS
    )
    lines.append(f"{'flat null':>{LABEL_COLUMN}}  {flat}")

    if (within := entry.get("per_expert")) is not None:
        cells = "  ".join(
            f"{within['rank_at_energy'][f'{level:.3f}']:>{RANK_COLUMN}.1f}"
            for level in ENERGY_LEVELS
        )
        lines.append(
            f"\nper-expert rank (mean of {within['sampled_experts']}, "
            f"full {within['full_rank']}, breakeven {within['breakeven_rank']:.0f}):"
        )
        lines.append(f"{'':>{LABEL_COLUMN}}  {cells}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot", type=Path, help="model snapshot directory")
    parser.add_argument(
        "--layers",
        help=f"comma-separated layer indices; default is {DEFAULT_LAYER_SAMPLES} "
        "spread evenly over the model's depth",
    )
    parser.add_argument("--projections", default=",".join(DEFAULT_PROJECTIONS))
    parser.add_argument("--expert-sample", type=int, default=DEFAULT_EXPERT_SAMPLE)
    parser.add_argument("--json", type=Path, help="write full results here")
    parser.add_argument(
        "--random-control",
        action="store_true",
        help="also measure a matched-shape Gaussian bank, so the flat null is "
        "observed rather than assumed",
    )
    args = parser.parse_args()

    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else sample_layers(args.snapshot, DEFAULT_LAYER_SAMPLES)
    )
    projections = [p.strip() for p in args.projections.split(",") if p.strip()]

    results = []
    for layer in layers:
        for projection in projections:
            bank, shape = load_expert_bank(args.snapshot, layer, projection)
            label = f"layer {layer} / {projection}"
            for name, matrix in [(label, bank)] + (
                [(f"{label} [RANDOM]", random_control(bank.shape))]
                if args.random_control
                else []
            ):
                results.append(summarise(matrix, name, shape))
                print(format_report(results[-1]), flush=True)
            del bank

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
