#!/usr/bin/env python3
"""Generate the QW-3.5B M-RoPE parity fixture from HF's own qwen3_5 rope.

The oracle needs the config only — no 27B weight load — so this fixture is
hermetic and committed rather than env-gated.

`q` is NOT stored: it is regenerated in Rust by `fixtures::lcg_values`, and
`the_generator_still_produces_the_values_this_fixture_was_built_from` in
the test pins the two implementations together. Only HF's output is data.

Usage:
  python3 scripts/gen_qw35b_mrope_fixture.py <checkpoint-dir> > <fixture.json>
"""
import json, sys, torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb,
)
import transformers

MASK = (1 << 64) - 1
MUL, INC = 6364136223846793005, 1442695040888963407

def lcg_values(n, seed):
    """Bit-exact port of larql-vindex `format::vindex3::fixtures::lcg_values`."""
    state = (seed * MUL + INC) & MASK
    out = []
    for _ in range(n):
        state = (state * MUL + INC) & MASK
        unit = (state >> 33) / float(1 << 31)
        out.append(float(torch.tensor((unit - 0.5) * 0.1, dtype=torch.float32)))
    return out

POSITIONS = [0, 1, 2, 5, 17, 100]
SEED = 0x5B35
N_HEADS = 2

def main(ckpt):
    raw = json.load(open(f"{ckpt}/config.json"))["text_config"]
    cfg = Qwen3_5TextConfig(**raw)
    hd = cfg.head_dim
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    n_pos = len(POSITIONS)

    flat = lcg_values(N_HEADS * n_pos * hd, SEED)
    q = torch.tensor(flat, dtype=torch.float32).reshape(1, N_HEADS, n_pos, hd)
    pos = torch.tensor(POSITIONS, dtype=torch.long)[None, :]

    x = torch.zeros(1, n_pos, hd)
    cos, sin = rot(x, pos)
    q_rot, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin, unsqueeze_dim=1)

    print(json.dumps({
        "provenance": {
            "generator": "scripts/gen_qw35b_mrope_fixture.py",
            "transformers": transformers.__version__,
            "checkpoint": "Qwen/Qwen3.8-27B",
            "oracle": "Qwen3_5TextRotaryEmbedding + apply_rotary_pos_emb",
        },
        "config": {
            "head_dim": hd,
            "rope_theta": cfg.rope_parameters["rope_theta"],
            "partial_rotary_factor": cfg.rope_parameters["partial_rotary_factor"],
            "mrope_section": rot.mrope_section,
            "mrope_interleaved": cfg.rope_parameters["mrope_interleaved"],
            "rotary_dim": cos.shape[-1],
            "n_freqs": rot.inv_freq.shape[0],
        },
        "lcg": {"seed": SEED, "count": N_HEADS * n_pos * hd, "num_heads": N_HEADS},
        "positions": POSITIONS,
        "q_rotated": q_rot.reshape(N_HEADS * n_pos, hd).tolist(),
    }, indent=1))

main(sys.argv[1])
