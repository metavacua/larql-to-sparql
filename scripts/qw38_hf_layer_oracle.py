#!/usr/bin/env python3
"""HF per-layer residual planes for the QW-3.6c divergence census.

The DEBUGGING instrument, not the acceptance gate: it captures the
residual stream leaving every layer so the FIRST diverging layer can be
identified, rather than hypothesising backwards from a wrong final token.

Qwen3.8's cadence is LLLF, so the answer is diagnostic by position:

    embedding differs        -> embedding / token pipeline
    L0 differs               -> recurrent layer integration
    L0..L2 agree, L3 differs -> softmax integration (M-RoPE, gate, KV)
    many agree, one differs  -> position- or layer-specific semantics

Only the LAST position is kept per plane — 5120 floats x 65 planes is
tiny beside the model, and it is the position the logits come from.

Usage:
  python3 scripts/qw38_hf_layer_oracle.py <ckpt> --tokens 760,6511,... --out planes.json
"""
import argparse, json, sys
import torch
from transformers import AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", required=True, help="comma-separated ids")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = torch.tensor([[int(t) for t in a.tokens.split(",")]], dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True, device_map="cpu"
    ).eval()

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # `hidden_states[0]` is the embedding output; [i+1] leaves layer i.
    planes = [h[0, -1].float().tolist() for h in out.hidden_states]
    logits = out.logits[0, -1].float()
    json.dump({
        "checkpoint": a.checkpoint,
        "dtype": a.dtype,
        "input_ids": ids[0].tolist(),
        "position": ids.shape[1] - 1,
        "hidden": len(planes[0]),
        "planes": planes,            # [num_layers+1][hidden], last position
        "argmax_id": int(logits.argmax()),
        "logits": logits.tolist(),
    }, open(a.out, "w"))
    print(f"{len(planes)} planes (embedding + {len(planes)-1} layers), "
          f"hidden {len(planes[0])}, argmax {int(logits.argmax())}", file=sys.stderr)

main()
