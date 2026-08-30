#!/usr/bin/env python3
"""HF greedy token trajectory — the QW-3.7 acceptance oracle.

Greedy so the comparison is deterministic on both sides: any sampling
would make a trajectory mismatch uninterpretable. Records the argmax id
AND its logit at every step, so a divergence can be read as "different
token" or "same token, drifting margin".

Usage:
  python3 scripts/qw38_hf_greedy_oracle.py <ckpt> --tokens 760,... --new 12 --out traj.json
"""
import argparse, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--new", type=int, default=12)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = [int(t) for t in a.tokens.split(",")]
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True, device_map="cpu"
    ).eval()

    steps = []
    cur = list(ids)
    with torch.no_grad():
        for i in range(a.new):
            out = model(torch.tensor([cur]))
            logits = out.logits[0, -1].float()
            nxt = int(logits.argmax())
            steps.append({
                "step": i,
                "id": nxt,
                "logit": float(logits[nxt]),
                "runner_up": float(torch.topk(logits, 2).values[1]),
                "token": tok.decode([nxt]),
            })
            cur.append(nxt)
            print(f"  {i}: {nxt} {tok.decode([nxt])!r}", file=sys.stderr)

    json.dump({
        "checkpoint": a.checkpoint, "dtype": a.dtype, "prompt_ids": ids,
        "steps": steps, "text": tok.decode(cur),
    }, open(a.out, "w"), indent=1)
    print(f"\n  {tok.decode(cur)!r}", file=sys.stderr)

main()
