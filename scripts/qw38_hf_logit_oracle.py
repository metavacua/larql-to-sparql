#!/usr/bin/env python3
"""HF reference logits for one Qwen3.8-27B decode position.

The oracle for the first LARQL-vs-HF parity gate. Deliberately ONE
position: a token trajectory can agree while the logits differ, so the
full logit vector is compared before any sentence is generated.

Precision is a decision, not a default. `--dtype float32` is the gate's
subject (LARQL's reference backend is f32); `--dtype bfloat16` is the
control that says how much of any gap is precision alone — the same split
QW-2 used when a 4.157e-3 miss turned out to be entirely an L2-norm dtype
choice.

Usage:
  python3 scripts/qw38_hf_logit_oracle.py <ckpt> --prompt "..." \
      [--dtype float32|bfloat16] --out logits.json
"""
import argparse, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    ids = tok(a.prompt, return_tensors="pt").input_ids
    print(f"prompt={a.prompt!r} tokens={ids.shape[1]} ids={ids[0].tolist()}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, dtype=dtype, low_cpu_mem_usage=True, device_map="cpu"
    ).eval()
    with torch.no_grad():
        out = model(ids)
    # Last position's logits — the distribution the next token is drawn from.
    logits = out.logits[0, -1].float()
    top = torch.topk(logits, a.topk)

    json.dump({
        "checkpoint": a.checkpoint,
        "dtype": a.dtype,
        "prompt": a.prompt,
        "input_ids": ids[0].tolist(),
        "argmax_id": int(logits.argmax()),
        "argmax_token": tok.decode([int(logits.argmax())]),
        "topk": [
            {"id": int(i), "logit": float(v), "token": tok.decode([int(i)])}
            for v, i in zip(top.values, top.indices)
        ],
        "logits": logits.tolist(),
    }, open(a.out, "w"))
    print(f"argmax={int(logits.argmax())} {tok.decode([int(logits.argmax())])!r}", file=sys.stderr)
    print(f"wrote {a.out} ({logits.numel()} logits)", file=sys.stderr)

main()
