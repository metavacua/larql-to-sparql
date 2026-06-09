#!/usr/bin/env python3
"""Generate an HF-transformers reference dump for the DSv4 parity test.

This is the *external authority* for `dsv4-hf-parity` (tasks #63/#69):
it runs the reference DeepSeek-V4-Flash under HuggingFace `transformers`
on a fixed prompt and writes the final-position next-token top-K to a
small JSON dump. The Rust test
(`crates/larql-inference/tests/test_dsv4_hf_parity.rs`) consumes that
dump and pins the GGUF forward to it — see that file + the
`openspec/changes/dsv4-hf-parity/` change for the contract.

Run out-of-band on a host with the HF weights + a transformers env:

    python scripts/dsv4_hf_reference.py \
        --model deepseek-ai/DeepSeek-V4-Flash \
        --prompt "The capital of France is" \
        --top-k 10 \
        --out crates/larql-inference/tests/goldens/dsv4_flash_hf_reference.json

The dump carries the HF-tokenized `token_ids` so the Rust side bypasses
any tokenizer mismatch. The script reports the top-1 -> top-2 logit gap
so you can pick a prompt with a confident (argmax-stable) completion.
"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="reference compute dtype (the GGUF is Q4_K; this is the "
        "full-precision authority we compare against)",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    print(f"loading tokenizer + model: {args.model} ({args.dtype})", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    token_ids = input_ids[0].tolist()
    print(f"prompt {args.prompt!r} -> {len(token_ids)} tokens: {token_ids}", file=sys.stderr)

    with torch.no_grad():
        out = model(input_ids.to(model.device))
    # Final-position next-token logits: [vocab].
    logits = out.logits[0, -1, :].float().cpu()

    vals, idx = torch.topk(logits, k=args.top_k)
    top_k = [{"token_id": int(t), "logit": float(v)} for t, v in zip(idx.tolist(), vals.tolist())]

    gap = top_k[0]["logit"] - top_k[1]["logit"] if len(top_k) > 1 else float("inf")
    print(
        f"top-1 token {top_k[0]['token_id']} ({tok.decode([top_k[0]['token_id']])!r}) "
        f"logit {top_k[0]['logit']:.3f}; top1->top2 gap {gap:.3f}",
        file=sys.stderr,
    )
    if gap < 1.0:
        print(
            "WARNING: small top1->top2 gap — argmax may not be Q4_K-stable; "
            "consider a more confident prompt.",
            file=sys.stderr,
        )

    dump = {
        "model": args.model,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "token_ids": token_ids,
        "top_k": top_k,
    }
    with open(args.out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
