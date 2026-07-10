#!/usr/bin/env python3
"""Emit the full leg list for the LQL strategy matrix as a JSON array (one object
per parallel CI job). A "leg" is a (source × produce-recipe × level) tuple: the
recipe produces a vindex, then the command corpus runs against it.

Design rule (see the matrix README + discussion): include EVERY CLI-invokable
combination — do NOT prune by predicted meaninglessness. E.g. `--quant q4k` is
crossed with every level, not just `all`, precisely to test the doc claim that it
"implies --level all". A leg is omitted only when it is structurally un-formable
(e.g. a convert recipe with no GGUF source). Predictions live as `expect_quant`
oracles, not as omissions.

Each leg: name (artifact-safe), hf (repo for snapshot_download + corpus_model),
source_kind, op, level, flags, expect_quant.
"""
import json

# (short-id, HF repo) — all MIT/Apache, ungated.
SAFETENSORS = [
    ("qwen05",    "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    ("smol135",   "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smol360",   "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("qwen15",    "Qwen/Qwen2.5-1.5B-Instruct"),
    ("granite1b", "ibm-granite/granite-3.0-1b-a400m-instruct"),
    ("bitnet2b",  "microsoft/bitnet-b1.58-2B-4T"),
]
LEVELS = ["browse", "attention", "inference", "all"]
# quant-state axis: (id, extract-flags, expected quant in produced index.json)
QUANTS = [("f16", "", "none"), ("f32", "--f32", "none"), ("q4k", "--quant q4k", "q4k")]

# GGUF sources for the convert path: (short-id, gguf repo, corpus model id)
GGUF = [("bitnetgguf", "microsoft/bitnet-b1.58-2B-4T-gguf", "microsoft/bitnet-b1.58-2B-4T")]
GGUF_LEVELS = ["browse", "inference", "all"]  # gguf-to-vindex's supported levels
# dequantize (default) vs retain native ternary
GGUF_QUANT = [("dequant", "", "none"), ("keepq", "--keep-quant", "ternary")]


def main():
    legs = []

    # extract path: full quant × level grid per safetensors source
    for mid, hf in SAFETENSORS:
        for qid, qflags, eq in QUANTS:
            for lv in LEVELS:
                legs.append({
                    "name": f"{mid}.{qid}.{lv}", "hf": hf, "corpus_model": hf,
                    "source_kind": "safetensors", "op": "extract",
                    "level": lv, "flags": qflags, "expect_quant": eq,
                })

    # convert path: gguf-to-vindex, dequant vs keep-quant, across its levels
    for mid, ghf, cm in GGUF:
        for lv in GGUF_LEVELS:
            for kid, kflags, eq in GGUF_QUANT:
                legs.append({
                    "name": f"{mid}.{kid}.{lv}", "hf": ghf, "corpus_model": cm,
                    "source_kind": "gguf", "op": "gguf-to-vindex",
                    "level": lv, "flags": kflags, "expect_quant": eq,
                })

    # post-hoc requantize on the baseline (invariant vs inline extract --quant q4k;
    # + the FP4 store path). Source needs --level inference/all; job extracts first.
    for sub, eq in [("q4k", "q4k"), ("fp4", "fp4")]:
        legs.append({
            "name": f"qwen05.quantize.{sub}", "hf": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "corpus_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "source_kind": "safetensors", "op": f"quantize-{sub}",
            "level": "inference", "flags": "", "expect_quant": eq,
        })

    print(json.dumps(legs))


if __name__ == "__main__":
    main()
