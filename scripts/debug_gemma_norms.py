#!/usr/bin/env python3
"""Print per-layer hidden state norms for Gemma — MLX F32 reference."""

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx.utils import tree_map


def main():
    model_path = "mlx-community/functiongemma-270m-it-bf16"
    prompt = "The capital of France is"

    model, tokenizer = mlx_load(model_path)
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) and p.dtype != mx.float32 else p, model.parameters()))
    mx.eval(model.parameters())

    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])
    m = model.model

    # Embedding
    h = m.embed_tokens(x)
    h = h * (model.args.hidden_size ** 0.5)
    mx.eval(h)
    h_np = np.array(h[0])
    print(f"embed: norm={np.linalg.norm(h_np[-1]):.6f} last[:4]={h_np[-1,:4]}")

    # Each layer
    for i, layer in enumerate(m.layers):
        h = layer(h, mask=None)
        mx.eval(h)
        h_np = np.array(h[0])
        norm = np.linalg.norm(h_np[-1])
        print(f"L{i:2d}: norm={norm:.6f} last[:4]={h_np[-1,:4]}")

    # Final norm
    h = m.norm(h)
    mx.eval(h)
    h_np = np.array(h[0])
    print(f"final_norm: norm={np.linalg.norm(h_np[-1]):.6f} last[:4]={h_np[-1,:4]}")

    # Logits
    logits = model.model.embed_tokens.as_linear(h)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)
    top5 = mx.argpartition(-last_logits, kth=5)[:5]
    top_vals = last_logits[top5]
    sort_idx = mx.argsort(-top_vals)
    print(f"\nTop 5 logits:")
    for i in sort_idx.tolist():
        idx = top5[i].item()
        val = top_vals[i].item()
        tok = tokenizer.decode([idx])
        print(f"  {tok!r:15s} logit={val:.4f}")


if __name__ == "__main__":
    main()
