#!/usr/bin/env python3
"""Debug Gemma forward pass — compare hidden states layer by layer."""

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from mlx.utils import tree_map


def main():
    model_path = "mlx-community/functiongemma-270m-it-bf16"
    prompt = "The capital of France is"

    print(f"Model: {model_path}")
    model, tokenizer = mlx_load(model_path)

    # Cast to F32
    def to_f32(p):
        if isinstance(p, mx.array) and p.dtype != mx.float32:
            return p.astype(mx.float32)
        return p
    model.update(tree_map(to_f32, model.parameters()))
    mx.eval(model.parameters())

    tokens = tokenizer.encode(prompt)
    print(f"Tokens: {tokens}")

    x = mx.array([tokens])

    # Step through the model manually
    m = model.model  # the inner model

    # Embedding
    h = m.embed_tokens(x)
    embed_scale = model.args.hidden_size ** 0.5
    h = h * embed_scale
    print(f"\nAfter embedding (scaled by {embed_scale:.2f}):")
    h_np = np.array(h[0])
    print(f"  shape: {h_np.shape}")
    print(f"  last token norm: {np.linalg.norm(h_np[-1]):.4f}")
    print(f"  last token[:8]: {h_np[-1, :8]}")

    # Layers
    for i, layer in enumerate(m.layers):
        h = layer(h, mask=None)
        h_np = np.array(h[0])
        norm = np.linalg.norm(h_np[-1])
        if i < 3 or i >= len(m.layers) - 3:
            print(f"  Layer {i:2d}: norm={norm:.4f}  last[:4]={h_np[-1, :4]}")
        elif i == 3:
            print(f"  ...")

    # Final norm
    h = m.norm(h)
    h_np = np.array(h[0])
    print(f"\nAfter final norm:")
    print(f"  last token norm: {np.linalg.norm(h_np[-1]):.4f}")
    print(f"  last[:8]: {h_np[-1, :8]}")

    # Logits
    logits = model.model.embed_tokens.as_linear(h)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)

    top_k = 5
    top_idx = mx.argpartition(-last_logits, kth=top_k)[:top_k]
    top_vals = last_logits[top_idx]
    sort_idx = mx.argsort(-top_vals)
    print(f"\nTop {top_k} logits:")
    for i in sort_idx.tolist():
        idx = top_idx[i].item()
        val = top_vals[i].item()
        tok = tokenizer.decode([idx])
        print(f"  {tok!r:20s} logit={val:.3f}  id={idx}")


if __name__ == "__main__":
    main()
