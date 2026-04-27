#!/usr/bin/env python3
"""Debug Gemma layer 0 step by step."""

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
    layer0 = m.layers[0]

    # Embedding
    h = m.embed_tokens(x)
    hidden_size = h.shape[-1]
    h = h * (hidden_size ** 0.5)
    mx.eval(h)

    h_np = np.array(h[0])
    print(f"embed: norm={np.linalg.norm(h_np[-1]):.4f} last[:4]={h_np[-1,:4]}")

    # Layer 0 attention
    # Gemma3 uses: input_layernorm -> attention -> post_attention_layernorm -> residual add
    residual = h

    # Input layernorm
    h_norm = layer0.input_layernorm(h)
    mx.eval(h_norm)
    h_norm_np = np.array(h_norm[0])
    print(f"L0 input_layernorm: norm={np.linalg.norm(h_norm_np[-1]):.4f} last[:4]={h_norm_np[-1,:4]}")

    # Self attention
    attn_out = layer0.self_attn(h_norm)
    mx.eval(attn_out)
    attn_np = np.array(attn_out[0])
    print(f"L0 attn_out: norm={np.linalg.norm(attn_np[-1]):.4f} last[:4]={attn_np[-1,:4]}")

    # Post attention layernorm (Gemma has post-norms)
    attn_normed = layer0.post_attention_layernorm(attn_out)
    mx.eval(attn_normed)
    attn_normed_np = np.array(attn_normed[0])
    print(f"L0 post_attn_layernorm: norm={np.linalg.norm(attn_normed_np[-1]):.4f} last[:4]={attn_normed_np[-1,:4]}")

    # Residual add
    h = residual + attn_normed
    mx.eval(h)
    h_np = np.array(h[0])
    print(f"L0 h_post_attn: norm={np.linalg.norm(h_np[-1]):.4f} last[:4]={h_np[-1,:4]}")

    # Pre-feedforward layernorm
    h_ffn_input = layer0.pre_feedforward_layernorm(h)
    mx.eval(h_ffn_input)
    h_ffn_np = np.array(h_ffn_input[0])
    print(f"L0 pre_ffn_layernorm: norm={np.linalg.norm(h_ffn_np[-1]):.4f} last[:4]={h_ffn_np[-1,:4]}")

    # FFN
    ffn_out = layer0.mlp(h_ffn_input)
    mx.eval(ffn_out)
    ffn_np = np.array(ffn_out[0])
    print(f"L0 ffn_out: norm={np.linalg.norm(ffn_np[-1]):.4f} last[:4]={ffn_np[-1,:4]}")

    # Post feedforward layernorm
    ffn_normed = layer0.post_feedforward_layernorm(ffn_out)
    mx.eval(ffn_normed)
    ffn_normed_np = np.array(ffn_normed[0])
    print(f"L0 post_ffn_layernorm: norm={np.linalg.norm(ffn_normed_np[-1]):.4f} last[:4]={ffn_normed_np[-1,:4]}")

    # Final residual add
    h = h + ffn_normed
    mx.eval(h)
    h_np = np.array(h[0])
    print(f"L0 final: norm={np.linalg.norm(h_np[-1]):.4f} last[:4]={h_np[-1,:4]}")


if __name__ == "__main__":
    main()
