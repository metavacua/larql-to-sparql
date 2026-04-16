#!/usr/bin/env python3
"""Debug Gemma attention step by step at layer 0."""

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
    attn = layer0.self_attn

    # Embedding
    h = m.embed_tokens(x)
    hidden_size = model.args.hidden_size
    h = h * (hidden_size ** 0.5)
    mx.eval(h)

    # Input layernorm
    h_norm = layer0.input_layernorm(h)
    mx.eval(h_norm)
    h_norm_np = np.array(h_norm[0])
    print(f"input_layernorm: norm={np.linalg.norm(h_norm_np[-1]):.4f} last[:4]={h_norm_np[-1,:4]}")

    # Q/K/V projections
    B, L, _ = h_norm.shape
    q = attn.q_proj(h_norm)
    k = attn.k_proj(h_norm)
    v = attn.v_proj(h_norm)
    mx.eval(q, k, v)

    q_np = np.array(q[0])
    k_np = np.array(k[0])
    v_np = np.array(v[0])
    print(f"q raw: norm={np.linalg.norm(q_np[-1]):.4f} last[:4]={q_np[-1,:4]}")
    print(f"k raw: norm={np.linalg.norm(k_np[-1]):.4f} last[:4]={k_np[-1,:4]}")
    print(f"v raw: norm={np.linalg.norm(v_np[-1]):.4f} last[:4]={v_np[-1,:4]}")

    # Reshape for heads
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim = q_np.shape[-1] // n_heads
    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    q = q.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

    # QK norm
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    mx.eval(q, k)
    q_np = np.array(q[0])
    k_np = np.array(k[0])
    print(f"q after qk_norm: head0 last norm={np.linalg.norm(q_np[0,-1]):.4f} last[:4]={q_np[0,-1,:4]}")
    print(f"k after qk_norm: head0 last norm={np.linalg.norm(k_np[0,-1]):.4f} last[:4]={k_np[0,-1,:4]}")

    # RoPE
    q = attn.rope(q)
    k = attn.rope(k)
    mx.eval(q, k)
    q_np = np.array(q[0])
    k_np = np.array(k[0])
    print(f"q after rope: head0 last norm={np.linalg.norm(q_np[0,-1]):.4f} last[:4]={q_np[0,-1,:4]}")
    print(f"k after rope: head0 last norm={np.linalg.norm(k_np[0,-1]):.4f} last[:4]={k_np[0,-1,:4]}")

    # Attention
    from mlx.nn.layers.transformer import scaled_dot_product_attention
    output = scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=None)
    mx.eval(output)
    out_np = np.array(output[0])
    print(f"attn_output: norm={np.linalg.norm(out_np[:,-1,:].flatten()):.4f}")

    # Reshape back
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # O projection
    o_out = attn.o_proj(output)
    mx.eval(o_out)
    o_np = np.array(o_out[0])
    print(f"after o_proj: norm={np.linalg.norm(o_np[-1]):.4f} last[:4]={o_np[-1,:4]}")


if __name__ == "__main__":
    main()
