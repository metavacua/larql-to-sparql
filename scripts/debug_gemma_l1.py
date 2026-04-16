#!/usr/bin/env python3
"""Debug Gemma layer 1 — compare attention and FFN output."""

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx.utils import tree_map


def norm(x_np):
    return np.linalg.norm(x_np[-1])


def main():
    model_path = "mlx-community/functiongemma-270m-it-bf16"
    prompt = "The capital of France is"

    model, tokenizer = mlx_load(model_path)
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) and p.dtype != mx.float32 else p, model.parameters()))
    mx.eval(model.parameters())

    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])
    m = model.model

    # Run through embedding + layer 0
    h = m.embed_tokens(x) * (model.args.hidden_size ** 0.5)
    h = m.layers[0](h, mask=None)
    mx.eval(h)
    print(f"After L0: norm={norm(np.array(h[0])):.6f}")

    # Now trace layer 1 step by step
    layer1 = m.layers[1]
    attn = layer1.self_attn

    residual = h

    # Input layernorm
    h_norm = layer1.input_layernorm(h)
    mx.eval(h_norm)
    print(f"L1 input_layernorm: norm={norm(np.array(h_norm[0])):.6f}")

    # Self attention
    attn_out = attn(h_norm)
    mx.eval(attn_out)
    print(f"L1 attn_out: norm={norm(np.array(attn_out[0])):.6f}")

    # Post attention layernorm
    attn_normed = layer1.post_attention_layernorm(attn_out)
    mx.eval(attn_normed)
    print(f"L1 post_attn_norm: norm={norm(np.array(attn_normed[0])):.6f}")

    # Residual add
    h = residual + attn_normed
    mx.eval(h)
    print(f"L1 h_post_attn: norm={norm(np.array(h[0])):.6f}")

    # Pre FFN norm
    h_ffn = layer1.pre_feedforward_layernorm(h)
    mx.eval(h_ffn)
    print(f"L1 pre_ffn_norm: norm={norm(np.array(h_ffn[0])):.6f}")

    # FFN
    ffn_out = layer1.mlp(h_ffn)
    mx.eval(ffn_out)
    print(f"L1 ffn_out: norm={norm(np.array(ffn_out[0])):.6f}")

    # Post FFN norm
    ffn_normed = layer1.post_feedforward_layernorm(ffn_out)
    mx.eval(ffn_normed)
    print(f"L1 post_ffn_norm: norm={norm(np.array(ffn_normed[0])):.6f}")

    # Final residual
    h = h + ffn_normed
    mx.eval(h)
    print(f"L1 final: norm={norm(np.array(h[0])):.6f}")

    # Manual SDPA verification
    print("\n--- L1 manual SDPA verification ---")
    B, L, _ = h_norm.shape
    seq = L if isinstance(L, int) else L.item()
    q2 = attn.q_proj(h_norm).reshape(B, seq, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k2 = attn.k_proj(h_norm).reshape(B, seq, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v2 = attn.v_proj(h_norm).reshape(B, seq, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q2 = attn.q_norm(q2)
    k2 = attn.k_norm(k2)
    q2 = attn.rope(q2)
    k2 = attn.rope(k2)
    mx.eval(q2, k2, v2)

    # Manual head 0 attention
    q0 = np.array(q2[0, 0])  # (seq, hd)
    k0 = np.array(k2[0, 0])  # (seq, hd)
    v0 = np.array(v2[0, 0])  # (seq, hd)
    scores = q0 @ k0.T * attn.scale
    for i in range(seq):
        for j in range(i+1, seq):
            scores[i, j] = -1e9
    exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
    out0 = weights @ v0
    print(f"Manual head 0 last token out norm: {np.linalg.norm(out0[-1]):.6f}")
    print(f"Attn weights last token: {weights[-1]}")

    # Also break down attention
    print("\n--- L1 attention breakdown ---")
    B, L, _ = h_norm.shape
    q = attn.q_proj(h_norm)
    k = attn.k_proj(h_norm)
    v = attn.v_proj(h_norm)
    mx.eval(q, k, v)
    print(f"q raw: norm={norm(np.array(q[0])):.6f}")
    print(f"k raw: norm={norm(np.array(k[0])):.6f}")
    print(f"v raw: norm={norm(np.array(v[0])):.6f}")

    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads
    q = q.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)

    q = attn.q_norm(q)
    k = attn.k_norm(k)
    mx.eval(q, k)
    print(f"q after qk_norm: head0 norm={np.linalg.norm(np.array(q[0,0,-1])):.6f}")

    q = attn.rope(q)
    k = attn.rope(k)
    mx.eval(q, k)
    print(f"q after rope: head0 norm={np.linalg.norm(np.array(q[0,0,-1])):.6f}")
    print(f"q after rope: head0 last[:4]={np.array(q[0,0,-1,:4])}")


if __name__ == "__main__":
    main()
