//! Rotary Position Embeddings (RoPE) — position-dependent rotation of Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.

use ndarray::Array2;

/// Apply full RoPE to Q or K vectors.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    apply_rope_partial(x, num_heads, head_dim, rope_base, 1.0)
}

/// Apply RoPE with partial rotation: only the first `fraction` of each head's
/// dimensions get rotary encoding. The rest pass through unchanged.
/// fraction = 1.0 means full rotation (standard RoPE).
pub fn apply_rope_partial(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
) -> Array2<f32> {
    apply_rope_partial_at(x, num_heads, head_dim, rope_base, fraction, 0)
}

/// Apply RoPE with a positional offset — row `i` in `x` is treated as
/// token position `position_offset + i`. Use this during KV-cached
/// decode: cached K already carries RoPE for positions 0..N-1, and
/// the new token needs RoPE at position N.
pub fn apply_rope_partial_at(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let rotary_dim = ((head_dim as f64 * fraction) as usize).max(2);
    let half_rotary = rotary_dim / 2;
    let inv_freq: Vec<f64> = (0..half_rotary)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
        .collect();

    for row in 0..seq_len {
        let pos = position_offset + row;
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_rotary {
                let theta = pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[row, offset + i]];
                let x1 = x[[row, offset + half_rotary + i]];

                out[[row, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[row, offset + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}
