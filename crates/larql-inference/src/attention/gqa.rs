//! Grouped-Query Attention (GQA) — causal attention with BLAS-fused dot products.
//!
//! Memory-efficient: O(seq) per position, never materializes full [seq, seq] matrix.
//! Uses BLAS gemv for both Q·K scores and softmax·V accumulation.

use ndarray::Array2;
use super::AttentionWeights;

/// GQA with causal masking (no weight capture).
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(q, k, v, num_q, head_dim, reps, scale, seq_len, false, None);
    out
}

/// GQA that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;

            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            let max_val = scores_buf[..causal_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for score in scores_buf.iter_mut().take(causal_len) {
                let e = ((*score - max_val) as f64).exp();
                *score = e as f32;
                sum += e;
            }
            let inv_sum = (1.0 / sum) as f32;
            for score in scores_buf.iter_mut().take(causal_len) {
                *score *= inv_sum;
            }

            if capture && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
                captured_heads.push(captured);
            }

            let v_block = v.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..causal_len]);
            let weighted_v = v_block.t().dot(&scores_view);

            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
    }

    let weights = if capture {
        Some(AttentionWeights { heads: captured_heads })
    } else {
        None
    };

    (out, weights)
}
