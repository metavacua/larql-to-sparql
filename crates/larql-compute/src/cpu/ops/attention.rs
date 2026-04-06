//! CPU causal attention: Q×K^T softmax V.
//!
//! Simple implementation for small seq_len (≤64). No tiling.
//! Uses online softmax (two-pass: max + exp/normalize).
//!
//! For the LARQL pipeline, this handles seq=1-6 where the
//! attention matrix is tiny and flash attention is overkill.

/// Causal attention for one head.
///
/// - `q`: [seq_len, head_dim] query vectors
/// - `k`: [seq_len, head_dim] key vectors
/// - `v`: [seq_len, head_dim] value vectors
/// - `scale`: 1/sqrt(head_dim)
/// - Returns: [seq_len, head_dim] attention output
pub fn causal_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, head_dim: usize, scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * head_dim];

    for qi in 0..seq_len {
        // Compute scores: q[qi] · k[0..=qi]
        let causal_len = qi + 1;

        // Find max for numerical stability
        let mut max_score = f32::NEG_INFINITY;
        for ki in 0..causal_len {
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            let score = score * scale;
            if score > max_score { max_score = score; }
        }

        // Softmax + weighted sum
        let mut sum_exp = 0.0f64;
        for ki in 0..causal_len {
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            let w = ((score * scale - max_score) as f64).exp();
            sum_exp += w;
            let w = w as f32;
            for d in 0..head_dim {
                out[qi * head_dim + d] += w * v[ki * head_dim + d];
            }
        }

        let inv_sum = 1.0 / sum_exp as f32;
        for d in 0..head_dim {
            out[qi * head_dim + d] *= inv_sum;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token_attention() {
        // seq=1: attention should just return V (softmax of one element = 1.0)
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.6, 0.7, 0.8];
        let out = causal_attention(&q, &k, &v, 1, 4, 0.5);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn causal_mask() {
        // seq=2: position 0 can only see position 0
        let q = vec![1.0, 0.0,  0.0, 1.0]; // 2 queries
        let k = vec![1.0, 0.0,  0.0, 1.0]; // 2 keys
        let v = vec![1.0, 0.0,  0.0, 1.0]; // 2 values
        let out = causal_attention(&q, &k, &v, 2, 2, 1.0);
        // Position 0 should only attend to position 0 → output = v[0]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn output_shape() {
        let seq = 6;
        let dim = 320;
        let q = vec![0.01f32; seq * dim];
        let k = vec![0.01f32; seq * dim];
        let v = vec![0.01f32; seq * dim];
        let out = causal_attention(&q, &k, &v, seq, dim, 1.0 / (dim as f32).sqrt());
        assert_eq!(out.len(), seq * dim);
    }
}
