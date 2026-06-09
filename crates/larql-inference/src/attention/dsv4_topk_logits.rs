//! Top-k logits extraction — building block for sampling and parity.
//!
//! The DSv4 model forward produces `(n_tokens, n_vocab)` logits. For
//! both sampling/generation and for HF/llama.cpp parity comparison
//! the useful question is "what are the top-k highest-scoring token
//! IDs for each position?". Token-ID agreement is robust to small
//! numerical drift, which makes top-k a better cross-implementation
//! parity signal than direct logit comparison.
//!
//! This module is dependency-free (no GGUF or model state) — it only
//! consumes the final logits tensor.

use ndarray::ArrayView2;

/// One (token_id, logit, softmax probability) entry in a top-k result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopKEntry {
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
}

/// Extract the top-k token IDs per row of a `(n_tokens, n_vocab)`
/// logits tensor. Returns `n_tokens` rows, each holding `k` entries
/// sorted by descending logit.
///
/// The probabilities are computed as a per-row softmax over the
/// **full vocab** (not just the top-k subset), giving the actual
/// model-output probability for each top token. Two rows can have
/// different probability masses among their top-k slices since
/// softmax sums over all `n_vocab` entries.
///
/// `k` must be > 0 and ≤ `n_vocab`.
pub fn dsv4_topk_logits(logits: ArrayView2<f32>, k: usize) -> Vec<Vec<TopKEntry>> {
    let n_tokens = logits.shape()[0];
    let n_vocab = logits.shape()[1];
    assert!(k > 0, "k must be > 0");
    assert!(k <= n_vocab, "k {k} exceeds n_vocab {n_vocab}");

    let mut out = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        // Partial sort: index every vocab entry, then keep the top k.
        // n_vocab is ~129k for DSv4-Flash and k is typically small
        // (1..32), so an O(n log k) heap-based partial sort is cheap.
        let mut indexed: Vec<(u32, f32)> = (0..n_vocab as u32)
            .map(|v| (v, logits[[t, v as usize]]))
            .collect();
        // Sort descending by logit, stable. For k « n_vocab, a
        // partial-sort would be faster but this stays simple.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax probability over the full row, using the row max
        // for numerical stability.
        let row_max = indexed[0].1; // post-sort, first entry has max logit
        let mut exp_sum = 0.0_f64;
        for v in 0..n_vocab {
            exp_sum += ((logits[[t, v]] - row_max) as f64).exp();
        }
        let inv_sum = 1.0 / exp_sum;

        let top: Vec<TopKEntry> = indexed
            .into_iter()
            .take(k)
            .map(|(token_id, logit)| {
                let p = ((logit - row_max) as f64).exp() * inv_sum;
                TopKEntry {
                    token_id,
                    logit,
                    probability: p as f32,
                }
            })
            .collect();
        out.push(top);
    }
    out
}

/// Greedy argmax — convenience wrapper for `dsv4_topk_logits(.., 1)`.
/// Returns the highest-logit token id for each token position.
pub fn dsv4_argmax_logits(logits: ArrayView2<f32>) -> Vec<u32> {
    let topk = dsv4_topk_logits(logits, 1);
    topk.into_iter().map(|row| row[0].token_id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Single-row logits — top-k is the k entries with descending logit.
    #[test]
    fn topk_single_row_sorted_descending() {
        // Vocab: 5 entries; logits = [1.0, 4.0, 2.0, 5.0, 3.0].
        let logits = Array2::<f32>::from_shape_vec((1, 5), vec![1.0, 4.0, 2.0, 5.0, 3.0]).unwrap();
        let top = dsv4_topk_logits(logits.view(), 3);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].len(), 3);
        // Token IDs sorted by descending logit: 3 (logit 5), 1 (4), 4 (3).
        assert_eq!(top[0][0].token_id, 3);
        assert_eq!(top[0][0].logit, 5.0);
        assert_eq!(top[0][1].token_id, 1);
        assert_eq!(top[0][1].logit, 4.0);
        assert_eq!(top[0][2].token_id, 4);
        assert_eq!(top[0][2].logit, 3.0);
    }

    /// k=1 reduces to argmax; argmax helper agrees with topk-1.
    #[test]
    fn argmax_agrees_with_topk_1() {
        let logits =
            Array2::<f32>::from_shape_fn((3, 7), |(t, v)| ((t * 13 + v) as f32 * 0.31).sin());
        let argmax = dsv4_argmax_logits(logits.view());
        let topk1 = dsv4_topk_logits(logits.view(), 1);
        for (t, row) in topk1.iter().enumerate() {
            assert_eq!(row[0].token_id, argmax[t]);
        }
    }

    /// Probabilities sum to 1.0 across the FULL vocab — not just the
    /// top-k slice (they only partially sum since they're sub-rows of
    /// the row softmax).
    #[test]
    fn topk_probability_invariants() {
        // 5 entries with widely separated logits: top-1 prob ≈ 1.0.
        let logits =
            Array2::<f32>::from_shape_vec((1, 5), vec![-100.0, -100.0, 50.0, -100.0, -100.0])
                .unwrap();
        let top = dsv4_topk_logits(logits.view(), 5);
        // Sum of probabilities over the top-5 (= full vocab) should
        // be 1.0 within f32 epsilon.
        let total: f32 = top[0].iter().map(|e| e.probability).sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax sum {total} != 1.0");
        // Top-1 (the only non-suppressed entry) should be ≈ 1.0.
        assert!(
            top[0][0].probability > 0.999,
            "top-1 prob {} too low",
            top[0][0].probability
        );
        // All but top-1 should be ≈ 0.
        for e in &top[0][1..] {
            assert!(
                e.probability < 1e-5,
                "expected ~0 prob, got {}",
                e.probability
            );
        }
    }

    /// Each row's probabilities are independent — different rows can
    /// have different distributions.
    #[test]
    fn topk_rows_are_independent() {
        // Row 0: token 0 dominates. Row 1: token 1 dominates.
        let logits = Array2::<f32>::from_shape_vec(
            (2, 3),
            vec![
                10.0, 0.0, 0.0, // row 0
                0.0, 10.0, 0.0, // row 1
            ],
        )
        .unwrap();
        let top = dsv4_topk_logits(logits.view(), 1);
        assert_eq!(top[0][0].token_id, 0);
        assert_eq!(top[1][0].token_id, 1);
        // Both top-1 probabilities should be high (logit gap of 10
        // means softmax ≈ exp(10)/(exp(10)+1+1) ≈ 0.9999).
        assert!(top[0][0].probability > 0.999);
        assert!(top[1][0].probability > 0.999);
    }

    /// Numerical stability with large logits — no NaN/Inf via the
    /// row-max subtraction.
    #[test]
    fn topk_handles_large_logits() {
        // Logits in the hundreds — without max-subtract these overflow exp.
        let logits =
            Array2::<f32>::from_shape_vec((1, 4), vec![800.0, 750.0, 802.0, 799.0]).unwrap();
        let top = dsv4_topk_logits(logits.view(), 4);
        for e in &top[0] {
            assert!(e.probability.is_finite(), "non-finite probability");
            assert!(e.probability >= 0.0);
        }
        assert_eq!(top[0][0].token_id, 2); // highest logit 802
    }

    /// Asserts on invalid k.
    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn topk_k_zero_panics() {
        let logits = Array2::<f32>::zeros((1, 5));
        let _ = dsv4_topk_logits(logits.view(), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds n_vocab")]
    fn topk_k_exceeds_vocab_panics() {
        let logits = Array2::<f32>::zeros((1, 5));
        let _ = dsv4_topk_logits(logits.view(), 6);
    }
}
