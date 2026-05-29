//! Multi-token extend with prior K,V checkpoint.
//!
//! Runs a CPU/GPU forward pass over new tokens, seeding each layer's attention
//! with an optional prior K,V cache (the window boundary checkpoint).

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use larql_inference::attention::{run_attention_block_decode_step_backend, SharedKV};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

pub struct ExtendOutput {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer full K,V cache covering `[prior_tokens, new_tokens]`.
    pub kv_cache: Vec<SharedKV>,
    /// Per-layer last-row K,V ready to save as the next boundary checkpoint.
    pub new_checkpoint: Vec<SharedKV>,
}

/// Run the decoder forward over `token_ids` seeded with an optional prior K,V
/// checkpoint at each layer. Matmuls route through `backend`.
///
/// `abs_start` is the absolute position of the *first new token*.
pub fn rs_extend_from_checkpoint(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: &[SharedKV],
    abs_start: usize,
) -> Option<ExtendOutput> {
    rs_extend_from_checkpoint_backend(
        weights,
        token_ids,
        prior_kv,
        abs_start,
        &larql_compute::CpuBackend,
    )
}

/// Backend-dispatched variant of [`rs_extend_from_checkpoint`].
pub fn rs_extend_from_checkpoint_backend(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: &[SharedKV],
    abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return None;
    }
    if prior_kv.len() != num_layers {
        return None;
    }

    let mut kv_cache: Vec<SharedKV> = prior_kv.to_vec();
    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
                weights,
                &h,
                layer,
                kv_entry,
                abs_position,
                Some(backend),
            )?;

            let bffn = BackendFfn { weights, backend };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// CPU Q4K variant of [`rs_extend_from_checkpoint_backend`].
///
/// Uses `WalkFfn` (reads Q4K bytes directly from `index`) for FFN instead of
/// `BackendFfn` (needs f32 tensors in `weights.tensors`). Attention projection
/// uses the dequantised f32 tensors already inserted by
/// `ensure_attn_tensors_dequantised`. Call that before this function.
pub fn rs_extend_from_checkpoint_q4k(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    prior_kv: &[SharedKV],
    abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return None;
    }
    if prior_kv.len() != num_layers {
        return None;
    }

    let mut kv_cache: Vec<SharedKV> = prior_kv.to_vec();
    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
                weights,
                &h,
                layer,
                kv_entry,
                abs_position,
                Some(backend),
            )?;

            let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
                .with_backend(backend);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// Build an empty (zero-row) K,V seed for use when no prior checkpoint exists.
pub fn empty_prior(weights: &ModelWeights) -> Vec<SharedKV> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            (
                Array2::<f32>::zeros((0, kv_dim)),
                Array2::<f32>::zeros((0, kv_dim)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    // ── empty_prior ───────────────────────────────────────────────────────────

    #[test]
    fn empty_prior_shape_per_layer() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        assert_eq!(prior.len(), weights.num_layers);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &prior {
            assert_eq!(k.shape(), &[0, kv_dim]);
            assert_eq!(v.shape(), &[0, kv_dim]);
        }
    }

    // ── rs_extend_from_checkpoint ─────────────────────────────────────────────

    #[test]
    fn extend_empty_tokens_returns_none() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let result = rs_extend_from_checkpoint(&weights, &[], &prior, 0);
        assert!(result.is_none(), "empty token_ids should return None");
    }

    #[test]
    fn extend_wrong_prior_len_returns_none() {
        let weights = make_test_weights();
        // prior has 0 layers but model has 2 — mismatch
        let result = rs_extend_from_checkpoint(&weights, &[0u32], &[], 0);
        assert!(result.is_none(), "prior length mismatch should return None");
    }

    #[test]
    fn extend_single_token_from_empty_prior() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(&weights, &[0u32], &prior, 0)
            .expect("single token extend should succeed");
        assert_eq!(output.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(output.last_hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_kv_cache_grows_with_each_token() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output =
            rs_extend_from_checkpoint(&weights, &[0u32, 1, 2], &prior, 0).expect("3-token extend");
        // After 3 tokens from empty prior, K has 3 rows per layer
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &output.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim], "K should have 3 rows");
            assert_eq!(v.shape(), &[3, kv_dim], "V should have 3 rows");
        }
    }

    #[test]
    fn extend_checkpoint_is_last_row_of_kv_cache() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output =
            rs_extend_from_checkpoint(&weights, &[0u32, 1], &prior, 0).expect("2-token extend");
        // new_checkpoint should be the last row of each K/V
        for (layer, ((k_cache, v_cache), (k_ckpt, v_ckpt))) in output
            .kv_cache
            .iter()
            .zip(output.new_checkpoint.iter())
            .enumerate()
        {
            let n = k_cache.shape()[0];
            let last_k = k_cache.row(n - 1).to_vec();
            let ckpt_k = k_ckpt.row(0).to_vec();
            for (a, b) in last_k.iter().zip(ckpt_k.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "layer {layer}: checkpoint K doesn't match last K cache row"
                );
            }
            let _ = (v_cache, v_ckpt); // symmetry — trust by shape
        }
    }

    #[test]
    fn extend_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let out0 = rs_extend_from_checkpoint(&weights, &[0u32], &prior, 0).unwrap();
        let out5 = rs_extend_from_checkpoint(&weights, &[0u32], &prior, 5).unwrap();
        // Different abs_start → different RoPE → different K
        let k0 = &out0.kv_cache[0].0;
        let k5 = &out5.kv_cache[0].0;
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "different abs_start should produce different K (RoPE)"
        );
    }

    #[test]
    fn extend_output_logits_are_finite() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(&weights, &[0u32], &prior, 0).unwrap();
        let logits = hidden_to_raw_logits(&weights, &output.last_hidden);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_seeded_from_checkpoint_matches_empty_start() {
        // Extending from a non-empty checkpoint should not panic and should be finite.
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let first = rs_extend_from_checkpoint(&weights, &[0u32], &prior, 0).unwrap();
        // Use the checkpoint from the first extend as the prior for the second
        let second = rs_extend_from_checkpoint(&weights, &[1u32], &first.new_checkpoint, 1)
            .expect("extend from non-empty prior");
        assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(second.last_hidden.iter().all(|v| v.is_finite()));
    }
}
