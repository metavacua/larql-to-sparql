//! Multi-token extend with prior K,V checkpoint.
//!
//! Runs a forward pass over new tokens, seeding each layer's attention with
//! an optional prior K,V cache (the window boundary checkpoint). Equivalent
//! to Python `UnlimitedContextEngine.replay_window` inner loop.
//!
//! The implementation loops over tokens calling
//! `run_attention_block_decode_step`, which extends a per-layer K,V cache by
//! one position per call. After N tokens, the per-layer cache has
//! `prior_len + N` rows of K and V.
//!
//! This is O(N × L × head_ops) per window replay — matching what Python's
//! `extend()` does in a single batched call, just unrolled sequentially.
//! Slightly slower on CPU but functionally identical; the `SharedKV`
//! returned by each call carries the exact same values the batched path
//! would produce.

use ndarray::Array2;

use larql_inference::attention::{run_attention_block_decode_step, SharedKV};
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;

/// Output of `rs_extend_from_checkpoint`.
pub struct ExtendOutput {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer full K,V cache covering `[prior_tokens, new_tokens]`.
    /// Shape of each K/V: `(prior_len + new_len, num_kv * head_dim)`.
    pub kv_cache: Vec<SharedKV>,
    /// Per-layer last-row K,V, ready to save as the next boundary
    /// checkpoint. Shape of each: `(1, num_kv * head_dim)`.
    pub new_checkpoint: Vec<SharedKV>,
}

/// Run the decoder forward over `token_ids` with an optional prior K,V
/// checkpoint seeded at each layer. Returns:
///   - `last_hidden`: hidden state at the last new token
///   - `kv_cache`: full K,V per layer after extension (prior + new)
///   - `new_checkpoint`: last-row K,V per layer for saving as a boundary
///
/// `prior_kv` should contain one K,V pair per layer. Each pair's K,V may be
/// empty (0 rows) for the "no prior" case (replay of window 0) or have 1
/// row for a standard boundary checkpoint. Multi-row priors are allowed —
/// in that case attention sees the prior as a multi-token prefix.
///
/// `abs_start` is the absolute position of the *first new token* in the
/// original sequence. RoPE is applied at that position and following.
pub fn rs_extend_from_checkpoint(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: &[SharedKV],
    abs_start: usize,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;
    let ffn = WeightFfn { weights };

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

            let (h_post_attn, new_kv) =
                run_attention_block_decode_step(weights, &h, layer, kv_entry, abs_position)?;

            let (h_out, _capture) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
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

/// Build an empty (zero-row) K,V seed for use as `prior_kv` when replaying
/// window 0 or any window with no prior checkpoint.
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
