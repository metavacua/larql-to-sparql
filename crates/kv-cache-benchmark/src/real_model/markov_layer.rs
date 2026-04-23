//! Markov Residual Stream (RS) strategy on the real model.
//!
//! ## Core claim
//!
//! The pre-layer residual vector IS the complete Markov state of the
//! transformer at that position.  Proven empirically on Gemma 3-4B:
//! transplanting full residuals from one forward pass into another
//! produces KL divergence = 0.0.  No K/V cache is needed; K and V can be
//! recomputed from the stored residual at decode time at zero information
//! loss.
//!
//! ## Three-tier storage
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Cold tier   │       Hot window        │    New token           │
//! │  (evicted)   │  (last W positions)     │    (current decode)    │
//! │  residuals   │    residuals            │    embedded            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! - **Hot window** (`stored`): the last `W` pre-layer residuals per layer,
//!   shape `[W, hidden_dim]`. These are recomputed into K/V at every decode
//!   step. W is small (e.g. 6–24 for the bounded-state experiment; 32 768
//!   for production RS+CA).
//!
//! - **Cold tier** (`cold_residuals`): residuals evicted from the hot window
//!   during prefill are *kept* rather than discarded. At decode time these
//!   are prepended to the hot window so the full attention prefix is
//!   visible, matching full-KV output exactly (cos h = 1.000000).
//!
//!   This is the Rust port of the Python `extend()` / `replay_window()`
//!   mechanism in `rs_generator.py` / `unlimited_engine.py`.
//!
//! - **New token** (`h_new`): the freshly embedded token being decoded.
//!   Its pre-layer residual is appended to the hot window after each step.
//!
//! ## Memory accounting (Gemma 3-4B: hidden=2560, num_kv=4, head_dim=256)
//!
//! ```text
//! Storage kind          Bytes / position / layer
//! ─────────────────────────────────────────────
//! Hot-window residual   10,240  (f32, hidden_dim × 4)
//! Cold-tier residual    10,240  (same — full residual saved)
//! Standard KV (fp16)     4,096  (K + V × num_kv × head_dim × 2 bytes)
//! ```
//!
//! For bounded-window decode experiments the cold tier stores the full
//! prefill history, so total memory equals standard KV × 2.5.  The
//! production boundary-residual approach (store one summary residual per
//! window boundary + token IDs for replay) reduces cold storage to
//! ≈ 4 bytes/token — the v12 "56 GB → 2.1 MB" insight — but that
//! optimisation is orthogonal to the Markov correctness claim tested here.
//!
//! ## Decode step
//!
//! ```text
//! For each layer:
//!   1. full_h = concat([cold_residuals[l], hot_window[l]])  // [C+W, hidden]
//!   2. (K, V) = recompute_kv(full_h, abs_start=cold_abs_start)
//!               (layernorm → K/V proj → QK-norm → RoPE at original positions)
//!   3. h_new  = GQA(Q_new, K, V)   // single-token query against full history
//!   4. h_new  = FFN(h_new)
//!   5. Append h_new residual to hot window; clip overflow to cold tier.
//! ```

use ndarray::{Array2, s};
use larql_inference::model::ModelWeights;
use larql_inference::forward::{embed_tokens_pub, run_ffn, apply_norm, dot_proj, add_bias};
use larql_inference::attention::{
    run_attention_with_kv, run_attention_block_decode_step,
    apply_rope_partial_at,
};
use larql_inference::residual::{rms_norm_heads, rms_norm_heads_no_weight};
use larql_inference::ffn::WeightFfn;

/// Per-layer pre-attention residuals for all stored positions.
/// `stored[i]` shape: `[S, hidden_dim]` — the residual entering layer `i`
/// for positions `[next_position - S, next_position)`.
///
/// Cold-tier: when the hot window is smaller than the full sequence,
/// the evicted rows are saved in `cold_residuals` (one per layer). At
/// decode time both tiers are concatenated so attention covers the full
/// history — same as the Python `extend()` replay mechanism.
pub struct RsStore {
    pub stored: Vec<Array2<f32>>,
    /// Evicted (cold-tier) residuals: `cold_residuals[i]` holds rows that
    /// were clipped from `stored[i]`. `None` when no eviction has occurred.
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    /// Absolute position of the first token in the cold tier (0 if no cold tier).
    pub cold_abs_start: usize,
    /// Absolute token position of the NEXT token to be appended.
    pub next_position: usize,
    /// Optional sliding window: if `Some(W)`, only the last W residuals
    /// are kept per layer; older ones are moved to the cold tier.
    pub max_window: Option<usize>,
}

impl RsStore {
    /// Memory used by the stored residuals in bytes (f32).
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        hot + cold
    }

    /// Evict old positions beyond the window, saving them in the cold tier.
    pub(crate) fn clip_layer(&mut self, layer: usize, cold: &mut Vec<Array2<f32>>) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let s = &self.stored[layer];
        let rows = s.shape()[0];
        if rows <= window {
            cold.push(Array2::zeros((0, s.shape()[1])));
            return;
        }
        let start = rows - window;
        cold.push(s.slice(s![..start, ..]).to_owned());
        self.stored[layer] = s.slice(s![start.., ..]).to_owned();
    }
}

/// Result of an RS prefill or decode step.
pub struct RsMarkovResult {
    /// Final hidden state (last token position) after the forward pass.
    pub hidden: Array2<f32>,
    /// Residual store — holds pre-layer residuals for the active window.
    pub store: RsStore,
    /// Total memory used by the RS store in bytes.
    pub memory_bytes: usize,
    /// Active window token count (how many positions are stored).
    pub window_tokens: usize,
    /// Wall clock for the forward pass in microseconds.
    pub forward_us: f64,
}

/// Run the full prefill forward pass, storing pre-layer residuals.
///
/// Equivalent to `capture_kv` but stores residuals instead of K/V.
/// The hidden state is identical — this is the same forward pass.
pub fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
) -> RsMarkovResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let ffn = WeightFfn { weights };

    let t0 = std::time::Instant::now();

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Store the pre-layer residual — this is the Markov state for this layer.
        stored.push(h.clone());

        let (h_post_attn, _k, _v) = run_attention_with_kv(weights, &h, layer)
            .expect("attention failed");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h = h_out;
    }

    let forward_us = t0.elapsed().as_secs_f64() * 1e6;

    let mut rs = RsStore {
        stored,
        cold_residuals: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    // Apply window clipping to all layers, saving evicted rows as cold tier.
    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }

    // How many cold rows were saved (use layer 0 as reference).
    let cold_rows = cold.first().map_or(0, |c| c.shape()[0]);
    if cold_rows > 0 {
        rs.cold_residuals = Some(cold);
        // cold tier starts at position 0 (beginning of the prefill).
        rs.cold_abs_start = 0;
    }

    let window_tokens = rs.stored.first().map_or(0, |s| s.shape()[0]);
    let memory_bytes = rs.memory_bytes();

    RsMarkovResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
        forward_us,
    }
}

/// Run one decode step for a new token using the RS store.
///
/// For each layer:
///   1. Recompute K/V from stored residuals (norm → proj → k-norm → RoPE at
///      original positions).
///   2. Run single-token decode attention against [K_old | K_new].
///   3. Run FFN on the new token.
///   4. Append the pre-layer residual of the new token to the store.
///
/// Returns the updated hidden state (1 × hidden_dim) and updated store.
pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
) -> Option<(Array2<f32>, RsStore)> {
    let num_layers = weights.num_layers;
    let ffn = WeightFfn { weights };
    let abs_position = rs.next_position;

    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer]; // [S_hot, hidden_dim]
        let s_hot = h_hot.shape()[0];

        // Concatenate cold tier + hot tier for full-history attention.
        let (h_full, full_abs_start) = if let Some(cold) = &rs.cold_residuals {
            let h_cold = &cold[layer];
            let s_cold = h_cold.shape()[0];
            if s_cold > 0 {
                let hidden = h_hot.shape()[1];
                let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                (combined, rs.cold_abs_start)
            } else {
                (h_hot.clone(), abs_position.saturating_sub(s_hot))
            }
        } else {
            (h_hot.clone(), abs_position.saturating_sub(s_hot))
        };

        // Recompute K/V from full history (cold + hot).
        let (k_recomputed, v_recomputed) =
            recompute_kv(weights, &h_full, layer, full_abs_start)?;

        // Save pre-layer residual for the new token before processing.
        new_stored.push(h_new.clone());

        // Decode-step attention: new token Q against [K_old | K_new].
        let (h_post_attn, _new_kv) = run_attention_block_decode_step(
            weights, &h_new, layer, Some(&(k_recomputed, v_recomputed)), abs_position,
        )?;

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h_new = h_out;
    }

    // Merge old hot residuals with new token's pre-layer residual.
    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    // Preserve cold tier; carry cold_abs_start forward.
    let cold_residuals = rs.cold_residuals;
    let cold_abs_start = rs.cold_abs_start;
    let max_window = rs.max_window;

    let mut updated_rs = RsStore {
        stored: updated_stored,
        cold_residuals,
        cold_abs_start,
        next_position: abs_position + 1,
        max_window,
    };

    // Clip hot tier; any newly evicted rows accumulate into the cold tier.
    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    // Merge overflow into existing cold tier (append at the end of each layer).
    let overflow_rows = overflow.first().map_or(0, |c| c.shape()[0]);
    if overflow_rows > 0 {
        match updated_rs.cold_residuals.as_mut() {
            Some(cold) => {
                for layer in 0..num_layers {
                    let hidden = cold[layer].shape()[1];
                    let c_old = cold[layer].shape()[0];
                    let c_new = overflow[layer].shape()[0];
                    let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                    merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                    merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                    cold[layer] = merged;
                }
            }
            None => {
                updated_rs.cold_residuals = Some(overflow);
            }
        }
    }

    Some((last_row(&h_new), updated_rs))
}

/// Recompute K/V from stored pre-layer residuals.
///
/// Mirrors the Python `_raw_step` K/V recomputation:
///   x_old = layernorm(h_old)
///   k_old = k_proj(x_old) → k_norm → RoPE at positions abs_start..
///   v_old = v_proj(x_old) → v_norm
pub(crate) fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,   // [S, hidden_dim]
    layer: usize,
    abs_start: usize,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };

    let h_norm = apply_norm(weights, h_stored, &arch.input_layernorm_key(layer), norm_offset);

    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k { w_k } else { weights.tensors.get(&arch.attn_v_key(layer))? };

    let mut k = dot_proj(&h_norm, w_k);
    let mut v = dot_proj(&h_norm, w_v);

    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v, bias);
    }

    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k, norm_w, num_kv, head_dim, qk_norm_off),
        None => k,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    // Apply RoPE at the original absolute positions of the stored tokens.
    let k_rope = apply_rope_partial_at(
        &k_normed, num_kv, head_dim, layer_rope_base, rotary_frac, abs_start,
    );

    Some((k_rope, v))
}

/// Memory used by a standard KV cache (FP16) for comparison.
pub fn kv_memory_bytes_for_seq(weights: &ModelWeights, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    let mut total = 0;
    for layer in 0..weights.num_layers {
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let kv_dim = num_kv * head_dim;
        // K + V, FP16 (2 bytes each)
        total += seq_len * kv_dim * 2 * 2;
    }
    total
}

/// Compare two hidden states (last-row cosine and MSE).
pub fn compare_hidden_states(h1: &Array2<f32>, h2: &Array2<f32>) -> (f64, f64) {
    let v1: Vec<f32> = h1.row(h1.shape()[0] - 1).to_vec();
    let v2: Vec<f32> = h2.row(h2.shape()[0] - 1).to_vec();
    let mse = crate::metrics::Metrics::compute_mse(&v1, &v2);
    let cosine = crate::metrics::Metrics::compute_cosine(&v1, &v2);
    (mse, cosine)
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rs(num_layers: usize, seq_len: usize, hidden: usize, window: Option<usize>) -> RsStore {
        let stored = (0..num_layers)
            .enumerate()
            .map(|(l, _)| {
                // Each layer gets distinct row values so splits are verifiable.
                let mut a = Array2::<f32>::zeros((seq_len, hidden));
                for i in 0..seq_len {
                    a.row_mut(i).fill((l * 1000 + i) as f32);
                }
                a
            })
            .collect();
        RsStore {
            stored,
            cold_residuals: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: window,
        }
    }

    // ── clip_layer ───────────────────────────────────────────────────────────

    #[test]
    fn clip_no_window_keeps_all() {
        let mut rs = make_rs(1, 10, 4, None);
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);
        assert_eq!(rs.stored[0].shape()[0], 10);
        assert!(cold.is_empty(), "no cold entry pushed when max_window is None");
    }

    #[test]
    fn clip_exact_window_keeps_all() {
        let mut rs = make_rs(1, 5, 4, Some(5));
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);
        assert_eq!(rs.stored[0].shape()[0], 5);
        assert_eq!(cold[0].shape()[0], 0, "no cold rows when seq_len == window");
    }

    #[test]
    fn clip_splits_hot_cold_correctly() {
        // 10 rows, window=4 → cold gets rows 0..6, hot keeps rows 6..10.
        let mut rs = make_rs(1, 10, 4, Some(4));
        let mut cold = Vec::new();
        rs.clip_layer(0, &mut cold);

        assert_eq!(cold[0].shape()[0], 6, "6 rows evicted to cold");
        assert_eq!(rs.stored[0].shape()[0], 4, "4 rows remain in hot window");

        // Cold contains the OLDEST rows (indices 0..6).
        for i in 0..6 {
            assert_eq!(cold[0][[i, 0]], i as f32, "cold row {i} has correct value");
        }
        // Hot contains the NEWEST rows (indices 6..10).
        for i in 0..4 {
            assert_eq!(rs.stored[0][[i, 0]], (6 + i) as f32, "hot row {i} has correct value");
        }
    }

    #[test]
    fn clip_multi_layer_consistent() {
        // Each layer has different values but the same split should apply.
        let mut rs = make_rs(3, 8, 4, Some(3));
        let mut cold = Vec::new();
        for layer in 0..3 {
            rs.clip_layer(layer, &mut cold);
        }
        for (l, (c, s)) in cold.iter().zip(rs.stored.iter()).enumerate() {
            assert_eq!(c.shape()[0], 5, "layer {l}: 5 cold rows");
            assert_eq!(s.shape()[0], 3, "layer {l}: 3 hot rows");
        }
    }

    // ── RsStore cold-tier field wiring (simulating rs_prefill clip) ──────────

    #[test]
    fn prefill_clip_wires_cold_residuals() {
        let num_layers = 2;
        let seq_len = 10;
        let window = 4;
        let hidden = 8;

        let mut rs = make_rs(num_layers, seq_len, hidden, Some(window));
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        let cold_rows = cold.first().map_or(0, |c| c.shape()[0]);
        assert_eq!(cold_rows, seq_len - window);

        rs.cold_residuals = Some(cold);
        rs.cold_abs_start = 0;

        assert_eq!(rs.stored[0].shape()[0], window, "hot window trimmed to {window}");
        let cold_ref = rs.cold_residuals.as_ref().unwrap();
        assert_eq!(cold_ref[0].shape()[0], seq_len - window, "cold tier has evicted rows");
        assert_eq!(rs.cold_abs_start, 0);
    }

    #[test]
    fn no_cold_when_seq_within_window() {
        let mut rs = make_rs(2, 3, 4, Some(6));
        let mut cold: Vec<Array2<f32>> = Vec::new();
        for layer in 0..2 {
            rs.clip_layer(layer, &mut cold);
        }
        let cold_rows = cold.first().map_or(0, |c| c.shape()[0]);
        assert_eq!(cold_rows, 0, "no cold tier when seq_len ≤ window");
    }

    // ── memory_bytes includes both tiers ─────────────────────────────────────

    #[test]
    fn memory_bytes_hot_only() {
        let rs = make_rs(2, 4, 8, None);
        // 2 layers × 4 rows × 8 hidden × 4 bytes = 256
        assert_eq!(rs.memory_bytes(), 2 * 4 * 8 * 4);
    }

    #[test]
    fn memory_bytes_includes_cold_tier() {
        let num_layers = 2;
        let seq_len = 10;
        let window = 4;
        let hidden = 8;
        let mut rs = make_rs(num_layers, seq_len, hidden, Some(window));
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        rs.cold_residuals = Some(cold);

        let hot_bytes  = num_layers * window            * hidden * 4;
        let cold_bytes = num_layers * (seq_len - window) * hidden * 4;
        assert_eq!(rs.memory_bytes(), hot_bytes + cold_bytes);
    }

    // ── cold-tier carry-forward in decode step ────────────────────────────────

    #[test]
    fn decode_step_overflow_merges_into_cold() {
        // Simulate the overflow merge: hot at capacity + 1 new row → 1 row
        // spills to cold, cold grows by 1.
        let window = 3;
        let hidden = 4;

        // Start: hot = [window rows], cold = [2 rows] already
        let hot: Vec<Array2<f32>> = vec![Array2::ones((window, hidden))];
        let existing_cold: Vec<Array2<f32>> = vec![Array2::zeros((2, hidden))];

        let mut rs = RsStore {
            stored: hot.clone(),
            cold_residuals: Some(existing_cold),
            cold_abs_start: 0,
            next_position: 2 + window, // cold=2, hot=3
            max_window: Some(window),
        };

        // Append one new row — hot grows to window+1, then clip evicts 1 row to overflow.
        let new_row = Array2::<f32>::from_elem((1, hidden), 9.0);
        let s_old = rs.stored[0].shape()[0];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden));
        combined.slice_mut(s![..s_old, ..]).assign(&rs.stored[0]);
        combined.slice_mut(s![s_old.., ..]).assign(&new_row);
        rs.stored[0] = combined;

        let mut overflow: Vec<Array2<f32>> = Vec::new();
        rs.clip_layer(0, &mut overflow);

        // overflow should have 1 row
        assert_eq!(overflow[0].shape()[0], 1);

        // Merge into existing cold
        if let Some(cold) = rs.cold_residuals.as_mut() {
            let c_old = cold[0].shape()[0];
            let c_new = overflow[0].shape()[0];
            let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
            merged.slice_mut(s![..c_old, ..]).assign(&cold[0]);
            merged.slice_mut(s![c_old.., ..]).assign(&overflow[0]);
            cold[0] = merged;
        }

        let cold_ref = rs.cold_residuals.as_ref().unwrap();
        assert_eq!(cold_ref[0].shape()[0], 3, "existing 2 + overflow 1 = 3 cold rows");
        assert_eq!(rs.stored[0].shape()[0], window, "hot stays at window size");
    }
}
