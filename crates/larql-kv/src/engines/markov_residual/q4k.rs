//! Q4K helpers — attention dequantisation and WalkFfn-backed forward paths.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{last_row, recompute_kv, RsPrefillResult};
use super::store::RsStore;
use larql_inference::attention::run_attention_with_kv_backend;
use larql_inference::attention::SharedKV;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

/// Dequantise attention Q4K weights (Q, K, V, O) for all layers into
/// `weights.tensors`. Idempotent — skips layers already present.
pub fn ensure_attn_tensors_dequantised(weights: &mut ModelWeights, index: &VectorIndex) {
    let num_layers = weights.num_layers;
    for layer in 0..num_layers {
        let arch = &*weights.arch;
        let q_key = arch.attn_q_key(layer);
        if weights.tensors.contains_key(&q_key) {
            continue;
        }
        let Some(attn) = index.attn_q4k_layer_data(layer) else {
            continue;
        };
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let hd = arch.head_dim_for_layer(layer);
        let hidden = weights.hidden_size;
        let q_dim = num_q * hd;
        let kv_dim = num_kv * hd;
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);
        weights.tensors.insert(q_key, w_q.into_shared());
        weights.tensors.insert(k_key, w_k.into_shared());
        weights.tensors.insert(v_key, w_v.into_shared());
        weights.tensors.insert(o_key, w_o.into_shared());
    }
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch")
}

/// Prefill using `WalkFfn` (Q4K FFN) instead of `BackendFfn` (f32 FFN).
pub(super) fn rs_prefill_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS Q4K prefill");
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        stored,
        cold_residuals: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };
    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = (0..num_layers)
            .map(|layer| {
                recompute_kv(weights, &cold[layer], layer, 0, backend)
                    .expect("cold K/V pre-computation failed")
            })
            .collect();
        rs.cold_residuals = Some(cold);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }
    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

/// Decode step using `WalkFfn` (Q4K FFN).
pub(super) fn rs_decode_step_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    use ndarray::s;

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend)?;
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = match &rs.cold_residuals {
                Some(cold) if cold[layer].shape()[0] > 0 => {
                    let h_cold = &cold[layer];
                    let s_cold = h_cold.shape()[0];
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                    combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                    combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend)?
        };

        new_stored.push(h_new.clone());

        let (h_post_attn, _new_kv) =
            larql_inference::attention::run_attention_block_decode_step_backend(
                weights,
                &h_new,
                layer,
                Some(&(k_full, v_full)),
                abs_position,
                Some(backend),
            )?;
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h_new = h_out;
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    if overflow.first().map_or(0, |c| c.shape()[0]) > 0 {
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
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}
