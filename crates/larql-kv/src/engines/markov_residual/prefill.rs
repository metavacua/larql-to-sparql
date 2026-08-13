//! Residual-stream prefill: build the store the decode step walks.

// DISCREPANCY vs the round-1 survey (which classified this file
// WHOLESALE_NATIVE by transitive dependency on `recompute_kv`): the
// mod.rs re-export notes explicitly flag `RsPrefillResult` as pure data
// that "could stay portable even though its only producer (rs_prefill)
// is native" -- doing that real split here rather than gating the
// whole file, mirroring compute.rs's own kv_memory_bytes_for_seq split.
#[cfg(not(target_arch = "wasm32"))]
use larql_compute::ComputeBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::{run_attention_with_kv_backend, SharedKV};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ffn::BackendFfn;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::embed_tokens_pub;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::ple::precompute_per_layer_inputs;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::kv_engine::EngineError;
use ndarray::Array2;

// `recompute_kv` is itself cfg-gated inside compute.rs, so this import
// must be native-only regardless of `last_row`'s own portability.
#[cfg(not(target_arch = "wasm32"))]
use super::compute::{last_row, recompute_kv};
use super::store::RsStore;

pub struct RsPrefillResult {
    pub hidden: Array2<f32>,
    pub store: RsStore,
    pub memory_bytes: usize,
    pub window_tokens: usize,
}

/// Run a full prefill, returning the hidden state and a freshly built store.
///
/// **Transactional by construction.** Everything is built into locals and the
/// store is handed back only on success, so a failure costs the caller nothing
/// it already had — an engine holding an earlier store keeps it, and a caller
/// that fixes the cause can drive the same prompt again.
#[cfg(not(target_arch = "wasm32"))]
pub fn rs_prefill(
    weights: larql_inference::WeightsView,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
) -> Result<RsPrefillResult, EngineError> {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(&weights, token_ids);
    // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be, None)
            .ok_or_else(|| EngineError::BackendFailure {
                details: format!(
                    "attention returned None during MarkovRS prefill at layer {layer}"
                ),
            })?;
        let bffn = BackendFfn {
            weights: weights.canonical(),
            backend,
        };
        h = crate::engines::layer_ffn_or_moe(
            weights.canonical(),
            &h_post_attn,
            layer,
            &bffn,
            moe_ffn,
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;
    }

    let mut rs = RsStore {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_residuals: None,
        cold_kv: None,
        cold_len: 0,
        hot_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    rs.finalise_hot_len_after_clip();
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, cold_layer) in cold.iter().enumerate().take(num_layers) {
            cold_kv.push(
                recompute_kv(weights, cold_layer, layer, 0, backend, None).ok_or_else(|| {
                    EngineError::BackendFailure {
                        details: format!("cold K/V pre-computation returned None at layer {layer}"),
                    }
                })?,
            );
        }
        // 2026-05-19 audit fix: route through the doubling-capacity
        // helper so cold_len is initialised correctly. Subsequent
        // decode-step overflows then append in amortised O(1).
        rs.append_cold_overflow(cold, Some(cold_kv));
        rs.cold_abs_start = 0;
    }

    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    Ok(RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::rs_prefill;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    const PROMPT: [u32; 3] = [0, 1, 2];
    const LONG_PROMPT: [u32; 5] = [0, 1, 2, 3, 4];
    const WINDOW: usize = 2;

    fn prefill(
        weights: &larql_inference::ModelWeights,
        tokens: &[u32],
        window: Option<usize>,
    ) -> super::RsPrefillResult {
        rs_prefill(
            larql_inference::WeightsView::dense(weights),
            tokens,
            window,
            &CpuBackend,
            None,
        )
        .expect("a dense prefill with no MoE hook cannot refuse")
    }

    #[test]
    fn rs_prefill_returns_correct_shape() {
        let weights = make_test_weights();
        let result = prefill(&weights, &PROMPT, None);
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_prefill_stores_all_layers() {
        let weights = make_test_weights();
        let result = prefill(&weights, &PROMPT[..1], None);
        assert_eq!(result.store.stored.len(), weights.num_layers);
        assert_eq!(result.store.next_position, 1);
    }

    #[test]
    fn rs_prefill_with_window_clips_hot_store() {
        let weights = make_test_weights();
        let result = prefill(&weights, &LONG_PROMPT, Some(WINDOW));
        assert!(
            result.window_tokens <= WINDOW,
            "window_tokens={} > {WINDOW}",
            result.window_tokens
        );
    }
}
