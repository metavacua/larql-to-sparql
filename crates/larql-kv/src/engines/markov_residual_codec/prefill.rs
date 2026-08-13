//! Codec-cold-tier prefill: build the store the decode step walks.
//!
//! Transactional by construction, exactly as the markov twin: the store is
//! built into locals and handed back only on success, so a refusal leaves an
//! engine's existing store untouched.

// DISCREPANCY vs the round-1 survey (WHOLESALE_NATIVE by transitive
// dependency on `recompute_kv`): same split as the markov_residual
// twin's prefill.rs -- `RsPrefillResultCodec` (pure data) stays
// portable; `rs_prefill_codec` and its native-only imports are gated.
// `roundtrip` is pure codec math (no recompute_kv/VectorIndex touch)
// but its only call site is inside the now-gated `rs_prefill_codec`,
// so it's gated too (CI-confirmed via wasm32 clippy dead-code, not
// left "portable but uncalled").
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

#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual::recompute_kv;
// ColdResidualCodec/EncodedColdLayer's only uses (rs_prefill_codec,
// roundtrip) are both native-gated; RsStoreCodec stays portable -- it's
// the field type of the still-portable RsPrefillResultCodec below.
#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
// last_row's own consumer here (rs_prefill_codec) is native-gated; see
// helpers.rs::last_row for the full native-only rationale.
#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual_codec::helpers::last_row;
#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual_codec::store::EncodedColdLayer;
use crate::engines::markov_residual_codec::store::RsStoreCodec;

pub struct RsPrefillResultCodec {
    pub hidden: Array2<f32>,
    pub store: RsStoreCodec,
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn rs_prefill_codec(
    weights: larql_inference::WeightsView,
    token_ids: &[u32],
    max_window: Option<usize>,
    codec: ColdResidualCodec,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
) -> Result<RsPrefillResultCodec, EngineError> {
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
                details: format!("attention returned None during codec prefill at layer {layer}"),
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

    let hidden_size = weights.hidden_size;
    let mut rs = RsStoreCodec {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_encoded: None,
        cold_kv: None,
        // Dense (f32) prefill path doesn't capture K/V — falls back to
        // recompute-from-residuals on decode. The Q4K walk path
        // (`rs_prefill_codec_walk`) is what production uses, and it
        // does capture.
        hot_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
        codec,
    };

    // Clip overflow per layer; encode and pre-compute K/V for cold once.
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    rs.finalise_hot_len_after_clip();
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let decoded_overflow = roundtrip(overflow, codec);
            cold_kv.push(
                recompute_kv(weights, &decoded_overflow, layer, 0, backend, None).ok_or_else(
                    || EngineError::BackendFailure {
                        details: format!("cold K/V pre-computation returned None at layer {layer}"),
                    },
                )?,
            );
            let mut enc = EncodedColdLayer::empty(hidden_size);
            enc.append(codec, overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    Ok(RsPrefillResultCodec {
        hidden: last_row(&h),
        store: rs,
    })
}

/// Apply the codec roundtrip to a block. Used during prefill cold setup so
/// that the cold K/V we precompute is consistent with what `decode` would
/// later produce.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = EncodedColdLayer::empty(block.shape()[1]);
    tmp.append(codec, block);
    tmp.decode(codec)
}

#[cfg(test)]
mod tests {
    use super::{roundtrip, rs_prefill_codec, ColdResidualCodec};
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;
    use ndarray::Array2;

    /// Prompt/window fixtures. Named so an arity change reads as a decision.
    const PROMPT: [u32; 3] = [0, 1, 2];
    const OVERFLOW_PROMPT: [u32; 4] = [0, 1, 2, 3];
    const WINDOW: usize = 2;
    /// `Bf16` is two bytes per element — the payload-size assertion below.
    const BF16_BYTES: usize = 2;

    fn prefill(
        weights: &larql_inference::ModelWeights,
        tokens: &[u32],
        window: Option<usize>,
    ) -> super::RsPrefillResultCodec {
        rs_prefill_codec(
            larql_inference::WeightsView::dense(weights),
            tokens,
            window,
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        )
        .expect("a dense prefill with no MoE hook cannot refuse")
    }

    #[test]
    fn prefill_returns_finite_hidden() {
        let weights = make_test_weights();
        let result = prefill(&weights, &PROMPT, None);
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_no_window_does_not_create_cold_tier() {
        let weights = make_test_weights();
        let result = prefill(&weights, &PROMPT[..2], None);
        assert!(result.store.cold_encoded.is_none());
        assert!(result.store.cold_kv.is_none());
    }

    #[test]
    fn prefill_with_overflow_creates_encoded_cold_tier() {
        let weights = make_test_weights();
        let result = prefill(&weights, &OVERFLOW_PROMPT, Some(WINDOW));
        assert!(result.store.cold_encoded.is_some());
        assert!(result.store.cold_kv.is_some());
        let layers = result.store.cold_encoded.as_ref().unwrap();
        assert_eq!(layers.len(), weights.num_layers);
        let cold_positions = OVERFLOW_PROMPT.len() - WINDOW;
        for l in layers {
            assert_eq!(l.n_positions, cold_positions);
            assert_eq!(
                l.payload.len(),
                cold_positions * weights.hidden_size * BF16_BYTES
            );
        }
    }

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn roundtrip_preserves_within_bf16_precision() {
        /// Loosest gap a bf16 round-trip may open on the small integers below.
        const BF16_TOLERANCE: f32 = 0.1;
        let block =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = roundtrip(&block, ColdResidualCodec::Bf16);
        for (orig, got) in block.iter().zip(out.iter()) {
            assert!((orig - got).abs() < BF16_TOLERANCE);
        }
    }
}
