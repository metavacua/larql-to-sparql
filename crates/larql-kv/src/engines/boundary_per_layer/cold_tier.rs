//! Cold-tier maintenance for `BoundaryPerLayerEngine`.
//!
//! Two responsibilities:
//!
//! 1. [`extend_cold_kv_with_overflow`] — append K/V for newly-evicted
//!    overflow rows onto each layer's existing `cold_kv` tensor.
//!    Replaces the previous "nuke cold_kv on every overflow" path
//!    which forced the next decode step to recompute K/V over the
//!    entire cold tier (bug B; O(N²) windowed-mode decode).
//! 2. [`roundtrip`] / [`last_row`] — small helpers that the walk and
//!    dispatch paths both need.
//!
//! All three are free functions so the walk, dispatch, and executor
//! paths can share them without going through `&self`. Sibling
//! modules call these via `super::cold_tier::*`.
//!
//! Lossy-codec contract: cold K/V is computed from the codec
//! round-trip of the evicted block (not the raw f32), so future
//! decode steps see a consistent set of "decode against bf16-decoded
//! residuals" K/V regardless of whether they hit the cold_kv cache
//! or recompute via cold_encoded.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;

/// Extend `cold_kv` to cover newly-evicted overflow rows.
///
/// Computes K/V on the codec round-trip of each layer's overflow
/// (preserving the lossy contract used at prefill) and concatenates
/// onto each layer's existing cold (K, V). If `cold_kv` is `None`,
/// initialises it.
///
/// `cold_abs_pos` must be the absolute position at which the new
/// overflow lands — caller MUST snapshot this BEFORE appending the
/// overflow to `cold_encoded` (which would advance `n_positions`).
///
/// Atomicity invariant: either EVERY layer's cold K/V is extended, or
/// `rs.cold_kv` is dropped wholesale and `None` is returned. A partial
/// per-layer failure must never leave `cold_kv` desynced from
/// `cold_encoded` (which the caller extends for all layers) — the walk
/// paths trust `cold_kv[layer]` row counts to line up across layers.
/// `cold_kv` is a derived cache: dropping it costs a recompute from
/// `cold_encoded` on the next decode step, never correctness.
pub(super) fn extend_cold_kv_with_overflow(
    weights: larql_inference::WeightsView,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    rs: &mut RsStorePerLayer,
    overflow_per_layer: &[Array2<f32>],
    cold_abs_pos: usize,
) -> Option<()> {
    let num_layers = weights.num_layers;
    let n_new = overflow_per_layer.first().map_or(0, |c| c.shape()[0]);
    if n_new == 0 {
        return Some(());
    }
    // Phase 1 (fallible): compute K/V for every layer BEFORE touching
    // `rs.cold_kv`, so a mid-loop failure cannot partially extend it.
    let mut new_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
        let codec = policy.codec_for(layer);
        let decoded = roundtrip(overflow, codec);
        match recompute_kv(weights, &decoded, layer, cold_abs_pos, backend, None) {
            Some(kv) => new_kv.push(kv),
            None => {
                // See the atomicity invariant above: drop the whole cache
                // and surface the failure to the caller.
                rs.cold_kv = None;
                return None;
            }
        }
    }
    // Phase 2 (infallible): splice the pre-computed rows in.
    match rs.cold_kv.as_mut() {
        Some(cold_kv) => {
            for (layer, (k_new, v_new)) in new_kv.into_iter().enumerate() {
                let (k_old, v_old) = &cold_kv[layer];
                let kv_dim = k_old.shape()[1];
                let l_old = k_old.shape()[0];
                let l_total = l_old + n_new;
                let mut k = Array2::<f32>::zeros((l_total, kv_dim));
                k.slice_mut(s![..l_old, ..]).assign(k_old);
                k.slice_mut(s![l_old.., ..]).assign(&k_new);
                let mut v = Array2::<f32>::zeros((l_total, kv_dim));
                v.slice_mut(s![..l_old, ..]).assign(v_old);
                v.slice_mut(s![l_old.., ..]).assign(&v_new);
                cold_kv[layer] = (k, v);
            }
        }
        None => {
            rs.cold_kv = Some(new_kv);
        }
    }
    Some(())
}

/// Encode `block` with `codec` then immediately decode it. Used to
/// derive the "cold K/V's view of the residuals" — see file docs.
pub(super) fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = PerLayerEncodedColdLayer::empty(codec, block.shape()[1]);
    tmp.append(block);
    tmp.decode()
}

/// Extract the last row of `h` as a (1, hidden) `Array2`.
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn roundtrip_non_empty_block_decodes_through_codec() {
        // Covers the non-empty path of roundtrip (encode + decode
        // through ColdResidualCodec::Bf16). Bf16 is lossy, so we
        // check shape preservation + values within bf16 tolerance.
        let block = Array2::<f32>::from_shape_fn((3, 4), |(i, j)| 0.1 * (i + j) as f32);
        let out = roundtrip(&block, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[3, 4]);
        for (orig, decoded) in block.iter().zip(out.iter()) {
            // bf16 round-trip precision: ~1% relative for small values
            assert!(
                (orig - decoded).abs() < 0.01,
                "bf16 round-trip drift too large: {orig} → {decoded}"
            );
        }
    }

    #[test]
    fn last_row_extracts_correct_row() {
        let mut h = Array2::<f32>::zeros((3, 4));
        for j in 0..4 {
            h[[2, j]] = (j + 1) as f32;
        }
        let r = last_row(&h);
        assert_eq!(r.shape(), &[1, 4]);
        for j in 0..4 {
            assert_eq!(r[[0, j]], (j + 1) as f32);
        }
    }

    /// Build a minimal `RsStorePerLayer` for tests — no `cold_kv`
    /// (None arm), per-layer codec set to Bf16.
    fn empty_store(num_layers: usize, hidden: usize) -> RsStorePerLayer {
        RsStorePerLayer {
            stored: (0..num_layers)
                .map(|_| Array2::<f32>::zeros((0, hidden)))
                .collect(),
            cold_encoded: None,
            cold_kv: None,
            hot_kv: None,
            cold_abs_start: 0,
            next_position: 0,
            max_window: None,
            policy_codecs: vec![ColdResidualCodec::Bf16; num_layers],
        }
    }

    #[test]
    fn extend_cold_kv_with_overflow_short_circuits_on_empty_overflow() {
        // n_new == 0 → early Some(()) return with no work.
        // Synthetic weights + CPU backend; the function shouldn't
        // touch them.
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let mut rs = empty_store(weights.num_layers, weights.hidden_size);
        let empty_per_layer: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::zeros((0, weights.hidden_size)))
            .collect();
        let result = extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &empty_per_layer,
            0,
        );
        assert!(result.is_some());
        assert!(rs.cold_kv.is_none(), "empty overflow must not init cold_kv");
    }

    #[test]
    fn extend_cold_kv_with_overflow_initialises_cold_kv_on_none_arm() {
        // None arm: rs.cold_kv starts None; non-empty overflow
        // initialises it. Verifies the None → Some(Vec) transition
        // and that recompute_kv runs successfully on each layer.
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let mut rs = empty_store(weights.num_layers, weights.hidden_size);
        let overflow_per_layer: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::from_elem((2, weights.hidden_size), 0.3f32))
            .collect();
        let result = extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &overflow_per_layer,
            0,
        );
        assert!(result.is_some(), "initial extend must succeed");
        let cold_kv = rs
            .cold_kv
            .as_ref()
            .expect("None arm should populate cold_kv");
        assert_eq!(cold_kv.len(), weights.num_layers);
        // Each layer's K should have 2 rows (matching overflow size).
        for (k, _v) in cold_kv {
            assert_eq!(k.shape()[0], 2, "K/V should match overflow row count");
        }
    }

    #[test]
    fn extend_cold_kv_partial_failure_is_atomic() {
        // A per-layer failure mid-extend must not leave `cold_kv` desynced
        // (some layers extended, others not, while `cold_encoded` advanced
        // for all). Contract: either every layer extends, or `cold_kv` is
        // dropped wholesale so the next decode rebuilds from `cold_encoded`.
        let mut weights = make_test_weights();
        assert!(
            weights.num_layers >= 2,
            "test needs ≥2 layers to observe partial extension"
        );
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let mut rs = empty_store(weights.num_layers, weights.hidden_size);

        // Seed cold_kv with 2 rows per layer while all tensors are intact.
        let first_overflow: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::from_elem((2, weights.hidden_size), 0.5f32))
            .collect();
        extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &first_overflow,
            0,
        )
        .expect("initial extend with intact tensors");

        // Break layer 1's K projection: `recompute_kv` returns None for that
        // layer only, so layer 0 would succeed and layer 1 fail.
        let k_key = weights.arch.attn_k_key(1);
        assert!(
            weights.tensors.remove(&k_key).is_some(),
            "fixture must carry the K projection we are removing"
        );

        let second_overflow: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::from_elem((3, weights.hidden_size), 0.7f32))
            .collect();
        let result = extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &second_overflow,
            2,
        );
        assert!(result.is_none(), "the per-layer failure must be reported");
        assert!(
            rs.cold_kv.is_none(),
            "partial failure must drop cold_kv wholesale (no layer desync); \
             the cold_encoded recompute path then covers the next decode"
        );
    }

    #[test]
    fn extend_cold_kv_with_overflow_appends_on_some_arm() {
        // Some arm: rs.cold_kv pre-populated with 2 rows; new
        // overflow of 3 rows. Total should be 5 rows per layer.
        let weights = make_test_weights();
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let mut rs = empty_store(weights.num_layers, weights.hidden_size);

        // Pre-populate cold_kv via a first extend call.
        let first_overflow: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::from_elem((2, weights.hidden_size), 0.5f32))
            .collect();
        extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &first_overflow,
            0,
        )
        .unwrap();

        // Now extend with another 3 rows — exercises the Some arm.
        let second_overflow: Vec<Array2<f32>> = (0..weights.num_layers)
            .map(|_| Array2::<f32>::from_elem((3, weights.hidden_size), 0.7f32))
            .collect();
        let result = extend_cold_kv_with_overflow(
            larql_inference::WeightsView::dense(&weights),
            &CpuBackend,
            &policy,
            &mut rs,
            &second_overflow,
            2, // cold_abs_pos after the first 2 rows
        );
        assert!(result.is_some(), "Some arm extend must succeed");
        let cold_kv = rs.cold_kv.as_ref().unwrap();
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], 5, "K should be 2 (old) + 3 (new) = 5 rows");
            assert_eq!(v.shape()[0], 5, "V should be 2 (old) + 3 (new) = 5 rows");
        }
    }
}
