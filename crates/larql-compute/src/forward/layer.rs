//! Layer dispatch — runs attention + FFN + PLE + layer_scalar for a single layer.
//!
//! Orchestrates the per-layer computation: attention (with optional KV sharing),
//! FFN, per-layer embeddings, and layer scalar multiplication.

use super::apply_norm;
use super::hooks::LayerHook;
use super::ple::apply_per_layer_embedding;
use crate::attention::{AttentionWeights, SharedKV};
use crate::ffn::FfnBackend;
use crate::residual::rms_norm_for_arch;
use larql_models::{ModelWeights, WeightsView};
use ndarray::Array2;

// No alloc_prelude import: this file's only `Vec` uses (the two stage-dump
// closures' `bytes: Vec<u8>` collect) are both inside
// `#[cfg(not(target_arch = "wasm32"))]` blocks -- nothing in the portable
// subset needs Vec/String/Box/ToOwned.

/// Public wrapper for run_attention — used by diagnostic/capture tooling.
pub fn run_attention_public(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    run_attention(weights, h, layer)
}

/// Run attention for a single layer. Returns the post-attention residual.
pub fn run_attention(weights: WeightsView, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    let (h_post_attn, _) = run_attention_inner(weights, h, layer, false, None)?;
    Some(h_post_attn)
}

/// Run attention with optional per-head weight capture and shared K/V.
pub fn run_attention_inner(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let (h_post_attn, _attn_projected, attn_weights) =
        crate::attention::run_attention_block_shared(
            weights,
            h,
            layer,
            capture_attention,
            shared_kv,
        )?;
    Some((h_post_attn, attn_weights))
}

/// Run attention returning post-processed K/V for caching (KV sharing source layers).
pub fn run_attention_with_kv_cache(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    let (h_post_attn, _, _, k_rope, v_final) =
        crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
    Some((h_post_attn, (k_rope, v_final)))
}

/// Run FFN for a single layer using the given backend. Returns the post-FFN residual.
pub fn run_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> (Array2<f32>, Option<Array2<f32>>) {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    // Layer-0 (or LARQL_STAGE_DUMP_LAYER) stage dumps — matches the Metal
    // `LARQL_METAL_DUMP_LAYERS` convention. Lets us diff per-stage
    // intermediates between CPU and Metal.
    let dump_cfg = super::dump_config::DumpConfig::get();
    let _stage_dump_dir = dump_cfg.stage_dir(layer);
    // std::fs has no core/alloc equivalent; wasm32v1-none has no filesystem
    // and stage_dump_dir is unconditionally None there, so the whole dump
    // path is dead code on that target (see attention/block.rs's identical
    // dump_f32 pattern for the fuller rationale).
    #[cfg(not(target_arch = "wasm32"))]
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = stage_dump_dir {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(super::dump_config::cpu_stage_path(dir, name), &bytes);
        }
    };
    #[cfg(target_arch = "wasm32")]
    let dump_f32 = |_name: &str, _arr: &Array2<f32>| {};
    dump_f32("h_post_attn", h_post_attn);

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm_for_arch(h_post_attn, None, norm_offset, &*weights.arch),
    };
    dump_f32("ffn_norm_out", &h_ffn);

    // Observation is opt-in: the uncaptured arm runs the hot `forward`,
    // which never allocates an activation buffer. The captured arm
    // densifies whatever the executed path honestly observed; an Absent
    // observation (cache hit, remote, unobserving backend) yields `None`
    // rather than fabricated zeros.
    let (ffn_out, activation) = if capture_activation {
        let (out, obs) = ffn.forward_observed(layer, &h_ffn);
        (out, obs.into_dense())
    } else {
        (ffn.forward(layer, &h_ffn), None)
    };
    dump_f32("ffn_out_raw", &ffn_out);

    let res_mult = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm_for_arch(&ffn_out, None, norm_offset, &*weights.arch),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    };

    (h_out, activation)
}

/// Apply per-layer scalar multiplier if present (e.g., Gemma 4 layer_scalar).
///
/// Skip when the scalar is 0.0 (absent / unloaded — multiplying would zero the
/// layer output, collapsing generation) or 1.0 (identity). Matches the Metal
/// `apply_whole_layer_scalar` in `metal/decode/moe_combine.rs:88-94` so the
/// CPU MoE path produces the same residual as the GPU path.
pub fn apply_layer_scalar(weights: &ModelWeights, h: &mut Array2<f32>, layer: usize) {
    if let Some(key) = weights.arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&key) {
            if let Some(&scalar) = scalars.first() {
                if scalar != 0.0 && scalar != 1.0 {
                    *h *= scalar;
                }
            }
        }
    }
}

/// Run a single transformer layer with the given FFN backend.
///
/// Handles: attention → FFN → per-layer embedding → layer_scalar.
/// All four steps are needed for Gemma 4 correctness. Exposed `pub` so
/// alternate forward drivers (notably `vindex::predict_kquant`) get the same
/// sequence as `predict_with_temperature` without duplicating logic.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_ffn(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = if shared_kv.is_some() {
        (
            run_attention_inner(weights, h, layer, false, shared_kv)?.0,
            None,
        )
    } else {
        let (h_pa, kv) = run_attention_with_kv_cache(weights, h, layer)?;
        (h_pa, Some(kv))
    };
    // Diagnostic: per-layer `h_post_attn` dump, paired with Metal's
    // `metal_layer_{LL}_h_post_attn.f32`. Lets the `residual_diff` tool
    // bisect any layer's drift into attention (compare h_post_attn) vs
    // FFN+PLE+scalar (compare h_out minus h_post_attn). Gated on the
    // same env var as the end-of-layer dump; no overhead when unset.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
        let _ = std::fs::write(&path, &bytes);
    }
    let (h_post_ffn, activation) = run_ffn(&weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(&weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(&weights, &mut h_out, layer);
    Some((h_out, activation, kv_out))
}

/// Run a single transformer layer, optionally capturing attention weights.
///
/// Backwards-compatible wrapper: behaves identically to the pre-hook version
/// by passing a [`super::hooks::NoopHook`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_layer_with_capture(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(
    Array2<f32>,
    Option<Array2<f32>>,
    Option<AttentionWeights>,
    Option<SharedKV>,
)> {
    run_layer_with_capture_hooked(
        weights,
        h,
        layer,
        ffn,
        capture_activation,
        capture_attention,
        ple_input,
        shared_kv,
        &mut super::hooks::NoopHook,
    )
}

/// Hook-aware sibling of [`run_layer_with_capture`]. Fires the [`LayerHook`]
/// callbacks at four points inside the layer: pre-layer, post-attention
/// (mut), attention-weights / FFN-activation if captured, post-layer (mut).
///
/// The two `&mut` callbacks (post-attention and post-layer) are what enable
/// activation patching, ablation, and steering.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_layer_with_capture_hooked(
    weights: WeightsView,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    hook: &mut dyn LayerHook,
) -> Option<(
    Array2<f32>,
    Option<Array2<f32>>,
    Option<AttentionWeights>,
    Option<SharedKV>,
)> {
    hook.on_pre_layer(layer, h);

    let (mut h_post_attn, attn_weights, kv_out) = if shared_kv.is_some() {
        let (h_post_attn, attn_weights) =
            run_attention_inner(weights, h, layer, capture_attention, shared_kv)?;
        (h_post_attn, attn_weights, None)
    } else {
        let (h_post_attn, _, attn_weights, k_rope, v_final) =
            crate::attention::run_attention_block_with_kv_out(
                weights,
                h,
                layer,
                capture_attention,
                None,
            )?;
        (h_post_attn, attn_weights, Some((k_rope, v_final)))
    };
    if let Some(ref w) = attn_weights {
        hook.on_attention_weights(layer, w);
    }
    hook.on_post_attention(layer, &mut h_post_attn);

    let (h_post_ffn, activation) = run_ffn(&weights, &h_post_attn, layer, ffn, capture_activation);
    if let Some(ref act) = activation {
        hook.on_ffn_activation(layer, act);
    }

    let mut h_out = apply_per_layer_embedding(&weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(&weights, &mut h_out, layer);
    hook.on_post_layer(layer, &mut h_out);

    Some((h_out, activation, attn_weights, kv_out))
}

#[cfg(test)]
mod tests {
    //! Direct unit tests use a stub `FfnBackend` impl (no inference dep,
    //! no dev-dep cycle). Integration tests with the real `WeightFfn`
    //! impl live in `larql-inference`'s shim file, where the impl is
    //! reachable without a dev-dep cycle.

    use super::*;
    use crate::ffn::FfnBackend;
    use larql_models::test_fixtures::make_test_weights;
    use ndarray::Array2;

    struct StubFfn<'a> {
        weights: &'a larql_models::ModelWeights,
    }
    impl FfnBackend for StubFfn<'_> {
        fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
            // Zero-impulse FFN: returns the input unchanged so the
            // layer dispatcher's plumbing (attention → norm → ffn →
            // residual) can be exercised in isolation.
            x.clone()
        }
        fn forward_observed(
            &self,
            layer: usize,
            x: &Array2<f32>,
        ) -> (Array2<f32>, crate::ffn::FfnActivations) {
            (
                self.forward(layer, x),
                crate::ffn::FfnActivations::Dense(Array2::zeros((
                    x.shape()[0],
                    self.weights.intermediate_size,
                ))),
            )
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn run_layer_with_ffn_stub_returns_finite() {
        let weights = make_test_weights();
        let ffn = StubFfn { weights: &weights };
        let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let result =
            run_layer_with_ffn(WeightsView::dense(&weights), &h, 0, &ffn, false, None, None);
        assert!(result.is_some(), "run_layer_with_ffn returned None");
        let (h_out, _act, _kv) = result.unwrap();
        assert_eq!(h_out.shape(), h.shape());
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_attention_public_returns_post_attention_residual() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let h_post = run_attention_public(WeightsView::dense(&weights), &h, 0)
            .expect("attention should return post-residual on standard weights");
        assert_eq!(h_post.shape(), h.shape());
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_attention_returns_same_shape_as_input() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.1f32);
        let h_post =
            run_attention(WeightsView::dense(&weights), &h, 0).expect("attention should succeed");
        assert_eq!(h_post.shape(), h.shape());
    }

    #[test]
    fn run_attention_inner_with_capture_returns_attention_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (h_post, attn_w) = run_attention_inner(
            WeightsView::dense(&weights),
            &h,
            0,
            /*capture_attention=*/ true,
            None,
        )
        .expect("inner attention");
        assert_eq!(h_post.shape(), h.shape());
        assert!(
            attn_w.is_some(),
            "capture_attention=true should return weights"
        );
    }

    #[test]
    fn run_attention_inner_without_capture_drops_attention_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (h_post, attn_w) = run_attention_inner(
            WeightsView::dense(&weights),
            &h,
            0,
            /*capture_attention=*/ false,
            None,
        )
        .expect("inner attention");
        assert_eq!(h_post.shape(), h.shape());
        assert!(
            attn_w.is_none(),
            "capture_attention=false should not return weights"
        );
    }

    #[test]
    fn run_attention_with_kv_cache_returns_kv_pair() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.1f32);
        let (h_post, (k, v)) = run_attention_with_kv_cache(WeightsView::dense(&weights), &h, 0)
            .expect("kv cache attention");
        assert_eq!(h_post.shape(), h.shape());
        // K and V have the same shape (seq_len × kv_dim).
        assert_eq!(k.shape(), v.shape());
        assert_eq!(k.shape()[0], 3);
    }

    #[test]
    fn run_layer_with_capture_hooked_invokes_every_hook_point() {
        use crate::forward::hooks::RecordHook;
        let weights = make_test_weights();
        let ffn = StubFfn { weights: &weights };
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let mut record = RecordHook::for_layers([0]);
        let result = run_layer_with_capture_hooked(
            WeightsView::dense(&weights),
            &h,
            /*layer=*/ 0,
            &ffn,
            /*capture_activation=*/ true,
            /*capture_attention=*/ true,
            /*ple_input=*/ None,
            /*shared_kv=*/ None,
            &mut record,
        );
        let (h_out, act, attn_w, kv_out) = result.expect("hooked layer succeeds");
        assert_eq!(h_out.shape(), h.shape());
        assert!(
            act.is_some(),
            "capture_activation=true should populate activation"
        );
        assert!(
            attn_w.is_some(),
            "capture_attention=true should populate weights"
        );
        assert!(kv_out.is_some(), "shared_kv=None forces fresh K/V path");
        // Hook recorded every callback.
        assert!(record.pre_layer.contains_key(&0));
        assert!(record.post_attention.contains_key(&0));
        assert!(record.attention_weights.contains_key(&0));
        assert!(record.ffn_activation.contains_key(&0));
        assert!(record.post_layer.contains_key(&0));
    }

    #[test]
    fn run_layer_with_capture_hooked_uses_shared_kv_branch() {
        let weights = make_test_weights();
        let ffn = StubFfn { weights: &weights };
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        // Run once to build a SharedKV.
        let (_, fresh_kv) =
            run_attention_with_kv_cache(WeightsView::dense(&weights), &h, 0).unwrap();
        let mut hook = crate::forward::NoopHook;
        let result = run_layer_with_capture_hooked(
            WeightsView::dense(&weights),
            &h,
            0,
            &ffn,
            false,
            false,
            None,
            Some(&fresh_kv),
            &mut hook,
        );
        let (_, _, _, kv_out) = result.expect("shared-kv layer succeeds");
        assert!(
            kv_out.is_none(),
            "shared_kv branch must not return fresh K/V"
        );
    }

    #[test]
    fn run_layer_with_capture_no_hook_wrapper_matches_hooked() {
        let weights = make_test_weights();
        let ffn = StubFfn { weights: &weights };
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let result = run_layer_with_capture(
            WeightsView::dense(&weights),
            &h,
            0,
            &ffn,
            true,
            true,
            None,
            None,
        );
        let (h_out, act, attn_w, kv_out) = result.expect("non-hooked capture wrapper");
        assert_eq!(h_out.shape(), h.shape());
        assert!(act.is_some());
        assert!(attn_w.is_some());
        assert!(kv_out.is_some());
    }

    #[test]
    fn run_layer_with_ffn_stub_advances_residual() {
        // The layer must transform the residual — output should NOT be
        // bit-identical to input even though our stub FFN is identity.
        // Attention + norm + residual_add change the values.
        let weights = make_test_weights();
        let ffn = StubFfn { weights: &weights };
        let h = Array2::from_elem((3, weights.hidden_size), 1.0f32);
        let (h_out, _, _) =
            run_layer_with_ffn(WeightsView::dense(&weights), &h, 0, &ffn, false, None, None)
                .unwrap();
        let differ = h
            .iter()
            .zip(h_out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "layer should transform the residual even with stub FFN"
        );
    }
}
