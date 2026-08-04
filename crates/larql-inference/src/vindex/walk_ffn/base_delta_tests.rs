//! Base+delta path tests — the observation contract of the
//! forward/forward_observed split (2026-07-30 review, items 15+16).
//!
//! Fixture: the Q4K vindex (Gemma-3 arch, GELU-tanh gated) with the
//! feature-major Q4K down sidecar attached, wrapped in a patch overlay
//! that answers `override_slots_at` — every base+delta precondition
//! holds, so the override arm routes `override:base_delta`.

use ndarray::Array2;

use super::observe::Observe;
use crate::ffn::{FfnActivations, FfnBackend};
use crate::test_utils::{attach_down_features_q4k_to_test_vindex, make_test_q4k_weights};
use crate::vindex::WalkFfn;
use larql_vindex::{
    FeatureMeta, FfnRowAccess as _, Fp4FfnAccess, GateLookup, NativeFfnAccess, OverrideSlot,
    PatchOverrides, QuantizedFfnAccess,
};

/// Q4K fixture wrapper with an exhaustive patch enumeration. Delegates
/// all storage access to the inner `VectorIndex`; the patch surface is
/// exactly `slots` (+ the optional gate/down vectors).
struct PatchedQ4kIndex<'a> {
    inner: &'a larql_vindex::VectorIndex,
    slots: Vec<OverrideSlot>,
    gate_ov: Option<(usize, Vec<f32>)>,
    up_ov: Option<(usize, Vec<f32>)>,
    down_ov: Option<(usize, Vec<f32>)>,
}

impl GateLookup for PatchedQ4kIndex<'_> {
    fn gate_knn(
        &self,
        layer: usize,
        residual: &ndarray::Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, residual, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.inner.feature_meta(layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }
}

impl PatchOverrides for PatchedQ4kIndex<'_> {
    fn has_overrides_at(&self, _layer: usize) -> bool {
        !self.slots.is_empty()
    }
    fn gate_override(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
        self.gate_ov
            .as_ref()
            .filter(|(f, _)| *f == feature)
            .map(|(_, v)| v.as_slice())
    }
    fn up_override(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
        self.up_ov
            .as_ref()
            .filter(|(f, _)| *f == feature)
            .map(|(_, v)| v.as_slice())
    }
    fn down_override(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_ov
            .as_ref()
            .filter(|(f, _)| *f == feature)
            .map(|(_, v)| v.as_slice())
    }
    fn override_slots_at(&self, _layer: usize) -> Option<Vec<OverrideSlot>> {
        Some(self.slots.clone())
    }
}

impl NativeFfnAccess for PatchedQ4kIndex<'_> {
    fn has_down_features(&self) -> bool {
        self.inner.has_down_features()
    }
    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.up_layer_matrix(layer)
    }
    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.down_layer_matrix(layer)
    }
    fn has_interleaved(&self) -> bool {
        self.inner.has_interleaved()
    }
    fn has_full_mmap_ffn(&self) -> bool {
        self.inner.has_full_mmap_ffn()
    }
}
impl Fp4FfnAccess for PatchedQ4kIndex<'_> {}

impl QuantizedFfnAccess for PatchedQ4kIndex<'_> {
    fn has_interleaved_kquant(&self) -> bool {
        self.inner.has_interleaved_kquant()
    }
    fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        self.inner.interleaved_kquant_layer_data(layer)
    }
    fn has_down_features_kquant(&self) -> bool {
        self.inner.has_down_features_kquant()
    }
    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(&[u8], &str, usize)> {
        self.inner.down_features_q4k_layer_data(layer)
    }
    fn kquant_ffn_layer(&self, layer: usize, component: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        self.inner.kquant_ffn_layer(layer, component)
    }
    fn kquant_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        self.inner.kquant_ffn_row_dot(layer, component, feat, x)
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner
            .kquant_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out)
    }
}

fn x(seq: usize, hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (seq, hidden),
        (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
    )
    .unwrap()
}

/// The fixture with its sidecar; the unpatched baseline runs through the
/// SAME wrapper shape (empty slots) so patched-vs-base comparisons see
/// identical kernels and bytes.
fn fixture() -> (larql_models::ModelWeights, larql_vindex::VectorIndex) {
    let weights = make_test_q4k_weights();
    let mut index = crate::test_utils::make_test_q4k_vindex(&weights);
    attach_down_features_q4k_to_test_vindex(&weights, &mut index);
    (weights, index)
}

fn unpatched<'a>(inner: &'a larql_vindex::VectorIndex) -> PatchedQ4kIndex<'a> {
    PatchedQ4kIndex {
        inner,
        slots: Vec::new(),
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    }
}

/// Record-mode forward through `WalkFfn`, returning (out, dense acts,
/// dispatch trace paths).
fn run_observed(
    weights: &larql_models::ModelWeights,
    index: &PatchedQ4kIndex<'_>,
    input: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>, Vec<&'static str>) {
    let ffn = WalkFfn::new_unlimited(weights, index).with_dispatch_trace();
    let (out, obs) = ffn.forward_observed(0, input);
    let paths = ffn.take_dispatch_trace().iter().map(|e| e.path).collect();
    let act = obs.into_dense().expect("base_delta observes densely");
    (out, act, paths)
}

fn assert_close(a: f32, b: f32, what: &str) {
    assert!(
        (a - b).abs() <= 1e-3 * b.abs().max(1.0),
        "{what}: {a} vs {b}"
    );
}

/// Down-only override: activations are untouched (the slot's a_new
/// equals its base activation), the output shifts by exactly
/// `a · (d_new − d_old)`, and Skip/Record produce identical outputs.
#[test]
fn base_delta_down_override_observed_activations_match_base() {
    let (weights, inner) = fixture();
    let hidden = weights.hidden_size;
    let feat = 1usize;
    let input = x(1, hidden);

    let (out_base, act_base, _) = run_observed(&weights, &unpatched(&inner), &input);

    let custom_down = vec![0.5f32; hidden];
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: false,
        }],
        gate_ov: None,
        up_ov: None,
        down_ov: Some((feat, custom_down.clone())),
    };
    let (out_p, act_p, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(
        paths.last(),
        Some(&"override:base_delta"),
        "all preconditions hold — the exact path must fire, got {paths:?}"
    );

    // Observation: a down-only override does not change any activation.
    assert_eq!(act_p.shape(), act_base.shape());
    for f in 0..act_base.shape()[1] {
        assert_close(act_p[[0, f]], act_base[[0, f]], "activation");
    }

    // Output: base + a · (d_new − d_old), with a and d_old read through
    // the same fused kernels / dequant cache the path itself used.
    let xs = input.row(0).to_owned().to_vec();
    let g = inner.ffn_row_dot(0, 0, feat, &xs).expect("gate row dot");
    let u = inner.ffn_row_dot(0, 1, feat, &xs).expect("up row dot");
    let a = crate::ffn::gelu_tanh(g) * u; // Gemma-3 → GELU-tanh
    assert!(a.abs() > 0.0, "fixture must activate the patched slot");
    let down_cache = inner.kquant_ffn_layer(0, 2).expect("down dequant cache");
    for h in 0..hidden {
        let expected = out_base[[0, h]] + a * (custom_down[h] - down_cache[feat * hidden + h]);
        assert_close(out_p[[0, h]], expected, "output");
    }

    // Skip/Record parity: both modes run the same routed body.
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    assert_eq!(ffn.forward(0, &input), out_p);
}

/// Gate override: the observation must report the POST-patch slot
/// activation φ(g_new·x)·(u_old·x), not the base value.
#[test]
fn base_delta_gate_override_observed_activation_is_post_patch() {
    let (weights, inner) = fixture();
    let hidden = weights.hidden_size;
    let feat = 2usize;
    let input = x(1, hidden);

    let (_, act_base, _) = run_observed(&weights, &unpatched(&inner), &input);

    let custom_gate: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: false,
        }],
        gate_ov: Some((feat, custom_gate.clone())),
        up_ov: None,
        down_ov: None,
    };
    let (_, act_p, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(paths.last(), Some(&"override:base_delta"));

    let xs = input.row(0).to_owned().to_vec();
    let g_new: f32 = custom_gate.iter().zip(&xs).map(|(a, b)| a * b).sum();
    let u_old = inner.ffn_row_dot(0, 1, feat, &xs).expect("up row dot");
    let expected = crate::ffn::gelu_tanh(g_new) * u_old;
    assert_close(act_p[[0, feat]], expected, "post-patch slot activation");
    assert!(
        (act_p[[0, feat]] - act_base[[0, feat]]).abs() > 0.0,
        "gate override must change the observed slot activation"
    );
    // Non-patched slots keep their base activations.
    for f in 0..act_base.shape()[1] {
        if f != feat {
            assert_close(act_p[[0, f]], act_base[[0, f]], "unpatched activation");
        }
    }
}

/// Tombstoned slot: observed activation is exactly 0 and the output
/// loses exactly the base contribution `a_old · d_old`. Skip mode
/// (`forward`) produces the identical output without any observation.
#[test]
fn base_delta_tombstone_observed_activation_is_zero() {
    let (weights, inner) = fixture();
    let hidden = weights.hidden_size;
    let feat = 3usize;
    let input = x(1, hidden);

    let (out_base, _, _) = run_observed(&weights, &unpatched(&inner), &input);

    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: true,
        }],
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let (out_p, act_p, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(paths.last(), Some(&"override:base_delta"));
    assert_eq!(
        act_p[[0, feat]],
        0.0,
        "tombstoned slot contributes nothing post-patch"
    );

    let xs = input.row(0).to_owned().to_vec();
    let g = inner.ffn_row_dot(0, 0, feat, &xs).expect("gate row dot");
    let u = inner.ffn_row_dot(0, 1, feat, &xs).expect("up row dot");
    let a_old = crate::ffn::gelu_tanh(g) * u;
    let down_cache = inner.kquant_ffn_layer(0, 2).expect("down dequant cache");
    for h in 0..hidden {
        let expected = out_base[[0, h]] - a_old * down_cache[feat * hidden + h];
        assert_close(out_p[[0, h]], expected, "tombstone output");
    }

    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    assert_eq!(ffn.forward(0, &input), out_p);
}

/// Decline branches of `base_delta_preconditions` /
/// `walk_ffn_base_delta` — each precondition failure must return
/// `None` (the caller then takes the sparse/fallback override
/// routes), never a wrong "exact" result.
#[test]
fn base_delta_declines_when_preconditions_fail() {
    let (weights, inner) = fixture();
    let hidden = weights.hidden_size;
    let input = x(1, hidden);
    let slot = |feat| {
        vec![OverrideSlot {
            feature: feat,
            tombstoned: false,
        }]
    };
    let patched = |slots| PatchedQ4kIndex {
        inner: &inner,
        slots,
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };

    // Non-gated arch: the φ(g·x)(u·x) decomposition doesn't apply.
    let starcoder = crate::test_utils::make_starcoder2_test_weights();
    let idx = patched(slot(0));
    let ffn = WalkFfn::new_unlimited(&starcoder, &idx);
    assert!(ffn.walk_ffn_base_delta(0, &input, Observe::Skip).is_none());

    // Explicit sparse K: no dense base exists to correct.
    let idx = patched(slot(0));
    let cfg = crate::vindex::WalkFfnConfig::sparse(weights.num_layers, 4);
    let ffn = WalkFfn::from_config(&weights, &idx, cfg);
    assert!(ffn.walk_ffn_base_delta(0, &input, Observe::Skip).is_none());

    // Wrong hidden width: decline before touching any storage.
    let idx = patched(slot(0));
    let ffn = WalkFfn::new_unlimited(&weights, &idx);
    let bad = x(1, hidden + 1);
    assert!(ffn.walk_ffn_base_delta(0, &bad, Observe::Skip).is_none());

    // Wrong-width gate override on an enumerated slot: decline
    // mid-computation rather than dot a mismatched vector.
    let bad_gate = PatchedQ4kIndex {
        inner: &inner,
        slots: slot(0),
        gate_ov: Some((0, vec![1.0; hidden + 1])),
        up_ov: None,
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &bad_gate);
    assert!(ffn
        .walk_ffn_base_delta(0, &input, Observe::Record)
        .is_none());

    // Wrong-width up and down overrides: same decline contract.
    let bad_up = PatchedQ4kIndex {
        inner: &inner,
        slots: slot(0),
        gate_ov: None,
        up_ov: Some((0, vec![1.0; hidden + 1])),
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &bad_up);
    assert!(ffn
        .walk_ffn_base_delta(0, &input, Observe::Record)
        .is_none());
    let bad_down = PatchedQ4kIndex {
        inner: &inner,
        slots: slot(0),
        gate_ov: None,
        up_ov: None,
        down_ov: Some((0, vec![1.0; hidden + 1])),
    };
    let ffn = WalkFfn::new_unlimited(&weights, &bad_down);
    assert!(ffn
        .walk_ffn_base_delta(0, &input, Observe::Record)
        .is_none());
}

/// Storage-coherence declines: a feature-major f32 up sidecar would
/// hijack the up row-dot away from the Q4K bytes the base consumed
/// (condition 1), and `has_down_features()` diverts the ladder itself.
#[test]
fn base_delta_declines_on_storage_conflicts() {
    let weights = make_test_q4k_weights();
    let input = x(1, weights.hidden_size);
    let slots = vec![OverrideSlot {
        feature: 0,
        tombstoned: false,
    }];

    // f32 up sidecar hijack.
    let mut with_up = crate::test_utils::make_test_q4k_vindex(&weights);
    attach_down_features_q4k_to_test_vindex(&weights, &mut with_up);
    crate::test_utils::attach_up_features_f32_only_to_test_vindex(&weights, &mut with_up);
    assert!(with_up.up_layer_matrix(0).is_some());
    let patched = PatchedQ4kIndex {
        inner: &with_up,
        slots: slots.clone(),
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    assert!(ffn.walk_ffn_base_delta(0, &input, Observe::Skip).is_none());

    // Feature-major f32 down storage → whole-storage conflict decline.
    let mut with_down = crate::test_utils::make_test_q4k_vindex(&weights);
    crate::test_utils::attach_down_features_f32_only_to_test_vindex(&weights, &mut with_down);
    assert!(with_down.has_down_features());
    let patched = PatchedQ4kIndex {
        inner: &with_down,
        slots,
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    assert!(ffn.walk_ffn_base_delta(0, &input, Observe::Skip).is_none());
}

/// A valid up override feeds the NEW term through the override vector
/// (condition: a_new = φ(g_old)·(u_new·x)); non-contiguous inputs take
/// the owned-slice fallback and must match the contiguous result.
#[test]
fn base_delta_up_override_and_non_contiguous_input() {
    let (weights, inner) = fixture();
    let hidden = weights.hidden_size;
    let feat = 1usize;
    let input = x(1, hidden);
    let custom_up: Vec<f32> = (0..hidden).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: false,
        }],
        gate_ov: None,
        up_ov: Some((feat, custom_up.clone())),
        down_ov: None,
    };
    let (_out_c, act_c, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(paths.last(), Some(&"override:base_delta"));
    let xs = input.row(0).to_owned().to_vec();
    let g_old = inner.ffn_row_dot(0, 0, feat, &xs).expect("gate row dot");
    let u_new: f32 = custom_up.iter().zip(&xs).map(|(a, b)| a * b).sum();
    let expected = crate::ffn::gelu_tanh(g_old) * u_new;
    assert_close(act_c[[0, feat]], expected, "up-override slot activation");

    // Column-major (non-contiguous rows) input, same values over two
    // positions — a `[h, 2]` buffer transposed to `[2, h]` has strided
    // rows, forcing the owned-slice fallback in the per-position loop.
    let x2 = x(2, hidden);
    let mut t = Vec::with_capacity(2 * hidden);
    for h in 0..hidden {
        for s in 0..2 {
            t.push(x2[[s, h]]);
        }
    }
    let x_cm = Array2::from_shape_vec((hidden, 2), t)
        .unwrap()
        .reversed_axes();
    assert_eq!(x_cm, x2);
    assert!(x_cm.row(0).as_slice().is_none(), "must be non-contiguous");
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    let (out_std, _) = ffn
        .walk_ffn_base_delta(0, &x2, Observe::Skip)
        .expect("preconditions hold");
    let (out_cm, _) = ffn
        .walk_ffn_base_delta(0, &x_cm, Observe::Skip)
        .expect("preconditions hold");
    // The dense-base ladder may pick a different (equivalent) kernel for
    // strided input — Q4K-native requires contiguous rows and declines
    // to the dequant path — so parity is float-tight, not bitwise.
    assert_eq!(out_cm.shape(), out_std.shape());
    for (a, b) in out_cm.iter().zip(out_std.iter()) {
        assert_close(*a, *b, "owned-slice fallback vs contiguous");
    }
}

/// Without the feature-major Q4K down sidecar the old-down subtraction
/// has no gatherable rows — exactness condition 2 fails and the path
/// declines (the ladder then serves the layer via the sparse walk).
#[test]
fn base_delta_declines_without_down_sidecar() {
    let weights = make_test_q4k_weights();
    let inner = crate::test_utils::make_test_q4k_vindex(&weights); // no sidecar
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: 0,
            tombstoned: false,
        }],
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    assert!(ffn
        .walk_ffn_base_delta(0, &x(1, weights.hidden_size), Observe::Skip)
        .is_none());
}

/// A slot with no base rows at all (feature beyond the layer) is a pure
/// insert with no override vectors: nothing to subtract, φ(0)·0 = 0 to
/// add — the output must equal the unpatched base exactly.
#[test]
fn base_delta_pure_insert_slot_without_vectors_is_identity() {
    let (weights, inner) = fixture();
    let input = x(1, weights.hidden_size);
    let (out_base, _, _) = run_observed(&weights, &unpatched(&inner), &input);
    let ghost = inner.num_features(0); // one past the last real feature
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: ghost,
            tombstoned: false,
        }],
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let (out_p, _, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(paths.last(), Some(&"override:base_delta"));
    assert_eq!(out_p, out_base, "a vectorless ghost slot changes nothing");
}

/// SiLU arch arm of φ (the Q4K fixture's SiLU twin): the observed slot
/// activation for a gate override is `silu(g_new)·u_old`.
#[test]
fn base_delta_silu_arch_uses_silu_activation() {
    let weights = crate::test_utils::make_test_q4k_weights_silu();
    let mut inner = crate::test_utils::make_test_q4k_vindex(&weights);
    attach_down_features_q4k_to_test_vindex(&weights, &mut inner);
    let hidden = weights.hidden_size;
    let feat = 1usize;
    let input = x(1, hidden);
    let custom_gate: Vec<f32> = (0..hidden).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: false,
        }],
        gate_ov: Some((feat, custom_gate.clone())),
        up_ov: None,
        down_ov: None,
    };
    let (_, act_p, paths) = run_observed(&weights, &patched, &input);
    assert_eq!(paths.last(), Some(&"override:base_delta"));
    let xs = input.row(0).to_owned().to_vec();
    let g_new: f32 = custom_gate.iter().zip(&xs).map(|(a, b)| a * b).sum();
    let u_old = inner.ffn_row_dot(0, 1, feat, &xs).expect("up row dot");
    let expected = g_new * crate::ffn::sigmoid(g_new) * u_old;
    assert_close(act_p[[0, feat]], expected, "SiLU post-patch activation");
}

/// Skip-mode base_delta must return `None` observation internally (the
/// seam: `walk_ffn_base_delta(.., Observe::Skip)`) while still producing
/// the corrected output — pins the mode plumbing itself.
#[test]
fn base_delta_skip_mode_returns_no_observation() {
    let (weights, inner) = fixture();
    let feat = 1usize;
    let input = x(1, weights.hidden_size);
    let patched = PatchedQ4kIndex {
        inner: &inner,
        slots: vec![OverrideSlot {
            feature: feat,
            tombstoned: true,
        }],
        gate_ov: None,
        up_ov: None,
        down_ov: None,
    };
    let ffn = WalkFfn::new_unlimited(&weights, &patched);
    let (out, obs) = ffn
        .walk_ffn_base_delta(0, &input, Observe::Skip)
        .expect("preconditions hold");
    assert!(obs.is_none(), "Skip mode must not build an observation");
    let (out_rec, obs_rec) = ffn
        .walk_ffn_base_delta(0, &input, Observe::Record)
        .expect("preconditions hold");
    assert_eq!(out, out_rec);
    assert!(matches!(obs_rec, Some(FfnActivations::Dense(_))));
}
