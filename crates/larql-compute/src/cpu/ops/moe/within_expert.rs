//! Within-expert feature routing (MoE-within-expert V1 aim-validation probe).
//!
//! Research instrument, **OFF by default**. When a [`WithinExpertRouting`] is
//! installed via [`set_routing`], each selected expert's gated FFN keeps only a
//! fraction of its `inter` intermediate features (the post-activation values
//! flowing into the `down` projection) for the layer set by
//! [`set_current_layer`], zeroing the rest before `down`.
//!
//! This is the MoE analogue of V1's per-layer FFN hash routing
//! (`examples/walk_ffn_v1_hash_routing.rs`). V1 falsified hash routing *within a
//! dense FFN*; on the 26B-A4B that harness measures the wrong object because the
//! per-layer block is 128 stacked experts, not one dense FFN. This module lets
//! the V1 three-phase protocol run on a single expert's own `inter`-feature
//! space instead, judged downstream in predictive units (KL / NLL), exactly as
//! V1 demanded (`feedback_metric_matches_operation`).
//!
//! **Parity**: with no routing installed, [`prune_act`] is a single relaxed
//! atomic load and returns immediately — byte-identical to the un-instrumented
//! path. The forward driver's [`set_current_layer`] call is one relaxed atomic
//! store per MoE layer. See `examples/walk_ffn_v1_moe_within_expert.rs`.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// core::sync::atomic is the same module std re-exports -- portable
// regardless of target, no cfg needed. RwLock/Arc<...>-backed ROUTING
// storage has no core/alloc equivalent (needs real synchronization);
// set_routing/set_current_layer/prune_act's real bodies are native-only.
// Their only in-scope caller for `cargo build` is prune_act (from
// cpu/ops/moe/{forward,expert/q4k}.rs); set_routing/set_current_layer
// are only ever called from an example binary (examples/
// walk_ffn_v1_moe_within_expert.rs, out of scope for a library build),
// so on wasm32 they're a same-signature no-op that leaves ACTIVE
// permanently false -- the exact state this "off by default" research
// instrument already starts in, matching the doc comment's own
// definition of "byte-identical to the un-instrumented path."
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicUsize;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, RwLock};

/// How the kept features are chosen inside each expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFeatureSelector {
    /// Oracle ceiling: keep the `k` features with the largest `|act|` (the
    /// post-activation magnitude entering `down`). Needs the full gate+up
    /// projection — the accuracy upper bound for any size-`k` route, and the
    /// direct analogue of V1's gate-oracle selection.
    ActMagnitude,
    /// Content-blind cheap route: keep `~k` features on a fixed stride. The
    /// realizability lower bound (Phase C) — needs no activation values, so it
    /// is the cheapest possible route. If even the oracle fails, this is moot.
    Strided,
}

/// Per-layer within-expert keep schedule.
///
/// `frac_per_layer[l] = Some(f)` keeps `~round(f * inter)` features for **every**
/// expert routed in layer `l`; `None` (or `f >= 1.0`) keeps all (dense expert).
/// Fractions rather than absolute `k` so the schedule is independent of the
/// expert intermediate size — the kernel knows `inter` and converts.
#[derive(Clone, Debug)]
pub struct WithinExpertRouting {
    pub frac_per_layer: Vec<Option<f32>>,
    pub selector: ExpertFeatureSelector,
}

impl WithinExpertRouting {
    /// All-dense schedule (no pruning anywhere) of length `num_layers`.
    pub fn dense(num_layers: usize) -> Self {
        Self {
            frac_per_layer: vec![None; num_layers],
            selector: ExpertFeatureSelector::ActMagnitude,
        }
    }

    /// Resolve the keep-count for `inter` features at the current layer, or
    /// `None` if this layer keeps all features. Pure helper (testable without
    /// touching the globals).
    pub fn keep_k(&self, layer: usize, inter: usize) -> Option<usize> {
        let frac = self.frac_per_layer.get(layer).copied().flatten()?;
        if frac >= 1.0 || inter == 0 {
            return None;
        }
        let k = ((frac * inter as f32).round() as usize).clamp(1, inter);
        if k >= inter {
            None
        } else {
            Some(k)
        }
    }
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(not(target_arch = "wasm32"))]
static CURRENT_LAYER: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ROUTING: RwLock<Option<Arc<WithinExpertRouting>>> = RwLock::new(None);

/// Install (or replace) the within-expert routing schedule. `None` disables it
/// and restores byte-exact parity with the un-instrumented path. Flips the
/// fast-path [`ACTIVE`] flag accordingly.
#[cfg(target_arch = "wasm32")]
pub fn set_routing(_routing: Option<WithinExpertRouting>) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn set_routing(routing: Option<WithinExpertRouting>) {
    let active = routing.is_some();
    {
        let mut guard = ROUTING.write().unwrap_or_else(|p| p.into_inner());
        *guard = routing.map(Arc::new);
    }
    ACTIVE.store(active, Ordering::Relaxed);
}

/// Set the layer whose `frac_per_layer` entry applies to subsequent expert
/// calls. Called once per MoE layer by the forward driver before the
/// per-position expert loop (layers run sequentially, so a single atomic store
/// is sufficient — all of a layer's expert workers read the same value).
#[cfg(target_arch = "wasm32")]
pub fn set_current_layer(_layer: usize) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn set_current_layer(layer: usize) {
    CURRENT_LAYER.store(layer, Ordering::Relaxed);
}

/// True when a routing schedule is installed (fast-path probe for callers that
/// want to skip per-expert work setup entirely).
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Prune the per-expert activation buffer in place to the configured
/// within-expert feature subset for the current layer.
///
/// `act[..inter]` holds the post-activation features; padding columns
/// (`inter..`) are already zero and are left untouched. No-op (a single relaxed
/// atomic load) when no routing is installed.
///
/// ACTIVE can never become true on wasm32 (set_routing is a no-op
/// there), so this always takes the fast no-op path -- kept as an
/// explicit wasm32 stub rather than relying on that implicitly, since
/// the real body also references the native-only ROUTING/CURRENT_LAYER
/// statics.
#[cfg(target_arch = "wasm32")]
pub fn prune_act(_act: &mut [f32], _inter: usize) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn prune_act(act: &mut [f32], inter: usize) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if inter == 0 || act.len() < inter {
        return;
    }
    let layer = CURRENT_LAYER.load(Ordering::Relaxed);
    let guard = ROUTING.read().unwrap_or_else(|p| p.into_inner());
    let Some(routing) = guard.as_ref() else {
        return;
    };
    let Some(k) = routing.keep_k(layer, inter) else {
        return;
    };
    match routing.selector {
        ExpertFeatureSelector::ActMagnitude => prune_top_k(&mut act[..inter], k),
        ExpertFeatureSelector::Strided => prune_strided(&mut act[..inter], k),
    }
}

/// Keep the `k` largest-magnitude features, zero the rest. `1 <= k < act.len()`
/// guaranteed by [`WithinExpertRouting::keep_k`].
fn prune_top_k(act: &mut [f32], k: usize) {
    let inter = act.len();
    // Partition indices so the first `k` are the largest by |act| (the set, not
    // the order, is what matters — we only zero the complement).
    let mut idx: Vec<usize> = (0..inter).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| act[b].abs().total_cmp(&act[a].abs()));
    let mut keep = vec![false; inter];
    for &i in &idx[..k] {
        keep[i] = true;
    }
    for (v, &keep_i) in act.iter_mut().zip(keep.iter()) {
        if !keep_i {
            *v = 0.0;
        }
    }
}

/// Keep `~k` features on a fixed stride (content-blind), zero the rest.
fn prune_strided(act: &mut [f32], k: usize) {
    let inter = act.len();
    let stride = (inter / k).max(1);
    for (i, v) in act.iter_mut().enumerate() {
        if !i.is_multiple_of(stride) {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests mutate process-global routing state; serialise them.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn keep_k_resolves_fraction_against_inter() {
        let mut r = WithinExpertRouting::dense(4);
        r.frac_per_layer[1] = Some(0.25);
        assert_eq!(r.keep_k(1, 704), Some(176)); // round(0.25*704)
        assert_eq!(r.keep_k(0, 704), None); // dense layer
        assert_eq!(r.keep_k(1, 0), None); // no features
    }

    #[test]
    fn keep_k_clamps_and_treats_full_fraction_as_dense() {
        let mut r = WithinExpertRouting::dense(2);
        r.frac_per_layer[0] = Some(1.0); // keep-all → None
        r.frac_per_layer[1] = Some(0.0); // round→0, clamped to 1
        assert_eq!(r.keep_k(0, 100), None);
        assert_eq!(r.keep_k(1, 100), Some(1));
        // frac that rounds to >= inter is dense.
        let mut r2 = WithinExpertRouting::dense(1);
        r2.frac_per_layer[0] = Some(0.999);
        assert_eq!(r2.keep_k(0, 100), None);
    }

    #[test]
    fn prune_off_is_identity() {
        let _g = guard();
        set_routing(None);
        let mut act = vec![3.0f32, -5.0, 1.0, 0.5, -9.0, 2.0];
        let before = act.clone();
        let n = act.len();
        prune_act(&mut act, n);
        assert_eq!(act, before, "no routing installed → byte-identical");
    }

    #[test]
    fn is_active_tracks_routing_install() {
        let _g = guard();
        set_routing(None);
        assert!(!is_active(), "no schedule installed → inactive");
        set_routing(Some(WithinExpertRouting::dense(2)));
        assert!(is_active(), "schedule installed → active");
        set_routing(None);
        assert!(!is_active(), "cleared → inactive again");
    }

    #[test]
    fn prune_guards_zero_inter_and_short_slice() {
        let _g = guard();
        // Active schedule, but the guards must short-circuit before any pruning.
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![Some(0.5)],
            selector: ExpertFeatureSelector::ActMagnitude,
        }));
        set_current_layer(0);
        // inter == 0 → no-op.
        let mut a = vec![1.0f32, 2.0, 3.0];
        let before = a.clone();
        prune_act(&mut a, 0);
        assert_eq!(a, before, "inter=0 must be a no-op");
        // act shorter than the claimed inter → no-op (defensive guard).
        let mut b = vec![1.0f32, 2.0];
        let before_b = b.clone();
        prune_act(&mut b, 8);
        assert_eq!(b, before_b, "act.len() < inter must be a no-op");
        set_routing(None);
    }

    #[test]
    fn prune_top_k_keeps_largest_magnitude() {
        let _g = guard();
        // |act| = [3,5,1,0.5,9,2]; top-3 by magnitude = indices 4,1,0.
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![Some(0.5)], // 0.5*6 = 3 features
            selector: ExpertFeatureSelector::ActMagnitude,
        }));
        set_current_layer(0);
        let mut act = vec![3.0f32, -5.0, 1.0, 0.5, -9.0, 2.0];
        prune_act(&mut act, 6);
        assert_eq!(act, vec![3.0, -5.0, 0.0, 0.0, -9.0, 0.0]);
        set_routing(None);
    }

    #[test]
    fn prune_strided_keeps_on_stride() {
        let _g = guard();
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![Some(0.5)], // k=3, stride = 6/3 = 2
            selector: ExpertFeatureSelector::Strided,
        }));
        set_current_layer(0);
        let mut act = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        prune_act(&mut act, 6);
        // keep indices 0,2,4.
        assert_eq!(act, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
        set_routing(None);
    }

    #[test]
    fn prune_honours_current_layer_and_dense_layers_untouched() {
        let _g = guard();
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![None, Some(0.5)],
            selector: ExpertFeatureSelector::ActMagnitude,
        }));
        let original = vec![3.0f32, -5.0, 1.0, 0.5, -9.0, 2.0];

        // Layer 0 is dense → untouched.
        set_current_layer(0);
        let mut a0 = original.clone();
        prune_act(&mut a0, 6);
        assert_eq!(a0, original);

        // Layer 1 prunes to top-3.
        set_current_layer(1);
        let mut a1 = original.clone();
        prune_act(&mut a1, 6);
        assert_eq!(a1, vec![3.0, -5.0, 0.0, 0.0, -9.0, 0.0]);
        set_routing(None);
    }

    #[test]
    fn prune_padding_columns_untouched() {
        let _g = guard();
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![Some(0.5)],
            selector: ExpertFeatureSelector::ActMagnitude,
        }));
        set_current_layer(0);
        // inter=4, buffer has 2 padding columns that must stay as-is.
        let mut act = vec![1.0f32, 8.0, 2.0, 7.0, 99.0, 99.0];
        prune_act(&mut act, 4);
        // top-2 of first 4 = indices 1,3; padding (4,5) untouched.
        assert_eq!(act, vec![0.0, 8.0, 0.0, 7.0, 99.0, 99.0]);
        set_routing(None);
    }
}
