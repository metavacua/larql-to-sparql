//! **R8-policy — refusals and reporting.**
//!
//! `window_policy_gates` drives the policy over a real `StandardEngine`,
//! so it only reaches caches that already satisfy every precondition.
//! These cover the other direction: the refusals the policy owes a
//! caller whose cache cannot express the question, and the reporting
//! helpers on the outcome.
//!
//! Every refusal here fires *before* a row is touched. That ordering is
//! the point — a policy that clipped some layers and then discovered it
//! could not clip the rest would leave the cache in a state no caller
//! asked for, and `WindowPolicyOutcome` would describe a cache that no
//! longer exists.

use larql_inference::kv_engine::{ExcisedKvRows, PerLayerKvAccess};
use larql_inference::kv_row_positions::KvRowPositions;
use ndarray::Array2;

use crate::engines::semantic_promotion::error::PromotionError;
use crate::engines::semantic_promotion::exclusion::LayerAttention;
use crate::engines::semantic_promotion::window_policy::{apply_window_policy, WindowPolicyOutcome};

/// A cache whose capabilities are dialled in directly, so each
/// precondition can be withdrawn on its own. Only the methods the policy
/// actually consults do anything.
struct StubCache {
    layer_count: Option<usize>,
    positions: Option<KvRowPositions>,
    /// What `clip_layer_to_logical_window` returns; `None` models a
    /// layer that refused the clip.
    clip_result: Option<usize>,
    clip_calls: std::cell::Cell<usize>,
}

impl StubCache {
    fn new(layers: usize) -> Self {
        Self {
            layer_count: Some(layers),
            positions: Some(KvRowPositions::contiguous(layers, 0)),
            clip_result: Some(0),
            clip_calls: std::cell::Cell::new(0),
        }
    }
}

impl PerLayerKvAccess for StubCache {
    fn layer_count(&self) -> Option<usize> {
        self.layer_count
    }
    fn logical_next_position(&self) -> usize {
        0
    }
    fn resident_rows(&self, _layer: usize) -> Option<usize> {
        Some(0)
    }
    fn read_layer_kv(&self, _layer: usize) -> Option<(Array2<f32>, Array2<f32>)> {
        None
    }
    fn excise_kv_rows(
        &mut self,
        _layer: usize,
        _rows: std::ops::Range<usize>,
    ) -> Option<ExcisedKvRows> {
        None
    }
    fn splice_kv_rows(&mut self, _layer: usize, _at: usize, _rows: &ExcisedKvRows) -> bool {
        false
    }
    fn replace_layer_kv(
        &mut self,
        _layer: usize,
        _k: &Array2<f32>,
        _v: &Array2<f32>,
        _positions: &[u64],
    ) -> bool {
        false
    }
    fn row_positions(&self) -> Option<&KvRowPositions> {
        self.positions.as_ref()
    }
    fn set_logical_next_position(&mut self, _position: usize) -> bool {
        false
    }
    fn clip_layer_to_logical_window(&mut self, _layer: usize, _window: u64) -> Option<usize> {
        self.clip_calls.set(self.clip_calls.get() + 1);
        self.clip_result
    }
}

/// `n` layers, the first `sliding` of them sliding, with `window`
/// declared.
fn attention(n: usize, sliding: usize, window: Option<usize>) -> LayerAttention {
    LayerAttention {
        is_sliding: (0..n).map(|i| i < sliding).collect(),
        sliding_window: window,
    }
}

// ── outcome reporting ────────────────────────────────────────────────────

/// `total_rows_dropped` sums across layers; `is_no_op` is that total
/// being zero. A cache already inside every window it declares is not a
/// failure, and must not read as one.
#[test]
fn outcome_totals_and_no_op_reporting() {
    let empty = WindowPolicyOutcome::default();
    assert_eq!(empty.total_rows_dropped(), 0);
    assert!(empty.is_no_op());

    let untouched = WindowPolicyOutcome {
        rows_dropped: vec![0, 0, 0],
        sliding_layers: 2,
        global_layers: 1,
    };
    assert_eq!(untouched.total_rows_dropped(), 0);
    assert!(untouched.is_no_op(), "no row moved, so this is a no-op");

    let clipped = WindowPolicyOutcome {
        rows_dropped: vec![3, 0, 4],
        sliding_layers: 2,
        global_layers: 1,
    };
    assert_eq!(clipped.total_rows_dropped(), 7);
    assert!(!clipped.is_no_op());
}

// ── refusals ─────────────────────────────────────────────────────────────

/// No per-layer access at all — a coarse whole-model handle has no
/// granularity to apply a per-layer policy to.
#[test]
fn a_cache_with_no_layer_count_is_refused() {
    let mut kv = StubCache::new(2);
    kv.layer_count = None;

    let err = apply_window_policy(&mut kv, &attention(2, 1, Some(4)))
        .expect_err("no per-layer access should refuse");
    assert!(matches!(err, PromotionError::UnsupportedCapability(_)));
    assert_eq!(kv.clip_calls.get(), 0, "refused before touching a layer");
}

/// The architecture and the cache disagreeing about layer count means
/// `is_sliding[layer]` would be indexing a different model than the one
/// resident — an out-of-bounds read at best, the wrong classification at
/// worst.
#[test]
fn a_layer_count_disagreement_is_refused() {
    let mut kv = StubCache::new(2);

    let err = apply_window_policy(&mut kv, &attention(6, 5, Some(4)))
        .expect_err("layer-count disagreement should refuse");
    assert!(matches!(err, PromotionError::InvariantViolation(_)));
    assert_eq!(kv.clip_calls.get(), 0, "refused before touching a layer");
}

/// Without a row-position map, logical age is unknowable: row count and
/// next position stop determining which positions are resident the
/// moment the cache has a hole.
#[test]
fn a_cache_with_no_row_position_map_is_refused() {
    let mut kv = StubCache::new(2);
    kv.positions = None;

    let err = apply_window_policy(&mut kv, &attention(2, 1, Some(4)))
        .expect_err("missing row positions should refuse");
    assert!(matches!(err, PromotionError::UnsupportedCapability(_)));
    assert_eq!(kv.clip_calls.get(), 0, "refused before touching a layer");
}

/// Sliding layers with no declared window have no bound to apply.
/// Silently keeping everything would report as a clip that did nothing,
/// which is indistinguishable from a cache already inside its window.
#[test]
fn sliding_layers_without_a_declared_window_are_refused() {
    let mut kv = StubCache::new(2);

    let err = apply_window_policy(&mut kv, &attention(2, 1, None))
        .expect_err("sliding layers with no window should refuse");
    assert!(matches!(err, PromotionError::UnsupportedCapability(_)));
    assert_eq!(kv.clip_calls.get(), 0, "refused before touching a layer");
}

/// An all-global architecture with no declared window is *not* a
/// refusal: there is no sliding layer for the missing window to bound,
/// so the policy is well defined and drops nothing.
#[test]
fn an_all_global_architecture_needs_no_window() {
    let mut kv = StubCache::new(3);

    let outcome = apply_window_policy(&mut kv, &attention(3, 0, None))
        .expect("no sliding layer means no window is needed");
    assert_eq!(outcome.rows_dropped, vec![0, 0, 0]);
    assert_eq!(outcome.sliding_layers, 0);
    assert_eq!(outcome.global_layers, 3);
    assert!(outcome.is_no_op());
    assert_eq!(
        kv.clip_calls.get(),
        0,
        "a global layer is not clipped, not clipped-with-a-large-window"
    );
}

/// A layer that refuses its clip surfaces as a failure rather than being
/// recorded as "dropped nothing" — the distinction between a correct
/// no-op and a failed clip is the entire content of the outcome.
#[test]
fn a_layer_that_refuses_its_clip_fails_the_policy() {
    let mut kv = StubCache::new(2);
    kv.clip_result = None;

    let err = apply_window_policy(&mut kv, &attention(2, 1, Some(4)))
        .expect_err("a refused clip should not be reported as zero rows");
    assert!(matches!(err, PromotionError::SourceMaskFailed));
}

// ── the accepting path ───────────────────────────────────────────────────

/// Sliding layers are clipped, global layers are left whole, and the
/// split is reported rather than summarised.
#[test]
fn only_sliding_layers_are_clipped_and_the_split_is_reported() {
    let mut kv = StubCache::new(4);
    kv.clip_result = Some(2);

    let outcome = apply_window_policy(&mut kv, &attention(4, 2, Some(4))).expect("policy applies");
    assert_eq!(
        outcome.rows_dropped,
        vec![2, 2, 0, 0],
        "global layers must report 0, not a clip that happened to drop nothing"
    );
    assert_eq!(outcome.sliding_layers, 2);
    assert_eq!(outcome.global_layers, 2);
    assert_eq!(outcome.total_rows_dropped(), 4);
    assert!(!outcome.is_no_op());
    assert_eq!(kv.clip_calls.get(), 2, "one clip per sliding layer only");
}
