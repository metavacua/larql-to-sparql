//! Mid-forward hook system — read and write the residual stream during a
//! forward pass.
//!
//! Lazarus-style mechanistic interp tools (capture, ablate, patch, steer,
//! probe, DLA) all collapse to one primitive: an in-process callback that
//! fires at well-defined points inside each transformer layer and may
//! optionally mutate the residual.
//!
//! The trait has five callbacks, each defaulting to a no-op so impls only
//! override what they need:
//!
//! - [`LayerHook::on_pre_layer`] — read residual entering the layer.
//! - [`LayerHook::on_post_attention`] — **read or write** post-attention
//!   residual, before FFN.
//! - [`LayerHook::on_attention_weights`] — read per-head attention.
//! - [`LayerHook::on_ffn_activation`] — read FFN gate activation.
//! - [`LayerHook::on_post_layer`] — **read or write** the residual exiting
//!   the layer.
//!
//! The two `&mut` callbacks are what unlock the entire intervention surface.
//! Ablation, steering, patching, and subspace surgery are all just
//! [`LayerHook`] impls over those points.
//!
//! Plumbing: `run_layer_with_capture` and `trace_forward_full_hooked` accept
//! a `&mut dyn LayerHook`. The existing zero-hook signatures stay as thin
//! wrappers passing [`NoopHook`], so call-sites that don't care pay no cost.

use crate::attention::AttentionWeights;
use ndarray::{Array1, Array2};
use crate::collections::{HashMap, HashSet};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Mid-forward callbacks. All defaults are no-ops; impls override only the
/// callbacks they need.
///
/// `on_post_attention` and `on_post_layer` take `&mut Array2<f32>` so a hook
/// can mutate the residual in place. The other three callbacks are
/// read-only.
#[allow(unused_variables)]
pub trait LayerHook {
    /// Fires before attention runs at `layer`. `h` is the residual entering
    /// the layer (post-norm has not yet been applied).
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {}

    /// Fires after attention, before FFN. The hook may mutate `h` in place
    /// — that is the insertion point for activation patching and
    /// pre-FFN steering.
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {}

    /// Fires when attention weights have been captured. Read-only.
    /// Only called on layers where `capture_attention=true` was requested.
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {}

    /// Fires when an FFN gate activation has been captured. Read-only.
    /// Only called on layers where `capture_activation=true` was requested.
    /// Shape is `(seq_len, ffn_dim)`.
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {}

    /// Fires after the full layer (attention + FFN + PLE + scalar). The
    /// hook may mutate `h` — that is the insertion point for residual-stream
    /// ablation, steering, and any "edit before the next layer sees it"
    /// transform.
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {}
}

/// Hook that does nothing. Used as the default when callers don't care.
pub struct NoopHook;
impl LayerHook for NoopHook {}

/// Captures pre-layer / post-attention / post-layer residuals (and optionally
/// FFN activations + attention weights) at the requested layers. Replaces
/// the file-output pattern of the legacy `LARQL_CPU_DUMP_LAYERS` env var.
///
/// Use [`RecordHook::for_layers`] to construct, then read the public maps
/// after the forward pass returns.
pub struct RecordHook {
    /// Layers to record. Other layers are skipped (zero overhead).
    pub layers: HashSet<usize>,
    /// `(seq_len, hidden)` residual entering each captured layer.
    pub pre_layer: HashMap<usize, Array2<f32>>,
    /// `(seq_len, hidden)` residual after attention at each captured layer.
    pub post_attention: HashMap<usize, Array2<f32>>,
    /// `(seq_len, hidden)` residual after the full layer.
    pub post_layer: HashMap<usize, Array2<f32>>,
    /// `(seq_len, ffn_dim)` FFN gate activation. Only populated when the
    /// outer trace was asked to capture FFN activations.
    pub ffn_activation: HashMap<usize, Array2<f32>>,
    /// Per-head attention weights for the last token position. Only
    /// populated when the outer trace was asked to capture attention.
    pub attention_weights: HashMap<usize, Vec<Vec<f32>>>,
}

impl RecordHook {
    /// Build a recorder that captures the listed layers.
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().collect(),
            pre_layer: HashMap::default(),
            post_attention: HashMap::default(),
            post_layer: HashMap::default(),
            ffn_activation: HashMap::default(),
            attention_weights: HashMap::default(),
        }
    }
}

impl LayerHook for RecordHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {
        if self.layers.contains(&layer) {
            self.pre_layer.insert(layer, h.clone());
        }
    }
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layers.contains(&layer) {
            self.post_attention.insert(layer, h.clone());
        }
    }
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {
        if self.layers.contains(&layer) {
            self.attention_weights.insert(layer, weights.heads.clone());
        }
    }
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {
        if self.layers.contains(&layer) {
            self.ffn_activation.insert(layer, gate.clone());
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layers.contains(&layer) {
            self.post_layer.insert(layer, h.clone());
        }
    }
}

/// Zeros rows of the post-layer residual at requested layers.
///
/// `positions == None` zeros every row at that layer (full-layer ablation).
/// `positions == Some(vec)` zeros only the listed token positions.
///
/// Implements lazarus's `ablate_layers` and per-position residual ablation.
pub struct ZeroAblateHook {
    pub layers: HashMap<usize, Option<Vec<usize>>>,
}

impl ZeroAblateHook {
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().map(|l| (l, None)).collect(),
        }
    }
}

impl LayerHook for ZeroAblateHook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        let Some(positions) = self.layers.get(&layer) else {
            return;
        };
        match positions {
            None => h.fill(0.0),
            Some(ps) => {
                let n_rows = h.nrows();
                for &p in ps {
                    if p < n_rows {
                        h.row_mut(p).fill(0.0);
                    }
                }
            }
        }
    }
}

/// Zeroes the FFN sublayer contribution at requested layers by snapshotting
/// the residual after attention and restoring it at post-layer. The FFN
/// computation still runs (so other hooks observing `on_ffn_activation` still
/// fire) but its addition to the residual stream — along with any PLE / scalar
/// adjustment applied inside the post-attention → post-layer step — is
/// discarded.
///
/// Pair with [`AttnZeroHook`] for sublayer-decomposition forwards: one pass
/// with each isolates the attention vs. FFN contribution to the final
/// residual.
pub struct FFNZeroHook {
    pub layers: HashSet<usize>,
    /// Per-layer snapshot taken in `on_post_attention`, consumed in
    /// `on_post_layer`.
    pub cache: HashMap<usize, Array2<f32>>,
}

impl FFNZeroHook {
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().collect(),
            cache: HashMap::default(),
        }
    }
}

impl LayerHook for FFNZeroHook {
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layers.contains(&layer) {
            self.cache.insert(layer, h.clone());
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if let Some(saved) = self.cache.remove(&layer) {
            h.assign(&saved);
        }
    }
}

/// Zeroes the attention sublayer contribution at requested layers by
/// snapshotting the residual entering the layer and restoring it at
/// post-attention. The attention computation still runs (so hooks observing
/// `on_attention_weights` still fire) but its addition to the residual stream
/// is discarded.
pub struct AttnZeroHook {
    pub layers: HashSet<usize>,
    pub cache: HashMap<usize, Array2<f32>>,
}

impl AttnZeroHook {
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().collect(),
            cache: HashMap::default(),
        }
    }
}

impl LayerHook for AttnZeroHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {
        if self.layers.contains(&layer) {
            self.cache.insert(layer, h.clone());
        }
    }
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        if let Some(saved) = self.cache.remove(&layer) {
            h.assign(&saved);
        }
    }
}

/// Adds `alpha * v` to the last-token row of the post-layer residual at
/// requested layers. Implements lazarus's `steer_and_generate`.
///
/// Use a separate `SteerHook` per (layer, vector) pair, or compose them in
/// [`CompositeHook`].
pub struct SteerHook {
    /// Layer → (steering vector of shape `(hidden,)`, scalar gain).
    pub steers: HashMap<usize, (Array1<f32>, f32)>,
}

impl SteerHook {
    pub fn new() -> Self {
        Self {
            steers: HashMap::default(),
        }
    }

    pub fn add(mut self, layer: usize, vector: Array1<f32>, alpha: f32) -> Self {
        self.steers.insert(layer, (vector, alpha));
        self
    }
}

impl Default for SteerHook {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerHook for SteerHook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        let Some((v, alpha)) = self.steers.get(&layer) else {
            return;
        };
        if h.nrows() == 0 || v.len() != h.ncols() {
            return;
        }
        let last = h.nrows() - 1;
        let mut row = h.row_mut(last);
        for (i, val) in row.iter_mut().enumerate() {
            *val += *alpha * v[i];
        }
    }
}

/// Runs an arbitrary collection of hooks in order. Useful for combining
/// (e.g.) a `RecordHook` with a `SteerHook` so you can both intervene and
/// measure in one pass.
pub struct CompositeHook<'a> {
    pub hooks: Vec<&'a mut dyn LayerHook>,
}

impl<'a> CompositeHook<'a> {
    pub fn new(hooks: Vec<&'a mut dyn LayerHook>) -> Self {
        Self { hooks }
    }
}

impl LayerHook for CompositeHook<'_> {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_pre_layer(layer, h);
        }
    }
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_post_attention(layer, h);
        }
    }
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {
        for hook in self.hooks.iter_mut() {
            hook.on_attention_weights(layer, weights);
        }
    }
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_ffn_activation(layer, gate);
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_post_layer(layer, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn noop_hook_compiles_and_does_nothing() {
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let mut hook = NoopHook;
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original);
    }

    #[test]
    fn record_hook_captures_only_requested_layers() {
        let mut hook = RecordHook::for_layers([1, 3]);
        let mut h: Array2<f32> = array![[1.0, 2.0]];

        hook.on_pre_layer(0, &h); // not in set
        hook.on_pre_layer(1, &h); // in set
        hook.on_post_layer(2, &mut h); // not in set
        hook.on_post_layer(3, &mut h); // in set

        assert!(!hook.pre_layer.contains_key(&0));
        assert!(hook.pre_layer.contains_key(&1));
        assert!(!hook.post_layer.contains_key(&2));
        assert!(hook.post_layer.contains_key(&3));
    }

    #[test]
    fn record_hook_clones_residual_so_later_writes_dont_pollute() {
        let mut hook = RecordHook::for_layers([0]);
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_pre_layer(0, &h);
        h[[0, 0]] = 999.0;
        let recorded = hook.pre_layer.get(&0).unwrap();
        assert_eq!(recorded[[0, 0]], 1.0, "RecordHook must snapshot, not alias");
    }

    #[test]
    fn zero_ablate_full_layer() {
        let mut hook = ZeroAblateHook::for_layers([2]);
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, array![[1.0, 2.0], [3.0, 4.0]], "wrong layer untouched");
        hook.on_post_layer(2, &mut h);
        assert_eq!(h, array![[0.0, 0.0], [0.0, 0.0]], "target layer zeroed");
    }

    #[test]
    fn zero_ablate_specific_positions() {
        let mut hook = ZeroAblateHook {
            layers: [(1, Some(vec![1, 3]))].into_iter().collect(),
        };
        let mut h: Array2<f32> = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        hook.on_post_layer(1, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![1.0, 1.0], "pos 0 untouched");
        assert_eq!(h.row(1).to_vec(), vec![0.0, 0.0], "pos 1 zeroed");
        assert_eq!(h.row(2).to_vec(), vec![3.0, 3.0], "pos 2 untouched");
        assert_eq!(h.row(3).to_vec(), vec![0.0, 0.0], "pos 3 zeroed");
    }

    #[test]
    fn zero_ablate_out_of_range_position_is_noop() {
        let mut hook = ZeroAblateHook {
            layers: [(0, Some(vec![99]))].into_iter().collect(),
        };
        let mut h: Array2<f32> = array![[1.0, 2.0]];
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original);
    }

    #[test]
    fn steer_adds_alpha_v_to_last_row() {
        let mut hook = SteerHook::new().add(0, array![10.0, 20.0], 0.5);
        let mut h: Array2<f32> = array![[1.0, 1.0], [2.0, 2.0]];
        hook.on_post_layer(0, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![1.0, 1.0], "non-last row untouched");
        assert_eq!(
            h.row(1).to_vec(),
            vec![2.0 + 0.5 * 10.0, 2.0 + 0.5 * 20.0],
            "last row += alpha * v"
        );
    }

    #[test]
    fn steer_silently_skips_on_dim_mismatch() {
        let mut hook = SteerHook::new().add(0, array![1.0, 2.0, 3.0], 1.0);
        let mut h: Array2<f32> = array![[1.0, 1.0]];
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original, "wrong-dim vector must not corrupt residual");
    }

    #[test]
    fn composite_runs_hooks_in_order() {
        // Steer then record: recorded value must include the steer.
        let mut steer = SteerHook::new().add(0, array![1.0, 1.0], 1.0);
        let mut record = RecordHook::for_layers([0]);
        let mut comp = CompositeHook::new(vec![&mut steer, &mut record]);
        let mut h: Array2<f32> = array![[5.0, 5.0]];
        comp.on_post_layer(0, &mut h);
        let recorded = record.post_layer.get(&0).unwrap();
        assert_eq!(recorded.row(0).to_vec(), vec![6.0, 6.0]);
    }

    /// `NoopHook` accepts every callback as a no-op; pin the trait
    /// surface so a future signature drift would break compile.
    #[test]
    fn noop_hook_accepts_every_callback() {
        let mut hook = NoopHook;
        let mut h: Array2<f32> = array![[0.0, 0.0]];
        let attn_weights = AttentionWeights {
            heads: vec![vec![1.0, 0.0]],
        };
        let gate: Array2<f32> = array![[0.1, 0.2]];
        hook.on_pre_layer(0, &h);
        hook.on_post_attention(0, &mut h);
        hook.on_attention_weights(0, &attn_weights);
        hook.on_ffn_activation(0, &gate);
        hook.on_post_layer(0, &mut h);
    }

    /// `RecordHook` records every callback at matching layers — drives
    /// the `on_post_attention`, `on_attention_weights`, `on_ffn_activation`
    /// paths in addition to the pre/post layer ones already covered.
    #[test]
    fn record_hook_records_every_callback_kind() {
        let mut record = RecordHook::for_layers([0, 1]);
        let mut h: Array2<f32> = array![[0.0, 0.0]];
        let attn = AttentionWeights {
            heads: vec![vec![0.5, 0.5]],
        };
        let gate: Array2<f32> = array![[0.1, 0.2]];

        // Layer 0 in scope — every callback records.
        record.on_post_attention(0, &mut h);
        record.on_attention_weights(0, &attn);
        record.on_ffn_activation(0, &gate);
        assert!(record.post_attention.contains_key(&0));
        assert!(record.attention_weights.contains_key(&0));
        assert!(record.ffn_activation.contains_key(&0));

        // Layer 5 NOT in scope — none recorded.
        record.on_post_attention(5, &mut h);
        record.on_attention_weights(5, &attn);
        record.on_ffn_activation(5, &gate);
        assert!(!record.post_attention.contains_key(&5));
        assert!(!record.attention_weights.contains_key(&5));
        assert!(!record.ffn_activation.contains_key(&5));
    }

    /// `CompositeHook` forwards every callback kind to every member.
    #[test]
    fn composite_hook_forwards_every_callback_kind() {
        let mut record_a = RecordHook::for_layers([0]);
        let mut record_b = RecordHook::for_layers([0]);
        {
            let mut comp = CompositeHook::new(vec![&mut record_a, &mut record_b]);
            let mut h: Array2<f32> = array![[0.0, 0.0]];
            let attn = AttentionWeights {
                heads: vec![vec![0.5, 0.5]],
            };
            let gate: Array2<f32> = array![[0.1, 0.2]];
            comp.on_pre_layer(0, &h);
            comp.on_post_attention(0, &mut h);
            comp.on_attention_weights(0, &attn);
            comp.on_ffn_activation(0, &gate);
            comp.on_post_layer(0, &mut h);
        }
        // Both members received every callback.
        assert!(record_a.pre_layer.contains_key(&0));
        assert!(record_a.post_attention.contains_key(&0));
        assert!(record_a.attention_weights.contains_key(&0));
        assert!(record_a.ffn_activation.contains_key(&0));
        assert!(record_a.post_layer.contains_key(&0));
        assert!(record_b.attention_weights.contains_key(&0));
    }

    /// `SteerHook::default()` calls `new()` — pin the default impl.
    #[test]
    fn steer_hook_default_is_empty() {
        let hook = SteerHook::default();
        assert!(hook.steers.is_empty());
    }

    #[test]
    fn ffn_zero_restores_post_attention_residual() {
        let mut hook = FFNZeroHook::for_layers([0]);
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_post_attention(0, &mut h);
        let after_ffn: Array2<f32> = array![[10.0, 20.0], [30.0, 40.0]];
        let mut h_after = after_ffn.clone();
        hook.on_post_layer(0, &mut h_after);
        assert_eq!(
            h_after,
            array![[1.0, 2.0], [3.0, 4.0]],
            "post_layer must be restored to the snapshot taken at post_attention"
        );
    }

    #[test]
    fn ffn_zero_only_affects_listed_layers() {
        let mut hook = FFNZeroHook::for_layers([1]);
        let mut h: Array2<f32> = array![[1.0, 2.0]];
        hook.on_post_attention(0, &mut h);
        let mut h_after: Array2<f32> = array![[99.0, 99.0]];
        hook.on_post_layer(0, &mut h_after);
        assert_eq!(
            h_after,
            array![[99.0, 99.0]],
            "layer 0 is not in set — post_layer is untouched"
        );
    }

    #[test]
    fn attn_zero_restores_pre_layer_residual() {
        let mut hook = AttnZeroHook::for_layers([0]);
        let h_pre: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_pre_layer(0, &h_pre);
        let mut h_after_attn: Array2<f32> = array![[100.0, 200.0], [300.0, 400.0]];
        hook.on_post_attention(0, &mut h_after_attn);
        assert_eq!(
            h_after_attn,
            array![[1.0, 2.0], [3.0, 4.0]],
            "post_attention must be restored to the pre_layer snapshot"
        );
    }

    #[test]
    fn attn_zero_only_affects_listed_layers() {
        let mut hook = AttnZeroHook::for_layers([1]);
        let h_pre: Array2<f32> = array![[1.0, 2.0]];
        hook.on_pre_layer(0, &h_pre);
        let mut h_after_attn: Array2<f32> = array![[99.0, 99.0]];
        hook.on_post_attention(0, &mut h_after_attn);
        assert_eq!(
            h_after_attn,
            array![[99.0, 99.0]],
            "layer 0 is not in set — post_attention is untouched"
        );
    }

    /// `SteerHook::on_post_layer` early-returns when h has zero rows or
    /// the vector width doesn't match — pin both guards.
    #[test]
    fn steer_hook_handles_shape_mismatch_gracefully() {
        let mut steer = SteerHook::new().add(0, array![1.0, 1.0, 1.0], 1.0);
        // Wrong width: vector is 3, h cols is 2 → no-op.
        let mut h: Array2<f32> = array![[5.0, 5.0]];
        steer.on_post_layer(0, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![5.0, 5.0]);
        // Zero rows: no-op.
        let mut h0: Array2<f32> = Array2::zeros((0, 3));
        steer.on_post_layer(0, &mut h0);
        // No matching layer: no-op.
        let mut h_other: Array2<f32> = array![[5.0, 5.0, 5.0]];
        steer.on_post_layer(99, &mut h_other);
        assert_eq!(h_other.row(0).to_vec(), vec![5.0, 5.0, 5.0]);
    }
}
