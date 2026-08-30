//! **R8-policy.** Mixed sliding/global retention over a sparse cache.
//!
//! [`super::exclusion::LayerAttention`] already reads a model's per-layer
//! attention classification, and the engine now knows the logical
//! position of every resident row. This is the operation that puts the
//! two together: on a sliding layer keep the rows inside the declared
//! window *by logical age*, on a global layer keep everything.
//!
//! Both halves are load-bearing and they pull in opposite directions:
//!
//! ```text
//! sliding layer   a pre-hole row is thousands of positions old
//!                 -> it must go, however few physical rows precede it
//! global layer    the same row is the source the record replaced
//!                 -> it must stay, however old it is
//! ```
//!
//! Uniform clipping cannot express that, which is why `clip_kv`'s
//! engine-level window gets one answer wrong whichever value it takes.
//!
//! **This does not change decode.** The CPU per-layer decode path still
//! calls `clip_kv` with the engine-level window and never consults
//! `is_sliding_window_layer`. Applying the policy is an explicit
//! operation a caller performs on a cache, gated here; routing decode
//! through it is a production change that R8d qualifies but does not
//! itself make.

use larql_inference::kv_engine::PerLayerKvAccess;

use super::error::PromotionError;
use super::exclusion::LayerAttention;

/// What the policy did, per layer.
///
/// Reported rather than summarised: "12 rows dropped" cannot distinguish
/// a correct sliding clip from a global layer that was wrongly clipped,
/// and that distinction is the entire content of the milestone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowPolicyOutcome {
    /// Rows removed from each layer, in layer order. Always `0` on a
    /// global layer.
    pub rows_dropped: Vec<usize>,
    /// Layers the window was applied to.
    pub sliding_layers: usize,
    /// Layers deliberately left whole.
    pub global_layers: usize,
}

impl WindowPolicyOutcome {
    pub fn total_rows_dropped(&self) -> usize {
        self.rows_dropped.iter().sum()
    }

    /// `true` when no layer lost a row — the cache was already inside
    /// every window it declares.
    pub fn is_no_op(&self) -> bool {
        self.total_rows_dropped() == 0
    }
}

/// Enforce `attention`'s per-layer retention on `kv`, by logical age.
///
/// Refusals, all before any row is touched:
///
/// - the cache offers no per-layer access, or no row-position map;
/// - the map's layer count disagrees with the architecture's;
/// - a layer is sliding but the architecture declares no window, so
///   there is no bound to apply and silently keeping everything would
///   misreport as a clip.
pub fn apply_window_policy(
    kv: &mut dyn PerLayerKvAccess,
    attention: &LayerAttention,
) -> Result<WindowPolicyOutcome, PromotionError> {
    let layers = kv
        .layer_count()
        .ok_or(PromotionError::UnsupportedCapability(
            "base engine has no per-layer K/V access on this path",
        ))?;
    if layers != attention.layer_count() {
        return Err(PromotionError::InvariantViolation(
            "architecture and cache disagree on layer count",
        ));
    }
    if kv.row_positions().is_none() {
        return Err(PromotionError::UnsupportedCapability(
            "base engine has no row-position map on this path",
        ));
    }
    let sliding = attention.is_sliding.iter().filter(|&&s| s).count();
    let window = match attention.sliding_window {
        Some(w) => w as u64,
        None if sliding == 0 => 0,
        None => {
            return Err(PromotionError::UnsupportedCapability(
                "architecture declares sliding layers but no window size",
            ))
        }
    };

    let mut rows_dropped = Vec::with_capacity(layers);
    for layer in 0..layers {
        if attention.is_sliding[layer] {
            let dropped = kv
                .clip_layer_to_logical_window(layer, window)
                .ok_or(PromotionError::SourceMaskFailed)?;
            rows_dropped.push(dropped);
        } else {
            // A global layer is not "clipped with a large window" — it is
            // not clipped. Saying so keeps the outcome readable.
            rows_dropped.push(0);
        }
    }
    Ok(WindowPolicyOutcome {
        rows_dropped,
        sliding_layers: sliding,
        global_layers: layers - sliding,
    })
}
