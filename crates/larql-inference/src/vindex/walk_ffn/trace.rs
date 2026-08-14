//! Runtime trace emission — the executed-path walk trace
//! (2026-07-30 review, item 17).
//!
//! The old `take_trace` recorded residuals during the forward and then
//! RE-RAN plain `gate_knn` afterwards, so the reported trace could be a
//! different feature set than the walk executed (it ignored the
//! selector, pools, and cell router, and any index mutation between
//! forward and drain changed it) — and it reported gate scores, not
//! executed values. This module replaces that: every record is folded
//! from the executed path's own observation
//! ([`FfnActivations`](crate::ffn::FfnActivations)) at the moment the
//! routing ladder returns, riding the `forward`/`forward_observed`
//! seam (item 15) rather than a parallel capture channel.
//!
//! Field vocabulary follows the chuk-introspect metric style
//! (snake_case families: `gate_score`, `up_score`, `activation`,
//! `down_row_norm`, `residual_delta_norm`) — one trace format, no
//! dependency.
//!
//! **Out of scope:** per-feature target-logit contribution. That
//! requires projecting `activation · down_row` through `final_norm` +
//! `lm_head`, which the FFN walk has no access to; compute it post-hoc
//! from these records if needed.

use ndarray::Array2;

use super::plan::FfnPlan;
use super::WalkFfn;
use crate::ffn::FfnActivations;
use larql_vindex::{WalkHit, WalkTrace};

/// `last_path` value before any walk path has executed — only visible
/// if a record were folded before the ladder ran, which the wrapper
/// ordering makes impossible.
pub(super) const PATH_UNROUTED: &str = "unrouted";

/// One executed feature at one (position, layer) — the values the
/// kernel that computed it actually used. Score fields are `None` when
/// the executed path did not compute that score (never fabricated).
#[derive(Debug, Clone, PartialEq)]
pub struct TracedFeature {
    /// Feature index within the layer.
    pub feature: usize,
    /// Executed visit rank: 0 = first feature the kernel visited
    /// (= selection rank on ranked routes, route order on precomputed
    /// routes).
    pub rank: usize,
    /// Gate-position score the executed kernel used.
    pub gate_score: Option<f32>,
    /// Up projection score `u·x`. `None` on non-gated paths.
    pub up_score: Option<f32>,
    /// Scalar pre-down activation.
    pub activation: f32,
    /// `‖down_row‖` served from the selector's lazy norm cache when it
    /// was already built — never computed just for tracing.
    pub down_row_norm: Option<f32>,
}

/// Runtime trace record for one (position, layer), emitted at the
/// routing-ladder exit of the executed forward.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerTraceRecord {
    pub layer: usize,
    /// Sequence position within the traced forward call.
    pub position: usize,
    /// The executed path's `trace_path` name. Per-position for the
    /// sparse walk (whose sub-kernels can differ by position), the
    /// ladder path name otherwise.
    pub path: &'static str,
    /// `‖out_row‖` — the L2 norm of the residual delta this layer's
    /// FFN wrote at this position.
    pub residual_delta_norm: f32,
    /// The executed plan's reason summary (2026-07-30 review, item
    /// 18) — why the ladder chose this path over every rung above it,
    /// including any rungs that declined at execution and were
    /// re-planned past. Same string for every position of one layer
    /// call.
    pub plan_reason: String,
    /// Executed features. Empty for dense whole-layer paths: they
    /// compute every feature with no per-feature selection data, so
    /// per-feature records would require extra compute — the trace
    /// declines rather than fabricating.
    pub features: Vec<TracedFeature>,
}

impl<'a> WalkFfn<'a> {
    /// `‖down_row‖` for one feature, served ONLY from the lazy cache
    /// `selector.rs` builds for the joint selectors. Returns `None`
    /// when the cache was never built for this layer — tracing must
    /// not trigger a full norm pass.
    fn peek_down_row_norm(&self, layer: usize, feature: usize) -> Option<f32> {
        self.down_norms_cache
            .borrow()
            .get(layer)
            .and_then(|slot| slot.as_ref().map(|norms| norms.get(feature).copied()))
            .flatten()
    }

    /// Fold one executed layer call into the runtime trace. Called by
    /// `forward_routed` after the routing ladder returns, so the
    /// observation and `trace_path` name describe the path that ran.
    pub(super) fn fold_runtime_trace(
        &self,
        layer: usize,
        out: &Array2<f32>,
        obs: Option<&FfnActivations>,
        plan: &FfnPlan,
    ) {
        let plan_reason = plan.reason().summary();
        let mut sink = self.runtime_trace.borrow_mut();
        for s in 0..out.shape()[0] {
            let residual_delta_norm = out.row(s).iter().map(|v| v * v).sum::<f32>().sqrt();
            let (path, features) = match obs {
                Some(FfnActivations::Sparse(sa)) => {
                    let features = sa
                        .position(s)
                        .iter()
                        .enumerate()
                        .map(|(rank, e)| TracedFeature {
                            feature: e.feature,
                            rank,
                            gate_score: e.gate_score,
                            up_score: e.up_score,
                            activation: e.activation,
                            down_row_norm: self.peek_down_row_norm(layer, e.feature),
                        })
                        .collect();
                    (
                        sa.kernel(s).unwrap_or_else(|| self.last_path.get()),
                        features,
                    )
                }
                // Dense / absent observations carry no per-feature
                // selection data — emit the layer summary only.
                _ => (self.last_path.get(), Vec::new()),
            };
            sink.push(LayerTraceRecord {
                layer,
                position: s,
                path,
                residual_delta_norm,
                plan_reason: plan_reason.clone(),
                features,
            });
        }
    }

    /// Drain the full per-(position, layer) runtime trace records.
    /// Empty unless the trace was enabled (`with_trace`).
    pub fn take_runtime_trace(&self) -> Vec<LayerTraceRecord> {
        std::mem::take(&mut *self.runtime_trace.borrow_mut())
    }

    /// Drain the trace as the public [`WalkTrace`] shape.
    ///
    /// The hits are the features the walk EXECUTED, carrying the
    /// kernels' own values — not a post-hoc `gate_knn` re-run (item
    /// 17). Per layer call, the LAST position's record is reported
    /// (matching the historic last-row semantics); features without
    /// `FeatureMeta` are dropped from this view (`WalkHit.meta` is
    /// mandatory) — use [`WalkFfn::take_runtime_trace`] for full
    /// fidelity. Also drains the captured residuals, preserving the
    /// old contract that `take_trace` consumes the trace.
    pub fn take_trace(&self) -> WalkTrace {
        self.trace_residuals.borrow_mut().clear();
        let records = self.take_runtime_trace();

        // Group records into per-layer-call runs. Within one forward
        // call every record shares `layer` with ascending `position`,
        // so a layer change or a position reset marks a new call.
        let mut layers: Vec<(usize, Vec<WalkHit>)> = Vec::new();
        let mut current: Option<&LayerTraceRecord> = None;
        let flush = |rec: Option<&LayerTraceRecord>, layers: &mut Vec<_>| {
            if let Some(rec) = rec {
                let hits: Vec<WalkHit> = rec
                    .features
                    .iter()
                    .filter_map(|f| {
                        let meta = self.index.feature_meta(rec.layer, f.feature)?;
                        Some(WalkHit {
                            layer: rec.layer,
                            feature: f.feature,
                            // 0.0 only when the executed path had no
                            // score for the feature — the honest
                            // `Option` lives in the runtime record.
                            gate_score: f.gate_score.unwrap_or(0.0),
                            meta,
                            up_score: f.up_score,
                            activation: Some(f.activation),
                            down_row_norm: f.down_row_norm,
                            rank: Some(f.rank),
                        })
                    })
                    .collect();
                layers.push((rec.layer, hits));
            }
        };
        for rec in &records {
            let new_call = match current {
                Some(prev) => prev.layer != rec.layer || rec.position <= prev.position,
                None => true,
            };
            if new_call {
                flush(current, &mut layers);
            }
            current = Some(rec);
        }
        flush(current, &mut layers);
        WalkTrace { layers }
    }
}
