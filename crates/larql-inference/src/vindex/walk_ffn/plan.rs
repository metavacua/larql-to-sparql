//! Explicit execution plan for the walk-FFN routing ladder
//! (2026-07-30 review, item 18).
//!
//! [`WalkFfn::plan_for`](super::WalkFfn::plan_for) (`planner.rs`) turns
//! the routing ladder's condition checks into a value: WHICH path will
//! run and WHY — the rung that fired, every higher-priority rung that
//! did not (with the reason it was passed over), and the threshold
//! comparisons knowable before execution. `forward` consumes the same
//! planner (`forward_ladder` in `mod.rs`), so the inspection surface
//! and the executed routing cannot drift.
//!
//! The plan is the STATIC decision. Some paths can still decline
//! mid-execution (return `None` — e.g. a storage flag is set but the
//! layer's bytes are missing): the executor then re-plans with the
//! declined rung excluded, and the re-planned reason's `skipped` list
//! records the decline explicitly. Declines are re-planned, never
//! hidden.
//!
//! Numeric cutoffs referenced by [`ThresholdCheck`] live in
//! `thresholds.rs`; surfacing them into `WalkFfnConfig` is a tracked
//! follow-up of the same review item.

/// Identifies one rung of the routing ladder — one possible execution
/// destination. Ladder priority is the declaration order here; names
/// (via [`PlanRung::name`]) match the ladder-level `trace_path`
/// vocabulary the dispatch tests assert on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanRung {
    /// Rung 0 — no vindex features: dense safetensors `WeightFfn`.
    ZeroFeaturesDense,
    /// Rung 1a — overridden layer, exact base+delta execution.
    OverrideBaseDelta,
    /// Rung 1b — overridden layer, override-aware sparse walk.
    OverrideSparse,
    /// L1 output-cache hit (single-position, hot calls only).
    L1CacheHit,
    /// Rung 2 — explicit per-layer sparse K.
    Sparse,
    /// Rung 3 — FP4/FP8 storage, routed through the sparse walk.
    Fp4Sparse,
    /// Rung 4 — K-quant direct matvec, no dequant cache.
    KquantNative,
    /// Rung 5 — Q4_0 interleaved slab.
    InterleavedQ4,
    /// Rung 6 — f32 interleaved mmap.
    InterleavedF32,
    /// Rung 7 — gate/up/down in three separate mmap files.
    FullMmap,
    /// Rung 8 — K-quant whole-layer dequant fallback.
    KquantDequant,
    /// Rung 9 — down from mmap, gate/up from safetensors.
    Exact,
    /// Rung 10 — last-resort sparse matmul against safetensors weights.
    WeightsFallback,
}

impl PlanRung {
    /// Ladder-level path name — the `trace_path` string family this
    /// rung emits on success (sub-kernel specialisations such as
    /// `sparse:serial` / `weights_fallback:override` extend these at
    /// runtime).
    pub fn name(self) -> &'static str {
        match self {
            Self::ZeroFeaturesDense => "zero_features_dense",
            Self::OverrideBaseDelta => "override:base_delta",
            Self::OverrideSparse => "override:sparse",
            Self::L1CacheHit => "l1_cache_hit",
            Self::Sparse => "sparse",
            Self::Fp4Sparse => "fp4_storage:sparse",
            Self::KquantNative => "interleaved_kquant:native",
            Self::InterleavedQ4 => "interleaved_q4",
            Self::InterleavedF32 => "interleaved",
            Self::FullMmap => "full_mmap",
            Self::KquantDequant => "interleaved_kquant:dequant",
            Self::Exact => "exact",
            Self::WeightsFallback => "weights_fallback",
        }
    }

    /// Bit for [`PlanExclusions`] membership.
    const fn bit(self) -> u16 {
        1 << self as u16
    }
}

/// Set of rungs the executor has already tried and seen decline —
/// re-planning input. Also the honest record that a decline happened:
/// the planner marks excluded rungs `skipped` with
/// [`super::planner::BECAUSE_DECLINED`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PlanExclusions(u16);

impl PlanExclusions {
    pub(super) fn contains(self, rung: PlanRung) -> bool {
        self.0 & rung.bit() != 0
    }

    #[must_use]
    pub(super) fn with(self, rung: PlanRung) -> Self {
        Self(self.0 | rung.bit())
    }
}

/// One pre-execution threshold comparison that shaped the plan. The
/// named cutoffs live in `thresholds.rs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThresholdCheck {
    /// Name of the cutoff constant, e.g. `"FULL_K_DENSITY"`.
    pub name: &'static str,
    /// The value compared against the cutoff (e.g. requested K).
    pub actual: usize,
    /// The computed cutoff for this layer.
    pub cutoff: usize,
    /// Whether the comparison fired.
    pub satisfied: bool,
}

/// A higher-priority rung that did NOT fire, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedRung {
    pub rung: PlanRung,
    pub because: &'static str,
}

/// Why a plan chose its rung — the structured record every [`FfnPlan`]
/// variant carries.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanReason {
    pub layer: usize,
    pub seq_len: usize,
    /// `GateIndex::num_features(layer)` at plan time.
    pub num_features: usize,
    /// `GateIndex::has_overrides_at(layer)` at plan time.
    pub has_overrides: bool,
    /// The condition that selected this rung.
    pub selected: &'static str,
    /// Every rung above the selected one in ladder priority, with the
    /// reason it was passed over — including runtime declines
    /// (re-planned rungs appear here as "declined at execution").
    pub skipped: Vec<SkippedRung>,
    /// Pre-execution threshold comparisons (empty when none apply).
    pub thresholds: Vec<ThresholdCheck>,
}

impl PlanReason {
    /// One-line human-readable rendering — the string the runtime
    /// trace records alongside the executed path name.
    pub fn summary(&self) -> String {
        use std::fmt::Write as _;
        let mut s = format!(
            "layer={} seq_len={} num_features={} has_overrides={} — {}",
            self.layer, self.seq_len, self.num_features, self.has_overrides, self.selected
        );
        for t in &self.thresholds {
            let _ = write!(
                s,
                "; {} actual={} cutoff={} satisfied={}",
                t.name, t.actual, t.cutoff, t.satisfied
            );
        }
        if !self.skipped.is_empty() {
            s.push_str("; skipped:");
            for sk in &self.skipped {
                let _ = write!(s, " {} ({})", sk.rung.name(), sk.because);
            }
        }
        s
    }
}

/// The routing decision for one `(layer, x)` call — one variant per
/// ladder destination, each carrying the [`PlanReason`] that selected
/// it. Produced by [`WalkFfn::plan_for`](super::WalkFfn::plan_for)
/// for inspection and by the internal planner for execution.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnPlan {
    ZeroFeaturesDense(PlanReason),
    OverrideBaseDelta(PlanReason),
    OverrideSparse(PlanReason),
    L1CacheHit(PlanReason),
    Sparse(PlanReason),
    Fp4Sparse(PlanReason),
    KquantNative(PlanReason),
    InterleavedQ4(PlanReason),
    InterleavedF32(PlanReason),
    FullMmap(PlanReason),
    KquantDequant(PlanReason),
    Exact(PlanReason),
    WeightsFallback(PlanReason),
}

impl FfnPlan {
    /// Wrap a reason in the variant for `rung`.
    pub(super) fn from_rung(rung: PlanRung, reason: PlanReason) -> Self {
        match rung {
            PlanRung::ZeroFeaturesDense => Self::ZeroFeaturesDense(reason),
            PlanRung::OverrideBaseDelta => Self::OverrideBaseDelta(reason),
            PlanRung::OverrideSparse => Self::OverrideSparse(reason),
            PlanRung::L1CacheHit => Self::L1CacheHit(reason),
            PlanRung::Sparse => Self::Sparse(reason),
            PlanRung::Fp4Sparse => Self::Fp4Sparse(reason),
            PlanRung::KquantNative => Self::KquantNative(reason),
            PlanRung::InterleavedQ4 => Self::InterleavedQ4(reason),
            PlanRung::InterleavedF32 => Self::InterleavedF32(reason),
            PlanRung::FullMmap => Self::FullMmap(reason),
            PlanRung::KquantDequant => Self::KquantDequant(reason),
            PlanRung::Exact => Self::Exact(reason),
            PlanRung::WeightsFallback => Self::WeightsFallback(reason),
        }
    }

    /// Which ladder rung this plan targets.
    pub fn rung(&self) -> PlanRung {
        match self {
            Self::ZeroFeaturesDense(_) => PlanRung::ZeroFeaturesDense,
            Self::OverrideBaseDelta(_) => PlanRung::OverrideBaseDelta,
            Self::OverrideSparse(_) => PlanRung::OverrideSparse,
            Self::L1CacheHit(_) => PlanRung::L1CacheHit,
            Self::Sparse(_) => PlanRung::Sparse,
            Self::Fp4Sparse(_) => PlanRung::Fp4Sparse,
            Self::KquantNative(_) => PlanRung::KquantNative,
            Self::InterleavedQ4(_) => PlanRung::InterleavedQ4,
            Self::InterleavedF32(_) => PlanRung::InterleavedF32,
            Self::FullMmap(_) => PlanRung::FullMmap,
            Self::KquantDequant(_) => PlanRung::KquantDequant,
            Self::Exact(_) => PlanRung::Exact,
            Self::WeightsFallback(_) => PlanRung::WeightsFallback,
        }
    }

    /// The structured reason, regardless of variant.
    pub fn reason(&self) -> &PlanReason {
        match self {
            Self::ZeroFeaturesDense(r)
            | Self::OverrideBaseDelta(r)
            | Self::OverrideSparse(r)
            | Self::L1CacheHit(r)
            | Self::Sparse(r)
            | Self::Fp4Sparse(r)
            | Self::KquantNative(r)
            | Self::InterleavedQ4(r)
            | Self::InterleavedF32(r)
            | Self::FullMmap(r)
            | Self::KquantDequant(r)
            | Self::Exact(r)
            | Self::WeightsFallback(r) => r,
        }
    }
}
