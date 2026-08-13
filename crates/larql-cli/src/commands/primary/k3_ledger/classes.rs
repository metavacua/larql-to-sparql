//! Per-tensor-class byte census and composed throughput ceiling.
//!
//! Pure. Every ceiling quoted before this module existed divided total bytes by
//! a single `eta`, which is wrong in a way that flatters: `eta` is a property of
//! a *kernel*, and different tensor classes run through different kernels with
//! measured efficiencies from 0.59 to 1.00. K3's attention projections are the
//! largest dense term AND run through the worst-measured class, so a scalar
//! quote overstates by ~1.17x.
//!
//! The composed form is a sum of per-class times, not a single division:
//!
//! ```text
//!   t_token = sum_c  bytes_c / (BW * eta_c)
//!   tok/s   = 1 / t_token
//! ```
//!
//! This module also provides the R4 discipline in scenario form: for a proposed
//! change, zero out or improve exactly what it targets and recompute. A lever
//! whose best case still sits under the goal cannot decide the goal, and finding
//! that out costs one division rather than a fortnight.

use serde::Serialize;

use super::geometry::K3Geometry;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Measured attainable GPU bandwidth (`project_memory_bandwidth_roofline`).
pub const BW_GB_S: f64 = 367.0;

/// A kernel class, with the efficiency measured for it by the batched profiler.
///
/// **MEASURED at K3 shapes under Q6K** as of `diag_shape_census`, which
/// resolved a confound worth recording. The composed ledger previously assigned
/// `q4k_matvec`'s 0.59 to the attention class — but crossing kernel against
/// shape showed that figure is **kernel-borne, not shape-borne**: `q6k_matvec`
/// reaches 0.87-0.90 at *every* large shape including K3's `[12288, 7168]`
/// attention, while `q4k_matvec` sits at 0.65-0.75 on the same geometry. So a
/// Q6K-transcoded image sidesteps the Wo problem entirely and attention runs at
/// 0.89, not 0.59.
///
/// The same run found the opposite effect where nobody was looking: K3's expert
/// branch `[3584, 3072]` gets only **0.64** from the identical q6k kernel. That
/// one *is* shape-borne — 896 threadgroups versus attention's 3072 leaves too
/// little work to hide latency — and it lands on the single largest byte term,
/// so routed experts now consume over half the time budget.
///
/// Historical note on the superseded figures. They came from
/// `diag_profile_kernels` batched at 34x per command buffer, which is the right
/// instrument — isolated single-dispatch numbers for the same kernels run
/// 3.0-6.7x lower and must never be substituted. But they were measured:
///
///   1. on **Gemma-class shapes** (2560x8192, 2560x10240, 10240x2560), not K3's
///      ([12288, 7168] attention, [3584, 3072] expert branches); and
///   2. in **Q4K / Q6K**, not in whatever format the image actually serves.
///
/// So applying them to a K3 census conflates tensor shape with kernel format
/// twice over. A ceiling computed from them is a *provisional composed ceiling*
/// and must be labelled as such until the real K3 shapes run through the same
/// batched, cold-rotating harness. `Eta::provenance` carries that condition so
/// it cannot be quoted bare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum KernelClass {
    /// `q4k_matvec` (Wo-shaped). Worst measured.
    AttnProjection,
    /// `q4k_ffn_gate_up_8sg`. Compute-bound on dequant at K=2560.
    GateUp,
    /// `q6k_matvec` at a large shape. Bandwidth-bound, best measured class.
    Down,
    /// `q6k_matvec` at K3's small expert-branch shape `[3584, 3072]`. The same
    /// kernel, measurably worse purely from geometry: 896 threadgroups against
    /// the attention shape's 3072, so there is far less work to hide latency
    /// behind. This is a SHAPE-borne penalty on the largest byte term.
    RoutedExpert,
    /// `f32_gemv`. Un-quantised, reaches the roofline.
    Unquantised,
}

/// Repeats below which no amount of apparent tightness counts.
pub const MIN_REPEATS: u32 = 5;
/// Relative standard error a class must reach to select a target.
pub const MAX_RELATIVE_SE: f64 = 0.01;

/// The measurement regime an efficiency was taken under. Not decoration: the
/// same kernel reads very differently from a buffer that is partly L2-resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Regime {
    /// Distinct weight buffers per call, working set far beyond L2.
    ColdRotating,
    /// Not a measurement at all — a modelling constant with no variance, such
    /// as unquantised f32 reaching the roofline by definition. Cannot swing, so
    /// it cannot select a bit-width, so the noise refusal does not apply.
    Definitional,
}

/// A measured efficiency **with its observed spread**, never a bare scalar.
///
/// Exists because the cold-rotate harness bug removed a false precision as well
/// as a bias. Once the rotation genuinely rotated, the small shapes moved
/// 10-12% across repeats while attention moved 2%. The composed total is stable
/// (attention dominates and is the steady term) but **every per-class decision
/// reads the noisy inputs directly** — which class to compress, the frontier's
/// target row, the serving-format choice. Aggregate robustness does not license
/// per-class precision, so the range travels with the number.
///
/// `observed_low`/`observed_high` are the **observed extremes**, not a standard
/// error. They are deliberately not summarised into `+/- x`: with three repeats
/// a spread cannot distinguish gaussian noise from two modes (page alignment,
/// thermal state, allocator luck), and reporting a symmetric interval would
/// claim it can.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MeasuredEfficiency {
    pub central: f64,
    pub observed_low: f64,
    pub observed_high: f64,
    /// Standard error of the mean. **This, not the spread, decides whether a
    /// class may select a target.** Observed extremes are the honest thing to
    /// *display*, but `max - min` is non-decreasing in sample count, so gating
    /// on it rewards collecting less data — measured directly: attention showed
    /// 2% spread over 3 repeats and 5.7% over 7, while its SE fell.
    pub std_error: f64,
    pub repeats: u32,
    pub regime: Regime,
}

impl MeasuredEfficiency {
    pub const fn new(
        central: f64,
        lo: f64,
        hi: f64,
        std_error: f64,
        repeats: u32,
        regime: Regime,
    ) -> Self {
        Self {
            central,
            observed_low: lo,
            observed_high: hi,
            std_error,
            repeats,
            regime,
        }
    }
    /// A modelling constant with no variance.
    pub const fn definitional(central: f64) -> Self {
        Self::new(central, central, central, 0.0, 0, Regime::Definitional)
    }

    /// Standard error as a fraction of the central estimate.
    pub fn relative_std_error(&self) -> f64 {
        if self.central == 0.0 {
            return f64::INFINITY;
        }
        self.std_error / self.central
    }
    /// Widest observed relative swing, as a fraction of `central`.
    pub fn spread_fraction(&self) -> f64 {
        if self.central == 0.0 {
            return 0.0;
        }
        (self.observed_high - self.observed_low) / self.central
    }
    /// True when the class is settled enough to select a target.
    ///
    /// A definitional constant qualifies with no repeats — it has no variance
    /// to be uncertain about. Everything else needs three repeats inside 5%.
    pub fn is_decision_grade(&self) -> bool {
        self.regime == Regime::Definitional
            || (self.repeats >= MIN_REPEATS && self.relative_std_error() <= MAX_RELATIVE_SE)
    }
}

impl KernelClass {
    /// What was actually measured, and under what conditions.
    pub fn provenance(self) -> &'static str {
        match self {
            Self::AttnProjection => "q6k_matvec, Q6K, measured at K3 KDA 12288x7168",
            Self::GateUp => "q6k_matvec, Q6K, measured at K3 shared expert 6144x7168",
            Self::Down => "q6k_matvec, Q6K, measured at K3 latent-up 7168x3584",
            Self::RoutedExpert => {
                "q6k_matvec ungrouped, K3 expert 3584x3072 (K3a grouped measures 0.89)"
            }
            Self::Unquantised => "f32_gemv lm_head 262Kx2560, f32, Gemma shape",
        }
    }

    /// True while this class's eta is a borrowed proxy rather than a K3-shape,
    /// K3-format measurement.
    ///
    /// Nothing is, since the control-gated campaign measured `[6144, 7168]`
    /// directly. `GateUp` was the last borrowed figure; it now carries its own
    /// number and fails on precision instead, which `is_decision_grade` reports.
    pub fn is_provisional(self) -> bool {
        false
    }

    /// Banked efficiency with its observed range.
    ///
    /// Banked 2026-08-01 from a 16-run campaign, **9 runs rejected** by a
    /// within-run control. `diag_eta_repeats` watches the attention cell, which
    /// is large and steady; across the campaign it collapsed from 0.89 to 0.06
    /// as the machine degraded under back-to-back load. Averaging all 16 would
    /// have banked attention at 0.52 and the down class at 0.53 — wrong by a
    /// factor, and it would have looked like more data.
    ///
    /// Runs are therefore NOT exchangeable samples on this hardware; they are a
    /// time series with a hard degradation after roughly seven censuses. Any
    /// future campaign needs the control gate, not more repeats.
    pub fn efficiency(self) -> MeasuredEfficiency {
        use Regime::ColdRotating as C;
        match self {
            Self::AttnProjection => MeasuredEfficiency::new(0.876, 0.85, 0.90, 0.0075, 7, C),
            Self::GateUp => MeasuredEfficiency::new(0.799, 0.70, 0.84, 0.0177, 7, C),
            Self::Down => MeasuredEfficiency::new(0.784, 0.76, 0.80, 0.0057, 7, C),
            Self::RoutedExpert => MeasuredEfficiency::new(0.614, 0.54, 0.67, 0.0145, 7, C),
            Self::Unquantised => MeasuredEfficiency::definitional(1.00),
        }
    }

    /// Central estimate. Use for ordinary reports; propagate the range through
    /// anything that selects a target.
    pub fn eta(self) -> f64 {
        self.efficiency().central
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AttnProjection => "attn-projection (Wo class)",
            Self::GateUp => "gate/up class",
            Self::Down => "down class",
            Self::RoutedExpert => "routed-expert shape (small)",
            Self::Unquantised => "unquantised",
        }
    }
}

/// One row of the census: a group of tensors sharing a kernel class and format.
#[derive(Debug, Clone, Serialize)]
pub struct ClassRow {
    pub name: &'static str,
    pub params: u64,
    pub class: KernelClass,
    /// All-in bits per weight for this class in the current image.
    pub bits: f64,
    /// Efficiency override, when a scenario improves this class specifically.
    pub eta_override: Option<f64>,
}

impl ClassRow {
    pub fn bytes(&self) -> f64 {
        self.params as f64 * self.bits / 8.0
    }
    pub fn eta(&self) -> f64 {
        self.eta_override.unwrap_or_else(|| self.class.eta())
    }
    pub fn seconds(&self, bw: f64) -> f64 {
        self.bytes() / (bw * 1e9 * self.eta())
    }
}

/// Efficiency the routed-expert class reaches once experts are put on the
/// dispatch grid's y axis — MEASURED by K3a, and inside the large-shape band.
pub const GROUPED_ROUTED_ETA: f64 = 0.89;

// All-in bits per container are NOT restated here. `Container::all_in_bits`
// derives them from block geometry and is the single authority; a second copy
// carrying the same literals is how the two drift apart, and the ledger's whole
// claim is that its numbers are derived rather than typed.

/// The per-class census for a K3 image at a given format.
///
/// Parameter counts are derived from the measured geometry rather than assumed:
/// a KDA layer carries five `[12288, 7168]` attention projections, an MLA layer
/// replaces three of them with the smaller q_a/q_b/kv_a/kv_b set, and every
/// layer carries shared experts, the LatentMoE projections and a router.
pub fn census(geom: &K3Geometry, dense_bits: f64, routed_bits: f64) -> Vec<ClassRow> {
    let h = geom.hidden_size as u64;
    let kda = geom.n_kda_layers as u64;
    let mla = geom.n_mla_layers as u64;
    let moe = geom.n_moe_layers() as u64;

    // KDA attention: q, k, v, g, o each [12288, 7168].
    let kda_attn = kda * 5 * 12288 * h;
    // MLA attention: g + o at full width, plus the compressed q/kv set.
    let mla_attn = mla * (2 * 12288 * h + 1536 * h + 18432 * 1536 + 576 * h + 24576 * 512);
    // Shared experts: gate, up, down at [6144, 7168].
    let shared = moe * 3 * 6144 * h;
    // LatentMoE wrapper: down + up at [3584, 7168].
    let latent = moe * 2 * 3584 * h;
    let router = moe * geom.n_experts as u64 * h;

    vec![
        ClassRow {
            name: "KDA attention projections",
            params: kda_attn,
            class: KernelClass::AttnProjection,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "MLA attention projections",
            params: mla_attn,
            class: KernelClass::AttnProjection,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "shared experts",
            params: shared,
            class: KernelClass::GateUp,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "LatentMoE projections",
            params: latent,
            class: KernelClass::Down,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "router",
            params: router,
            class: KernelClass::Down,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "embeddings + lm_head",
            params: geom.embedding_params,
            class: KernelClass::Unquantised,
            bits: 16.0,
            eta_override: None,
        },
        ClassRow {
            name: "routed experts (top-k)",
            params: geom.routed_activated_params(),
            class: KernelClass::RoutedExpert,
            bits: routed_bits,
            eta_override: None,
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
pub struct ComposedCeiling {
    pub total_bytes: f64,
    pub seconds_per_token: f64,
    pub tok_s: f64,
    /// What a single-eta quote would have claimed, and by how much it overstates.
    pub scalar_tok_s: f64,
    pub scalar_overstates_by: f64,
    pub rows: Vec<ClassRow>,
}

/// All-in bits per weight the banked efficiencies were **measured under**.
///
/// R7: eta is meaningful only within a fixed storage representation. Every
/// figure in `KernelClass::efficiency` came from `q6k_matvec` reading Q6_K
/// bytes. Recomposing them at another density holds eta fixed across a
/// container change, which is exactly what R7 forbids as a performance claim —
/// the crossover measured the penalty directly (MXFP4 arm D 0.724 against
/// Q6_K 0.89 on the routed class).
pub const ETA_MEASURED_UNDER_BITS: f64 = 210.0 * 8.0 / 256.0;

/// Does this composition reuse the banked etas outside the container they were
/// measured in? If so the result is a **density-only upper bound**, not a
/// reachable rate.
pub fn density_only_bound(dense_bits: f64, routed_bits: f64) -> bool {
    (dense_bits - ETA_MEASURED_UNDER_BITS).abs() > 1e-9
        || (routed_bits - ETA_MEASURED_UNDER_BITS).abs() > 1e-9
}

/// Classes carrying bytes whose efficiency is too noisy to select a target.
///
/// Returned rather than warned about: the house precedent is a bench that exits
/// instead of handing the frontier a number it cannot stand behind. A warning
/// beside a plausible float gets read past; a refusal does not.
pub fn not_decision_grade(rows: &[ClassRow]) -> Vec<(KernelClass, MeasuredEfficiency)> {
    let mut seen: Vec<KernelClass> = Vec::new();
    let mut out = Vec::new();
    for r in rows {
        if r.bits <= 0.0 || r.eta_override.is_some() || seen.contains(&r.class) {
            continue;
        }
        seen.push(r.class);
        let m = r.class.efficiency();
        if !m.is_decision_grade() {
            out.push((r.class, m));
        }
    }
    out
}

/// Control-passing runs in the banked campaign.
pub const ACCEPTED_RUNS: usize = 7;

/// Per-run efficiency samples, **paired by run index** across classes.
///
/// Paired, because every class moved together: the campaign's degradation was
/// common-mode. Combining each class's independent minimum into one low bound
/// and each maximum into one high bound produces a *component-extrema
/// envelope*, which is 1.18x wider than anything actually observed and
/// describes a run that never happened. Composing run-by-run and reporting the
/// min/median/max of those compositions reports only real machine states.
pub fn samples(class: KernelClass) -> [f64; ACCEPTED_RUNS] {
    match class {
        KernelClass::AttnProjection => [0.89, 0.88, 0.90, 0.89, 0.85, 0.87, 0.85],
        KernelClass::GateUp => [0.79, 0.83, 0.84, 0.81, 0.80, 0.82, 0.70],
        KernelClass::Down => [0.79, 0.78, 0.80, 0.80, 0.76, 0.79, 0.77],
        KernelClass::RoutedExpert => [0.67, 0.62, 0.62, 0.61, 0.54, 0.62, 0.62],
        KernelClass::Unquantised => [1.0; ACCEPTED_RUNS],
    }
}

/// Compose the ledger once per accepted run, preserving the pairing.
///
/// A row with an `eta_override` (a scenario) holds that value across all runs —
/// the scenario asserts an efficiency rather than having measured one.
pub fn compose_observed(rows: &[ClassRow], bw: f64) -> Vec<f64> {
    (0..ACCEPTED_RUNS)
        .map(|run| {
            let secs: f64 = rows
                .iter()
                .map(|r| {
                    let e = r.eta_override.unwrap_or_else(|| samples(r.class)[run]);
                    r.bytes() / (bw * 1e9 * e)
                })
                .sum();
            1.0 / secs
        })
        .collect()
}

/// Min / max of the per-run compositions — an OBSERVED range, not an envelope.
pub fn observed_range(rows: &[ClassRow], bw: f64) -> (f64, f64) {
    let c = compose_observed(rows, bw);
    (
        c.iter().cloned().fold(f64::INFINITY, f64::min),
        c.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Compose per-class times into one throughput ceiling.
pub fn compose(rows: Vec<ClassRow>, bw: f64, scalar_eta: f64) -> ComposedCeiling {
    let total: f64 = rows.iter().map(ClassRow::bytes).sum();
    let secs: f64 = rows.iter().map(|r| r.seconds(bw)).sum();
    let scalar = bw * 1e9 * scalar_eta / total;
    ComposedCeiling {
        total_bytes: total,
        seconds_per_token: secs,
        tok_s: 1.0 / secs,
        scalar_tok_s: scalar,
        scalar_overstates_by: scalar * secs,
        rows,
    }
}

/// **R4 in scenario form.** Apply a best-case change and recompute the ceiling.
///
/// Each variant expresses what one proposed experiment would achieve *if it
/// fully succeeded* — not a forecast, a ceiling. Anything whose best case sits
/// under the goal cannot decide the goal.
#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    /// Every class runs at the best measured quantised efficiency.
    AllClassesAtBestEta,
    /// One class's efficiency is lifted to a target.
    LiftClass(KernelClass, f64),
    /// One named row's format changes to a new bit width.
    Rebits(&'static str, f64),
    /// A named row's bytes vanish entirely — the strict R4 zero-out.
    Zero(&'static str),
}

/// Apply several scenarios in sequence — the stacked best case.
///
/// Single levers are informative but no proposal ships alone; the programme
/// question is what a *combination* reaches, and whether the combination clears
/// the target that any one of them misses.
pub fn apply_all(rows: &[ClassRow], scenarios: &[Scenario]) -> Vec<ClassRow> {
    scenarios
        .iter()
        .fold(rows.to_vec(), |acc, s| apply(&acc, *s))
}

pub fn apply(rows: &[ClassRow], s: Scenario) -> Vec<ClassRow> {
    rows.iter()
        .map(|r| {
            let mut r = r.clone();
            match s {
                Scenario::AllClassesAtBestEta => {
                    if r.class != KernelClass::Unquantised {
                        r.eta_override = Some(KernelClass::Down.eta());
                    }
                }
                Scenario::LiftClass(c, eta) => {
                    if r.class == c {
                        r.eta_override = Some(eta);
                    }
                }
                Scenario::Rebits(name, bits) => {
                    if r.name == name {
                        r.bits = bits;
                    }
                }
                Scenario::Zero(name) => {
                    if r.name == name {
                        r.bits = 0.0;
                    }
                }
            }
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;
    use crate::commands::primary::k3_ledger::serving_format::Container;

    fn base() -> Vec<ClassRow> {
        let bits = Container::Mxfp4.all_in_bits();
        census(&k3_reference(), bits, bits)
    }

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn census_reproduces_the_measured_activated_total() {
        // Independent path to the same 104.83 B figure the ledger derives from
        // per-layer headers — a cross-check that the class split is complete.
        let total: u64 = base().iter().map(|r| r.params).sum();
        approx(total as f64 / 1e9, 104.83, 1.5);
    }

    #[test]
    fn attention_is_the_largest_dense_class() {
        let rows = base();
        let attn: u64 = rows
            .iter()
            .filter(|r| r.class == KernelClass::AttnProjection)
            .map(|r| r.params)
            .sum();
        let others: u64 = rows
            .iter()
            .filter(|r| {
                r.class != KernelClass::AttnProjection && r.name != "routed experts (top-k)"
            })
            .map(|r| r.params)
            .sum();
        assert!(attn > others, "attention {attn} vs other dense {others}");
    }

    #[test]
    fn composing_is_stricter_than_a_scalar_quote() {
        // A single eta flatters whenever any class sits below it, and one
        // always does. Quoting the best-class eta over total bytes is the
        // specific mistake this module exists to prevent.
        let c = compose(base(), BW_GB_S, KernelClass::Down.eta());
        assert!(c.scalar_overstates_by > 1.0, "{}", c.scalar_overstates_by);
        assert!(c.tok_s < c.scalar_tok_s);
    }

    #[test]
    fn the_routed_expert_shape_is_now_the_worst_class() {
        // Reordered by measurement: before diag_shape_census the attention
        // class held 0.59 and routed experts a borrowed 0.85. Crossing kernel
        // against shape moved attention to 0.89 (its 0.59 was q4k's problem,
        // not the geometry's) and routed experts DOWN to 0.64 (896 threadgroups
        // against attention's 3072). The worst class changed hands.
        assert!(KernelClass::RoutedExpert.eta() < KernelClass::AttnProjection.eta());
        assert!(KernelClass::RoutedExpert.eta() < KernelClass::Down.eta());
    }

    #[test]
    fn fixing_the_routed_expert_shape_is_the_largest_single_lever() {
        // Consequence of the reordering: routed experts carry over half the
        // time budget at the worst eta, so lifting THAT class beats lifting any
        // other. This test is the programme's ordering, made executable.
        let rows = base();
        let ceiling = |c: KernelClass| {
            compose(apply(&rows, Scenario::LiftClass(c, 0.89)), BW_GB_S, 0.87).tok_s
        };
        let routed = ceiling(KernelClass::RoutedExpert);
        for other in [
            KernelClass::AttnProjection,
            KernelClass::GateUp,
            KernelClass::Down,
        ] {
            assert!(
                routed > ceiling(other),
                "lifting routed ({routed}) should beat lifting {:?} ({})",
                other,
                ceiling(other)
            );
        }
    }

    #[test]
    fn routed_experts_dominate_the_time_budget() {
        let c = compose(base(), BW_GB_S, 0.87);
        let routed = c
            .rows
            .iter()
            .find(|r| r.name == "routed experts (top-k)")
            .unwrap()
            .seconds(BW_GB_S);
        assert!(
            routed / c.seconds_per_token > 0.5,
            "routed share {:.2} should exceed half",
            routed / c.seconds_per_token
        );
    }

    #[test]
    fn zeroing_the_routed_bank_leaves_the_dense_ceiling() {
        // R4: no expert-side lever can lift a target above this.
        let c = compose(
            apply(&base(), Scenario::Zero("routed experts (top-k)")),
            BW_GB_S,
            0.85,
        );
        assert!(
            c.tok_s < 20.0,
            "dense-only ceiling {} should refuse 20",
            c.tok_s
        );
    }

    #[test]
    fn q6k_transcode_costs_bytes_and_therefore_throughput() {
        let mxfp4 = compose(base(), BW_GB_S, 0.85);
        let q6k_bits = Container::Q6K.all_in_bits();
        let q6k = compose(census(&k3_reference(), q6k_bits, q6k_bits), BW_GB_S, 0.85);
        assert!(q6k.total_bytes > mxfp4.total_bytes);
        assert!(q6k.tok_s < mxfp4.tok_s);
        approx(q6k_bits / Container::Mxfp4.all_in_bits(), 1.544, 0.01);
    }

    #[test]
    fn three_bit_kda_attention_is_a_real_lever() {
        // Exp 3's target: the largest dense class at 3 bits instead of 4.25.
        let rows = base();
        let before = compose(rows.clone(), BW_GB_S, 0.85);
        let after = compose(
            apply(&rows, Scenario::Rebits("KDA attention projections", 3.0)),
            BW_GB_S,
            0.85,
        );
        assert!(after.tok_s > before.tok_s * 1.05, "should move the needle");
    }
}
