//! Frequency-mass coverage — what fraction of selection *events* a resident set
//! serves, as a function of its size. The instrument that grades the resident
//! bank (DEC-8.4), the static slice (DEC-8.6) and compact-dense (DEC-8.1).
//!
//! Support and mass are different questions and the ladder has conflated them
//! once already. Support asks *which* symbols a session touches; mass asks *how
//! often*. A session can touch the whole bank and still take most of its events
//! from a handful of symbols — only the second quantity prices a cache.
//!
//! ## The fair contest
//!
//! Every arm scores on the **same** held-out events. Arms differ only in the
//! key used to rank symbols for residency:
//!
//! ```text
//! key = lambda * pooled_prior(leave-one-out) + (1 - lambda) * session_fit
//! ```
//!
//! | arm | meaning |
//! |---|---|
//! | `lambda = 1` | static slice — pooled prior only, fitted without the scored session |
//! | `lambda = 0` | causal adaptive cache — this session's own history only |
//! | interior | shrinkage cache — the design that exists if neither pure arm wins |
//! | oracle | ranks by the eval events themselves; unrealisable upper bound |
//! | null | eval half redrawn from the pooled marginal, session identity destroyed |
//!
//! The null is load-bearing, not decoration. A short fit window concentrates by
//! counting noise alone, so `oracle - null` is the locality signal and `oracle`
//! on its own overstates it. Scoring adaptation without the null is how a
//! finite-sample artifact gets read as a 3x lever.
//!
//! ## "No layer class wins" is about ADAPTATION, not about structure
//!
//! On the DEC-0 Gemma pool the per-stratum spread of `causal - static` was
//! -0.012 +/- 0.015 with no layer where adaptation won. That sentence is quotable
//! and will be misquoted, so state its scope: it says **per-layer cache budgets
//! keyed on session adaptation are unsupported**. It says nothing about whether
//! layers differ statically — and they differ enormously.
//!
//! [`super::symbol_mass`] measured, on the same pool: unobserved experts per
//! layer ranging 3 to 68 of 128, and effective bank size (participation ratio)
//! from 14.8 to 50.5. Static per-layer heterogeneity is therefore **strongly
//! supported** by exactly the pool that refuted the adaptive form.
//!
//! **What varies is mostly how many experts get used, not how unevenly.** The
//! tempting headline is `corr(unobserved, top-8 coverage) = +0.861`, and it is
//! inflated: top-*k* at fixed `k` is not independent of support size — eight of
//! 60 observed is 13% of the support, eight of 125 is 6.4%, so a smaller-support
//! layer scores higher under an identical probability *shape*. Controlling for
//! that, three shape-only statistics agree at roughly half the strength:
//!
//! | statistic | corr with unobserved count |
//! |---|---|
//! | top-8 coverage (fixed count, confounded) | +0.861 |
//! | top-10% of support (fixed fraction) | +0.543 |
//! | normalised participation ratio | -0.501 |
//! | Gini over observed | +0.659 |
//!
//! So the shape effect is real, consistent in direction, and **moderate**: a
//! narrow-bank layer does use its survivors somewhat more unevenly, but not
//! dramatically (normalised PR spans only 0.207-0.436, mean 0.289).
//!
//! Call this a **narrow effective bank**, not router collapse. Collapse names a
//! mechanism — a few dominant experts running away with the mass — which would
//! drive normalised PR sharply down in the dead-set layers, and it does not.
//! L5 is a smaller bank used with roughly ordinary evenness. The distinction
//! sets the lever: prune the unused set, and do not expect a second win from
//! pruning within the survivors.
//!
//! Both statements are needed. A four-layer-class serving policy gets no support
//! in its adaptive form and strong support in its static form, and the two must
//! not be collapsed into one verdict on "per-layer anything".
//!
//! ## R2: this is the least transferable number in the set
//!
//! K3's Stable LatentMoE applies quantile balancing, which exists precisely to
//! flatten routing load. Per-layer tail heterogeneity is close to a direct
//! measurement of what QB is designed to remove — more so than the coverage
//! curve, which measures skew QB tolerates. Treat the spread as characterising
//! *this* model's routing, not as a K3 design input; the R2/R3 adapter rungs
//! supply the K3 version.

use serde::Serialize;

use super::rng::SplitMix64;
use super::selection_trace::SelectionTrace;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Deterministic default so a re-run reproduces the null exactly.
pub const DEFAULT_NULL_SEED: u64 = 20_260_801;

/// Fraction of each session's steps used to fit the adaptive key; the remainder
/// is scored. A cache cannot rank on events it has not seen.
pub const DEFAULT_FIT_FRACTION: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct MassCurveConfig {
    /// Resident-set sizes, in symbols per stratum.
    pub cache_sizes: Vec<usize>,
    /// Shrinkage weights on the pooled prior; 1.0 = static, 0.0 = causal.
    pub lambdas: Vec<f64>,
    pub fit_fraction: f64,
    pub seed: u64,
}

impl Default for MassCurveConfig {
    fn default() -> Self {
        Self {
            cache_sizes: vec![1, 2, 4, 8, 16, 32, 64],
            lambdas: vec![0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
            fit_fraction: DEFAULT_FIT_FRACTION,
            seed: DEFAULT_NULL_SEED,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ShrinkageArm {
    pub lambda: f64,
    /// Mean coverage at each configured cache size.
    pub coverage: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MassCurve {
    pub cache_sizes: Vec<usize>,
    /// Cache size as a fraction of the bank — the axis that transfers.
    pub residency_frac: Vec<f64>,
    pub alphabet: usize,
    pub activation_fraction: f64,
    pub fit_steps: usize,
    pub eval_steps: usize,
    pub shrinkage: Vec<ShrinkageArm>,
    pub oracle: Vec<f64>,
    pub null: Vec<f64>,
    /// `(session, stratum)` pairs that contributed.
    pub cells_scored: usize,
}

impl MassCurve {
    fn arm_at(&self, lambda: f64) -> Option<&ShrinkageArm> {
        self.shrinkage
            .iter()
            .find(|a| (a.lambda - lambda).abs() < 1e-12)
    }

    /// The static slice: pooled prior, no adaptation.
    pub fn static_arm(&self) -> Option<&ShrinkageArm> {
        self.arm_at(1.0)
    }

    /// The pure causal adaptive cache. The renderer prints every lambda arm, so
    /// this named accessor exists for callers grading the arm specifically;
    /// unit tests are its only readers today.
    #[allow(dead_code)]
    pub fn causal_arm(&self) -> Option<&ShrinkageArm> {
        self.arm_at(0.0)
    }

    /// Best realisable arm at one cache size, and its lambda.
    pub fn best_realisable(&self, size_idx: usize) -> Option<(f64, f64)> {
        self.shrinkage
            .iter()
            .map(|a| (a.lambda, a.coverage[size_idx]))
            .max_by(|x, y| x.1.total_cmp(&y.1))
    }

    /// Coverage bought per unit of residency — the axis DEC-8.4 is scored on,
    /// since cheaper cold symbols buy *more bank at fixed RAM* rather than
    /// cutting traffic in proportion to their own share of events.
    pub fn mass_per_residency(&self, coverage: &[f64]) -> Vec<f64> {
        coverage
            .iter()
            .zip(&self.residency_frac)
            .map(|(c, r)| if *r > 0.0 { c / r } else { 0.0 })
            .collect()
    }

    /// How much of the miss stream adaptation removes at one cache size:
    /// `(static miss / best realisable miss, static miss / oracle miss)`.
    /// The first is the prize on offer; the second is the prize with every
    /// benefit of the doubt granted.
    pub fn miss_ratios(&self, size_idx: usize) -> Option<(f64, f64)> {
        let stat = 1.0 - self.static_arm()?.coverage[size_idx];
        let best = 1.0 - self.best_realisable(size_idx)?.1;
        let orc = 1.0 - self.oracle[size_idx];
        let safe = |d: f64| if d > 0.0 { stat / d } else { f64::INFINITY };
        Some((safe(best), safe(orc)))
    }

    /// Locality actually present in the trace, separated from counting noise.
    pub fn locality_signal(&self, size_idx: usize) -> f64 {
        self.oracle[size_idx] - self.null[size_idx]
    }
}

fn normalise(counts: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return vec![0.0; counts.len()];
    }
    counts.iter().map(|c| c / total).collect()
}

/// Rank descending, ties broken by ascending index so runs are reproducible.
fn rank(key: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..key.len()).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]).then(a.cmp(&b)));
    order
}

/// Coverage of `events` by the top-`c` of `order`, for each configured size.
fn coverage_along(order: &[usize], events: &[f64], total: f64, sizes: &[usize]) -> Vec<f64> {
    let mut cum = 0.0;
    let mut next = 0;
    let mut out = vec![0.0; sizes.len()];
    // `sizes` is sorted ascending by the caller, so one pass serves all of them.
    for (taken, &idx) in order.iter().enumerate() {
        cum += events[idx];
        while next < sizes.len() && sizes[next] == taken + 1 {
            out[next] = cum / total;
            next += 1;
        }
    }
    // Sizes at or beyond the alphabet see the whole distribution.
    for slot in out.iter_mut().skip(next) {
        *slot = 1.0;
    }
    out
}

/// Draw `width` *distinct* symbols per step from `marginal`. Matching the real
/// selection process matters: sampling with replacement would let a symbol
/// repeat within one step, over-concentrating the null and understating
/// locality by construction.
fn draw_null(marginal: &[f64], width: usize, steps: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let mut counts = vec![0.0; marginal.len()];
    let mut pool = marginal.to_vec();
    for _ in 0..steps {
        pool.copy_from_slice(marginal);
        let mut remaining: f64 = pool.iter().sum();
        for _ in 0..width.min(marginal.len()) {
            if remaining <= 0.0 {
                break;
            }
            let mut target = rng.next_unit() * remaining;
            let mut chosen = pool.len() - 1;
            for (i, w) in pool.iter().enumerate() {
                target -= w;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            counts[chosen] += 1.0;
            remaining -= pool[chosen];
            pool[chosen] = 0.0;
        }
    }
    counts
}

pub fn mass_curve(trace: &SelectionTrace, cfg: &MassCurveConfig) -> Result<MassCurve, String> {
    if cfg.cache_sizes.is_empty() {
        return Err("frequency mass: no cache sizes requested".into());
    }
    if cfg.lambdas.is_empty() {
        return Err("frequency mass: no shrinkage weights requested".into());
    }
    let mut sizes = cfg.cache_sizes.clone();
    sizes.retain(|&c| c > 0);
    sizes.sort_unstable();
    sizes.dedup();
    if sizes.is_empty() {
        return Err("frequency mass: every requested cache size was zero".into());
    }

    let steps = trace.steps();
    let fit_steps = ((steps as f64 * cfg.fit_fraction).round() as usize).clamp(1, steps - 1);
    let eval_steps = steps - fit_steps;
    let e = trace.alphabet();

    let mut shrink = vec![vec![0.0; sizes.len()]; cfg.lambdas.len()];
    let mut oracle = vec![0.0; sizes.len()];
    let mut null = vec![0.0; sizes.len()];
    let mut cells = 0usize;
    let mut rng = SplitMix64(cfg.seed);

    for &stratum in trace.active_strata() {
        let mut pooled = vec![0.0; e];
        for b in 0..trace.sessions() {
            trace.accumulate(b, stratum, 0..steps, &mut pooled);
        }
        let marginal = normalise(&pooled);

        for b in 0..trace.sessions() {
            let mut own = vec![0.0; e];
            trace.accumulate(b, stratum, 0..steps, &mut own);
            let mut fit = vec![0.0; e];
            trace.accumulate(b, stratum, 0..fit_steps, &mut fit);
            let mut eval = vec![0.0; e];
            trace.accumulate(b, stratum, fit_steps..steps, &mut eval);

            let total: f64 = eval.iter().sum();
            if total <= 0.0 {
                continue;
            }
            cells += 1;

            // Leave-one-out: a static set must never be fitted on the session
            // it is scored against, or it reports self-coverage.
            let loo: Vec<f64> = pooled.iter().zip(&own).map(|(p, o)| p - o).collect();
            let loo = normalise(&loo);
            let fit_n = normalise(&fit);

            for (li, &lambda) in cfg.lambdas.iter().enumerate() {
                let key: Vec<f64> = loo
                    .iter()
                    .zip(&fit_n)
                    .map(|(p, f)| lambda * p + (1.0 - lambda) * f)
                    .collect();
                let cov = coverage_along(&rank(&key), &eval, total, &sizes);
                for (slot, c) in shrink[li].iter_mut().zip(&cov) {
                    *slot += c;
                }
            }

            let cov = coverage_along(&rank(&eval), &eval, total, &sizes);
            for (slot, c) in oracle.iter_mut().zip(&cov) {
                *slot += c;
            }

            let drawn = draw_null(&marginal, trace.width(), eval_steps, &mut rng);
            let drawn_total: f64 = drawn.iter().sum();
            if drawn_total > 0.0 {
                let cov = coverage_along(&rank(&drawn), &drawn, drawn_total, &sizes);
                for (slot, c) in null.iter_mut().zip(&cov) {
                    *slot += c;
                }
            }
        }
    }

    if cells == 0 {
        return Err("frequency mass: no (session, stratum) cell had scoreable events".into());
    }
    let n = cells as f64;
    let mean = |v: Vec<f64>| v.into_iter().map(|x| x / n).collect::<Vec<_>>();

    Ok(MassCurve {
        residency_frac: sizes.iter().map(|&c| c as f64 / e as f64).collect(),
        shrinkage: cfg
            .lambdas
            .iter()
            .zip(shrink)
            .map(|(&lambda, cov)| ShrinkageArm {
                lambda,
                coverage: mean(cov),
            })
            .collect(),
        oracle: mean(oracle),
        null: mean(null),
        cache_sizes: sizes,
        alphabet: e,
        activation_fraction: trace.activation_fraction(),
        fit_steps,
        eval_steps,
        cells_scored: cells,
    })
}

/// One point of the support curve: how many distinct symbols a session has
/// touched by `steps`, measured against the memoryless-uniform null.
#[derive(Debug, Clone, Serialize)]
pub struct SupportPoint {
    pub steps: usize,
    pub measured: f64,
    pub uniform: f64,
    pub measured_frac: f64,
    /// How far the uniform null overstates the measured support.
    pub overstatement: f64,
}

pub fn support_curve(trace: &SelectionTrace, step_grid: &[usize]) -> Vec<SupportPoint> {
    let strata = trace.active_strata();
    step_grid
        .iter()
        .filter(|&&t| t > 0 && t <= trace.steps())
        .map(|&t| {
            let mut total = 0.0;
            let mut n = 0usize;
            for &s in strata {
                for b in 0..trace.sessions() {
                    total += trace.support(b, s, t) as f64;
                    n += 1;
                }
            }
            let measured = if n > 0 { total / n as f64 } else { 0.0 };
            let uniform = trace.uniform_support(t);
            SupportPoint {
                steps: t,
                measured,
                uniform,
                measured_frac: measured / trace.alphabet() as f64,
                overstatement: if measured > 0.0 {
                    uniform / measured
                } else {
                    f64::INFINITY
                },
            }
        })
        .collect()
}
