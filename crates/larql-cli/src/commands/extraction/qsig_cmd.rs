//! `larql quantum-signature` — the SP2 over-constrained falsification runner.
//!
//! Per head, the full witness battery (W1–W6 via `Witnesses`, W3/W8 sheaf via the
//! contextual fraction) is evaluated on the REAL coupling and on three nulls
//! (Gaussian, singular-value-matched, sign-randomized) constructed from that
//! coupling and pushed through the IDENTICAL `from_matrix → reduce → witness`
//! pipe (predicativity). Both metric duals are run: raw Euclidean and the
//! canonical (Cholesky-whitened) metric when `canonical_meta.json` is present.
//! The reported quantity is the predicative signal `real − null`, never a bare
//! "looks quantum". W7 (reproducibility ⟹ pseudo-random) is determinative and
//! bounds every interpretation to STRUCTURE only — see the spec.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use larql_hilbert::{bell_empirical_model, reduced_rho2, Witnesses};
use larql_vindex::canonical::{head_block, head_coupling, kv_head_for_query, CanonicalMeta};
use larql_vindex::format::filenames::CANONICAL_META_JSON;
use larql_vindex::format::{attn_load::load_attention_qk, load::load_vindex_config};

use crate::commands::extraction::qsig_metric::canonical_coupling;
use crate::commands::extraction::qsig_nulls::{gaussian_null, sign_randomized_null, sv_matched_null};
use crate::commands::extraction::qsig_sheaf::contextual_fraction;

/// Determinative W7 statement recorded in every report (the conclusive negative).
const W7_DETERMINATIVE: &str = "reproducible under fixed seed ⟹ pseudo-random ⟹ \
NOT quantum-random — holds for the LLM, the QLM-sim, and all nulls; genuine \
quantum randomness is conclusively refuted for every computable source. All \
witnesses test the STRUCTURE of pseudo-random outputs, never randomness-as-such.";

/// The full witness battery as averageable numbers (`lattice_ok` carried as a
/// fraction so head/draw means are well-defined).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WitnessMeans {
    pub mutual_information: f64,   // W1
    pub negativity: f64,          // W2
    pub chsh: f64,                // W3
    pub entanglement_entropy: f64, // W4
    pub gap: f64,                 // W5
    pub hilbertian_residual: f64, // W6
    pub contextual_fraction: f64, // W3/W8 sheaf
    pub lattice_ok: f64,          // self-check (1.0 = holds)
}

impl WitnessMeans {
    /// Full battery on one coupling (the single pipe; CF reuses `reduced_rho2`).
    fn from_coupling(c: &Array2<f64>) -> Self {
        let w = Witnesses::from_coupling(c);
        let cf = contextual_fraction(&bell_empirical_model(&reduced_rho2(c)));
        WitnessMeans {
            mutual_information: w.mutual_information,
            negativity: w.negativity,
            chsh: w.chsh,
            entanglement_entropy: w.entanglement_entropy,
            gap: w.gap,
            hilbertian_residual: w.hilbertian_residual,
            contextual_fraction: cf,
            lattice_ok: if w.lattice_ok { 1.0 } else { 0.0 },
        }
    }
    fn add(&mut self, o: &Self) {
        self.mutual_information += o.mutual_information;
        self.negativity += o.negativity;
        self.chsh += o.chsh;
        self.entanglement_entropy += o.entanglement_entropy;
        self.gap += o.gap;
        self.hilbertian_residual += o.hilbertian_residual;
        self.contextual_fraction += o.contextual_fraction;
        self.lattice_ok += o.lattice_ok;
    }
    fn scale(&mut self, s: f64) {
        self.mutual_information *= s;
        self.negativity *= s;
        self.chsh *= s;
        self.entanglement_entropy *= s;
        self.gap *= s;
        self.hilbertian_residual *= s;
        self.contextual_fraction *= s;
        self.lattice_ok *= s;
    }
    fn sub(&self, o: &Self) -> Self {
        let mut r = self.clone();
        r.mutual_information -= o.mutual_information;
        r.negativity -= o.negativity;
        r.chsh -= o.chsh;
        r.entanglement_entropy -= o.entanglement_entropy;
        r.gap -= o.gap;
        r.hilbertian_residual -= o.hilbertian_residual;
        r.contextual_fraction -= o.contextual_fraction;
        r.lattice_ok -= o.lattice_ok;
        r
    }
}

/// Mean of `n` `WitnessMeans` produced by `f` (n ≥ 1).
fn mean_of(n: usize, f: impl Fn(usize) -> WitnessMeans) -> WitnessMeans {
    let mut acc = WitnessMeans::default();
    for i in 0..n {
        acc.add(&f(i));
    }
    acc.scale(1.0 / n.max(1) as f64);
    acc
}

/// One head, one metric: the real battery and each null kind (each averaged over
/// `draws` independent seeds to suppress single-sample null variance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadArm {
    pub real: WitnessMeans,
    pub gaussian: WitnessMeans,
    pub sv_matched: WitnessMeans,
    pub sign_randomized: WitnessMeans,
}

/// Evaluate the battery on `real_c` and the three nulls derived from it via the
/// IDENTICAL pipe. `draws` = null replicates per kind (averaged). `seed_base`
/// makes the whole head reproducible (W7 self-consistency).
pub fn analyze_coupling(real_c: &Array2<f64>, seed_base: u64, draws: usize) -> HeadArm {
    let d = real_c.shape()[0];
    let draws = draws.max(1);
    HeadArm {
        real: WitnessMeans::from_coupling(real_c),
        gaussian: mean_of(draws, |i| {
            WitnessMeans::from_coupling(&gaussian_null(d, d, seed_base ^ (0x1000 + i as u64)))
        }),
        sv_matched: mean_of(draws, |i| {
            WitnessMeans::from_coupling(&sv_matched_null(real_c, seed_base ^ (0x2000 + i as u64)))
        }),
        sign_randomized: mean_of(draws, |i| {
            WitnessMeans::from_coupling(&sign_randomized_null(real_c, seed_base ^ (0x3000 + i as u64)))
        }),
    }
}

/// Population summary for one metric: head-mean real, head-mean null (mean of the
/// three kinds), and the predicative signal `real − null`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub metric: String,
    pub heads: usize,
    pub real: WitnessMeans,
    pub null: WitnessMeans,
    pub signal: WitnessMeans,
}

pub fn summarize(metric: &str, arms: &[HeadArm]) -> MetricSummary {
    let n = arms.len().max(1);
    let real = mean_of(arms.len(), |i| arms[i].real.clone());
    let null = mean_of(arms.len(), |i| {
        let mut m = arms[i].gaussian.clone();
        m.add(&arms[i].sv_matched);
        m.add(&arms[i].sign_randomized);
        m.scale(1.0 / 3.0);
        m
    });
    let signal = real.sub(&null);
    MetricSummary { metric: metric.to_string(), heads: n, real, null, signal }
}

/// Root report written to `quantum_signature_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QsigReport {
    pub version: u32,
    pub model: String,
    pub head_dim: usize,
    pub nulls_per_head: usize,
    pub determinative_w7: String,
    pub metrics: Vec<MetricSummary>,
}

#[derive(Args)]
pub struct QsigArgs {
    /// Path to the .vindex directory to analyse.
    vindex: PathBuf,
    /// Null replicates per kind per head (averaged to suppress single-sample variance).
    #[arg(long, default_value_t = 8)]
    nulls_per_head: usize,
}

fn print_metric(s: &MetricSummary) {
    println!("  ── metric: {} ({} heads) ──", s.metric, s.heads);
    let row = |name: &str, r: f64, nu: f64, sig: f64| {
        println!("    {name:<22} real {r:+.4}  null {nu:+.4}  signal(real−null) {sig:+.4}");
    };
    row("W1 mutual_info", s.real.mutual_information, s.null.mutual_information, s.signal.mutual_information);
    row("W2 negativity", s.real.negativity, s.null.negativity, s.signal.negativity);
    row("W3 chsh", s.real.chsh, s.null.chsh, s.signal.chsh);
    row("W4 entangle_entropy", s.real.entanglement_entropy, s.null.entanglement_entropy, s.signal.entanglement_entropy);
    row("W5 gap", s.real.gap, s.null.gap, s.signal.gap);
    row("W6 hilbertian_resid", s.real.hilbertian_residual, s.null.hilbertian_residual, s.signal.hilbertian_residual);
    row("W3/W8 contextual_frac", s.real.contextual_fraction, s.null.contextual_fraction, s.signal.contextual_fraction);
    println!("    lattice_ok fraction   real {:.3}  null {:.3}", s.real.lattice_ok, s.null.lattice_ok);
}

pub fn run(args: QsigArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = &args.vindex;
    let draws = args.nulls_per_head.max(1);
    let t0 = Instant::now();
    println!("Quantum-signature falsification apparatus for vindex at {}", dir.display());

    let config = load_vindex_config(dir)?;
    let mc = config
        .model_config
        .as_ref()
        .ok_or("quantum-signature: index.json has no model_config (need head_dim / head counts)")?;
    let (num_q, num_kv, head_dim) = (mc.num_q_heads, mc.num_kv_heads, mc.head_dim);
    println!(
        "  model: {} ({}), {} layers, q_heads={num_q}, kv_heads={num_kv}, head_dim={head_dim}, nulls/head={draws}",
        config.model, config.family, config.num_layers
    );

    // Canonical metric dual: present only if the vindex has been canonicalized.
    let l_opt: Option<Array2<f64>> = {
        let p = dir.join(CANONICAL_META_JSON);
        if p.exists() {
            let meta: CanonicalMeta = serde_json::from_slice(&std::fs::read(&p)?)?;
            let l = meta.unpack_cholesky_l().map_err(|e| format!("canonical_meta: {e}"))?;
            println!("  canonical metric: loaded Cholesky L ({}×{})", l.shape()[0], l.shape()[1]);
            Some(l)
        } else {
            println!("  canonical metric: SKIPPED (no canonical_meta.json — run `larql canonicalize` to enable the dual)");
            None
        }
    };

    let qk = load_attention_qk(dir, config.num_layers)?;
    let mut raw_arms: Vec<HeadArm> = Vec::new();
    let mut canon_arms: Vec<HeadArm> = Vec::new();

    for (layer, (wq, wk)) in qk.iter().enumerate() {
        if wq.nrows() < num_q * head_dim || wk.nrows() < num_kv * head_dim {
            return Err(format!(
                "layer {layer}: attention weights too small (wq {} rows, wk {} rows; need {} and {})",
                wq.nrows(), wk.nrows(), num_q * head_dim, num_kv * head_dim
            )
            .into());
        }
        let wq64 = wq.mapv(|v| v as f64);
        let wk64 = wk.mapv(|v| v as f64);
        for h in 0..num_q {
            let g = kv_head_for_query(h, num_q, num_kv);
            let wq_h = head_block(&wq64, h, head_dim);
            let wk_g = head_block(&wk64, g, head_dim);
            let seed = (layer as u64) << 32 | h as u64;
            let raw_c = head_coupling(&wq_h, &wk_g);
            raw_arms.push(analyze_coupling(&raw_c, seed, draws));
            if let Some(l) = &l_opt {
                let canon_c = canonical_coupling(&wq_h, &wk_g, l);
                canon_arms.push(analyze_coupling(&canon_c, seed, draws));
            }
        }
    }

    let mut metrics = vec![summarize("raw", &raw_arms)];
    if !canon_arms.is_empty() {
        metrics.push(summarize("canonical", &canon_arms));
    }

    println!("\n  Predicative signal = real − null (≈0 with poles passing ⇒ conclusive negative;");
    println!("  positive signal ⇒ 'not ruled out', confound-prone, NOT a quantum claim):");
    for m in &metrics {
        print_metric(m);
    }
    println!("\n  W7 (determinative): {W7_DETERMINATIVE}");
    println!(
        "  NOTE: a classical/quantum verdict is admissible only from the FULL apparatus \n        (all witnesses + all nulls + lattice + analytic poles). This run reports signals."
    );

    let report = QsigReport {
        version: 1,
        model: config.model.clone(),
        head_dim,
        nulls_per_head: draws,
        determinative_w7: W7_DETERMINATIVE.to_string(),
        metrics,
    };
    let out = dir.join("quantum_signature_meta.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("  wrote {}", out.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_coupling() -> Array2<f64> {
        Array2::from_shape_vec((4, 4), (0..16).map(|i| (i as f64 * 0.37).sin()).collect()).unwrap()
    }

    #[test]
    fn analyze_runs_real_and_three_nulls_through_the_identical_pipe() {
        let arm = analyze_coupling(&sample_coupling(), 7, 4);
        for w in [&arm.real, &arm.gaussian, &arm.sv_matched, &arm.sign_randomized] {
            assert!(w.mutual_information.is_finite() && w.mutual_information >= -1e-9);
            assert!(w.negativity.is_finite() && w.negativity >= -1e-9);
            assert!(w.chsh.is_finite());
            assert!(w.contextual_fraction.is_finite() && w.contextual_fraction >= -1e-9);
            assert!((0.0..=1.0).contains(&w.lattice_ok));
        }
    }

    #[test]
    fn nulls_are_reproducible_under_fixed_seed() {
        // W7 in the apparatus itself: same seed ⇒ identical null (classical reflexivity).
        let c = sample_coupling();
        let a = analyze_coupling(&c, 7, 4);
        let b = analyze_coupling(&c, 7, 4);
        assert_eq!(a.gaussian.negativity, b.gaussian.negativity);
        assert_eq!(a.sv_matched.chsh, b.sv_matched.chsh);
    }

    #[test]
    fn signal_is_real_minus_mean_null() {
        let arms = vec![
            analyze_coupling(&sample_coupling(), 1, 2),
            analyze_coupling(&Array2::<f64>::eye(4), 2, 2),
        ];
        let s = summarize("raw", &arms);
        assert_eq!(s.heads, 2);
        // signal.negativity == real.negativity − null.negativity (exact arithmetic gate)
        assert!((s.signal.negativity - (s.real.negativity - s.null.negativity)).abs() < 1e-12);
        assert!((s.signal.chsh - (s.real.chsh - s.null.chsh)).abs() < 1e-12);
    }
}
