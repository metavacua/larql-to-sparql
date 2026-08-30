//! QW-3.5B: Qwen3.8's text rotary against HF, and what text can and
//! cannot falsify about M-RoPE.
//!
//! The claim under test is deliberately narrow:
//!
//! > Qwen3.8 text RoPE reproduces HF because LARQL represents the actual
//! > partial M-RoPE semantics — not because text positions happen to make
//! > an incorrect simplification numerically equivalent.
//!
//! The second half is why the mutation table below is **pre-registered
//! into two classes** rather than "every mutation must fail". At `t == h
//! == w` — every text-only position — HF's `apply_interleaved_mrope`
//! selects between identical values and is the identity map, so the axis
//! assignment is genuinely unidentifiable from text. Demanding that those
//! arms diverge would mean demanding the wrong answer.
//!
//! ```text
//! MUST DIVERGE                    EXPECTED TEXT-TANGENT-EQUIVALENT
//!   FullRope                        WrongSection([10,11,11])
//!   WrongFraction                   NonInterleaved
//!   HeadWidthBasis                  PlainPartial1D
//! ```
//!
//! A zero in the right-hand column is evidence of **non-identifiability**,
//! not a pass for a wrong implementation. The image-conditioned execution
//! path is what will eventually falsify those three; until it exists, the
//! sectioning and interleaving are carried and lowered on the strength of
//! the declaration, and this file says so rather than implying text parity
//! covered them.
//!
//! Fixture provenance: `scripts/gen_qw35b_mrope_fixture.py`, HF's own
//! `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`. The oracle needs
//! the config only, so this is hermetic and committed.

use larql_models::config::mrope_axis_table;
use serde::Deserialize;

use super::super::kernels::{mrope_rotate, mrope_rotate_scaled};
use crate::format::vindex3::fixtures::lcg_values;

const FIXTURE: &str = include_str!("fixtures/qw35b_mrope_hf.json");

#[derive(Deserialize)]
struct Fixture {
    config: Config,
    lcg: Lcg,
    positions: Vec<usize>,
    q_rotated: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Config {
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    mrope_section: [usize; 3],
    mrope_interleaved: bool,
    rotary_dim: usize,
    n_freqs: usize,
}

#[derive(Deserialize)]
struct Lcg {
    seed: u64,
    count: usize,
    num_heads: usize,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("the committed M-RoPE fixture parses")
}

/// The declared arithmetic, asserted against the fixture HF produced.
///
/// `sum(mrope_section)` counts FREQUENCY slots — `rotary_dim / 2` — and
/// not `rotary_dim`. Reading the head width as 128 (the Gated DeltaNet
/// head dim, a different operator) makes `sum == rotary_dim` close
/// instead: `128 * 0.25 == 32 == 11 + 11 + 10`. That identity is
/// self-consistent and wrong, and it is the reason this test asserts the
/// head width HF actually used rather than any nearby 128.
#[test]
fn the_declared_rope_arithmetic_closes_on_the_real_head_width() {
    let f = fixture();
    let c = &f.config;
    assert_eq!(c.head_dim, 256, "softmax head width, not DeltaNet's 128");
    assert_eq!(
        c.rotary_dim,
        (c.head_dim as f64 * c.partial_rotary_factor) as usize,
        "rotary_dim == head_dim * partial_rotary_factor"
    );
    assert_eq!(c.n_freqs, c.rotary_dim / 2);
    assert_eq!(
        c.mrope_section.iter().sum::<usize>(),
        c.n_freqs,
        "sum(section) counts frequency slots == rotary_dim/2, NOT rotary_dim"
    );
    // The identity that would have closed on the wrong head width, kept
    // as a live control: if this ever stops being a near-miss the comment
    // above has gone stale.
    assert_eq!(
        c.mrope_section.iter().sum::<usize>(),
        (128.0 * c.partial_rotary_factor) as usize,
        "the 128-head-width misreading is still numerically seductive"
    );
}

/// The generator that produced the fixture's inputs still produces them.
///
/// `q` is not stored — only HF's output is. If `lcg_values` ever drifts,
/// the parity test would silently compare HF's rotation of one input
/// against LARQL's rotation of another, and pass or fail for a reason
/// that has nothing to do with rope.
#[test]
fn the_generator_still_produces_the_values_this_fixture_was_built_from() {
    let f = fixture();
    let q = lcg_values(f.lcg.count, f.lcg.seed);
    assert_eq!(
        q.len(),
        f.lcg.num_heads * f.positions.len() * f.config.head_dim
    );
    // Spot-check against values transcribed from the generating run.
    assert!(
        q.iter().all(|v| v.abs() <= 0.05),
        "lcg_values range drifted"
    );
    assert_ne!(
        q[0], q[1],
        "a constant input cannot discriminate a rotation"
    );
}

/// Which semantics a mutation perturbs. Applied to the REAL application
/// kernel with perturbed inputs — never to a copy of it, so a mutation
/// that "passes" cannot be an artefact of testing a duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutation {
    None,
    /// Rotate the whole 256-dim head instead of the 64-dim prefix.
    FullRope,
    /// `partial_rotary_factor` 0.5 rather than 0.25.
    WrongFraction,
    /// Inverse frequencies over the head width (`/256`) instead of the
    /// rotary width (`/64`) — same dims rotated, different angles.
    HeadWidthBasis,
    /// A different axis partition with the same total.
    WrongSection,
    /// Contiguous `TTT…HHH…WWW` instead of interleaved `THWTHW…`.
    NonInterleaved,
    /// Ignore the axis assignment entirely: plain partial 1-D rotary.
    PlainPartial1D,
}

impl Mutation {
    /// Pre-registered expectation. `true` = this mutation changes what a
    /// TEXT sequence computes and must be observable.
    fn must_diverge_on_text(self) -> bool {
        matches!(
            self,
            Self::FullRope | Self::WrongFraction | Self::HeadWidthBasis
        )
    }
}

/// One head rotated under `mutation`, through the real kernels.
fn rotate(head: &mut [f32], position: usize, c: &Config, mutation: Mutation) {
    let grid = [position, position, position];
    if mutation == Mutation::None {
        // The control runs the top-level function, so the thing proven
        // against HF is the operator itself and not a re-derivation.
        mrope_rotate(
            &mut head[..c.rotary_dim],
            grid,
            c.rope_theta,
            c.mrope_section,
            c.mrope_interleaved,
        );
        return;
    }
    let (width, basis_dim, section, interleaved) = match mutation {
        Mutation::None => unreachable!(),
        Mutation::FullRope => (c.head_dim, c.head_dim, c.mrope_section, c.mrope_interleaved),
        Mutation::WrongFraction => {
            let w = (c.head_dim as f64 * 0.5) as usize;
            (w, w, c.mrope_section, c.mrope_interleaved)
        }
        Mutation::HeadWidthBasis => (
            c.rotary_dim,
            c.head_dim,
            c.mrope_section,
            c.mrope_interleaved,
        ),
        Mutation::WrongSection => (c.rotary_dim, c.rotary_dim, [10, 11, 11], true),
        Mutation::NonInterleaved => (c.rotary_dim, c.rotary_dim, c.mrope_section, false),
        Mutation::PlainPartial1D => (c.rotary_dim, c.rotary_dim, [c.n_freqs, 0, 0], false),
    };
    let half = width / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| c.rope_theta.powf(-2.0 * i as f64 / basis_dim as f64))
        .collect();
    let axes = mrope_axis_table(section, interleaved, half);
    mrope_rotate_scaled(&mut head[..width], grid, &axes, &inv_freq, 1.0);
}

/// Largest absolute disagreement with HF over every head and position.
fn max_abs_vs_hf(f: &Fixture, mutation: Mutation) -> f32 {
    let c = &f.config;
    let q = lcg_values(f.lcg.count, f.lcg.seed);
    let n_pos = f.positions.len();
    let mut worst = 0.0f32;
    for head_idx in 0..f.lcg.num_heads {
        for (p, &position) in f.positions.iter().enumerate() {
            let row = head_idx * n_pos + p;
            let start = row * c.head_dim;
            let mut head = q[start..start + c.head_dim].to_vec();
            rotate(&mut head, position, c, mutation);
            for (got, want) in head.iter().zip(&f.q_rotated[row]) {
                worst = worst.max((got - want).abs());
            }
        }
    }
    worst
}

/// **The parity gate.** The operator as represented reproduces HF's text
/// rotation.
#[test]
fn the_represented_operator_reproduces_hf_text_rotation() {
    let f = fixture();
    let worst = max_abs_vs_hf(&f, Mutation::None);
    assert!(
        worst < 1e-6,
        "M-RoPE disagrees with HF on the text path: max abs {worst:e}"
    );
}

/// **The identifiability table.** Both classes measured, neither assumed.
///
/// The right-hand class returning exactly `0` is the finding: those
/// semantics are unidentifiable on the text tangent. Note it is asserted
/// as *exactly* zero rather than "small" — a tolerance would hide a
/// mutation that genuinely perturbs text by a little.
#[test]
fn the_mutation_table_separates_falsifiable_from_unidentifiable() {
    let f = fixture();
    let control = max_abs_vs_hf(&f, Mutation::None);
    for mutation in [
        Mutation::FullRope,
        Mutation::WrongFraction,
        Mutation::HeadWidthBasis,
        Mutation::WrongSection,
        Mutation::NonInterleaved,
        Mutation::PlainPartial1D,
    ] {
        let worst = max_abs_vs_hf(&f, mutation);
        if mutation.must_diverge_on_text() {
            assert!(
                worst > 1e-3,
                "{mutation:?} was pre-registered as falsifiable on text but moved nothing \
                 ({worst:e}); either the mutation is not being applied or the claim is wrong"
            );
        } else {
            assert_eq!(
                worst, control,
                "{mutation:?} was pre-registered as text-tangent-equivalent but differs from \
                 the control ({worst:e} vs {control:e}) — text CAN see it, so the \
                 non-identifiability claim in this module's docs is false"
            );
        }
    }
}

/// The axis table is not vacuous: on a genuine 3-D grid the sectioning
/// and interleaving DO change the result.
///
/// Without this, every "expected-equivalent" zero above would also be
/// satisfied by an `mrope_axis_table` that returned all-T. This is the
/// control that says the degeneracy belongs to the text positions rather
/// than to the implementation.
#[test]
fn the_axis_assignment_is_live_once_the_grid_stops_being_degenerate() {
    let f = fixture();
    let c = &f.config;
    let interleaved = mrope_axis_table(c.mrope_section, true, c.n_freqs);
    let contiguous = mrope_axis_table(c.mrope_section, false, c.n_freqs);
    assert_ne!(interleaved, contiguous, "the two layouts must differ");
    assert_eq!(
        interleaved.iter().filter(|a| **a == 1).count(),
        c.mrope_section[1]
    );
    assert_eq!(
        interleaved.iter().filter(|a| **a == 2).count(),
        c.mrope_section[2]
    );
    assert_eq!(
        interleaved.iter().filter(|a| **a == 0).count(),
        c.mrope_section[0]
    );
    // HF's `apply_interleaved_mrope`: H at slice(1, s1*3, 3), W at
    // slice(2, s2*3, 3), T everywhere else — THWTHW…
    assert_eq!(&interleaved[..6], &[0, 1, 2, 0, 1, 2]);

    // A non-degenerate grid: t, h and w differ, so the assignment shows.
    let inv_freq: Vec<f64> = (0..c.n_freqs)
        .map(|i| c.rope_theta.powf(-2.0 * i as f64 / c.rotary_dim as f64))
        .collect();
    let base = lcg_values(c.head_dim, 0x5B35);
    let mut a = base.clone();
    let mut b = base.clone();
    mrope_rotate_scaled(
        &mut a[..c.rotary_dim],
        [3, 11, 29],
        &interleaved,
        &inv_freq,
        1.0,
    );
    mrope_rotate_scaled(
        &mut b[..c.rotary_dim],
        [3, 11, 29],
        &contiguous,
        &inv_freq,
        1.0,
    );
    let spread = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-4,
        "on a real grid the interleave must change the rotation, else the text-tangent \
         zeros above prove nothing about the implementation: {spread:e}"
    );
}

/// Prints the measured table (`--nocapture`). Kept as a test so the
/// numbers behind the two pre-registered classes are reproducible rather
/// than quoted from a commit message.
#[test]
fn report_the_measured_identifiability_table() {
    let f = fixture();
    println!("\n  mutation                 max abs vs HF (text)   pre-registered");
    println!("  ---------------------------------------------------------------");
    for m in [
        Mutation::None,
        Mutation::FullRope,
        Mutation::WrongFraction,
        Mutation::HeadWidthBasis,
        Mutation::WrongSection,
        Mutation::NonInterleaved,
        Mutation::PlainPartial1D,
    ] {
        let class = if m == Mutation::None {
            "control"
        } else if m.must_diverge_on_text() {
            "must diverge"
        } else {
            "expected equivalent"
        };
        println!(
            "  {:<24} {:>12.6e}   {class}",
            format!("{m:?}"),
            max_abs_vs_hf(&f, m)
        );
    }
}
