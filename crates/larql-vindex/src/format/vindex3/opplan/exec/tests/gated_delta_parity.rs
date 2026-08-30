//! QW-2C: the reference recurrence against HF, on real Qwen3.8 weights.
//!
//! Opt-in. The fixture is a multi-gigabyte capture from the real 27B
//! checkpoint and lives outside the repo; point `QW2_FIXTURE` at its
//! directory to run these. Absent, they skip rather than silently pass —
//! a test that reports success when its subject is missing is worse than
//! no test.
//!
//! The oracle is HF's own `torch_recurrent_gated_delta_rule`, captured with
//! the model's real weights and real hidden activations. See
//! `capture_qw2.py` beside the fixture for provenance.
//!
//! Both outputs are gated, and the state matters more than the output: a
//! recurrence with a wrong state update produces a perfectly plausible
//! first position and a wrong second one.

use crate::format::vindex3::opplan::exec::cpu::WeightRows;
use std::path::{Path, PathBuf};

use crate::format::vindex3::opplan::exec::continuation::{RecurrentBuffer, RecurrentState};
use crate::format::vindex3::opplan::exec::gated_delta::{
    layer_forward, recurrence_step, recurrence_step_mutated, state_geometry, GatedDeltaWeights,
    Mutation, RecurrenceStep,
};
use crate::format::vindex3::opplan::{GatedDeltaOp, OperandRef};

/// Real Qwen3.8-27B geometry, as the captured fixture declares it.
const HIDDEN: usize = 5120;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
/// Layers captured: early, middle, late. All `linear_attention`.
const LAYERS: [usize; 3] = [0, 30, 62];

/// Compared against the reference driven with the SAME f32 inputs this
/// implementation reads, so the only remaining difference is arithmetic
/// order. Tight on purpose.
///
/// The looser number is measured, not guessed: HF against HF, differing
/// only in whether the L2-norm ran in bf16 or f32, disagrees by
/// `rel_rms 4.157e-3` at layer 0 t0 — which is exactly what this
/// implementation scored against the original bf16-path capture. The whole
/// gap was that one precision decision. Comparing against the f32-input
/// reference removes it instead of widening the gate to hide it.
const MAX_REL_RMS: f32 = 1e-5;
const MIN_COSINE: f32 = 0.999_999;
/// What the checkpoint's own bf16 dtype costs, end to end.
///
/// MEASURED, not chosen: across all three layers and both positions the
/// real bf16 forward differs from the f32 path by `3.555e-3 … 6.735e-3`.
/// The bound sits just above that range, so it bounds a known precision
/// cost rather than hiding an implementation change behind slack.
///
/// This covers the WHOLE bf16 path — projections, convolution and the
/// recurrence — not merely the L2-norm. An earlier `6e-3` was calibrated
/// when only the L2-norm differed between the two references, and went
/// stale the moment the f32 reference was regenerated across the full
/// chain. A bound that is not re-derived when its subject changes is a
/// bound that stops meaning anything.
const BF16_PATH_REL_RMS: f32 = 1e-2;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("QW2_FIXTURE").map(PathBuf::from)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let path = dir.join(name);
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture missing {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".into(),
        tensor: "fixture".into(),
        dtype: "F32".into(),
        shape: vec![1],
    }
}

fn op() -> GatedDeltaOp {
    GatedDeltaOp {
        num_key_heads: KEY_HEADS,
        num_value_heads: VALUE_HEADS,
        key_head_dim: HEAD_DIM,
        value_head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
        in_proj_qkv: operand(),
        in_proj_a: operand(),
        in_proj_b: operand(),
        in_proj_z: operand(),
        conv1d: operand(),
        a_log: operand(),
        dt_bias: operand(),
        norm: operand(),
        out_proj: operand(),
    }
}

struct Agreement {
    max_abs: f32,
    rel_rms: f32,
    cosine: f32,
}

/// Three metrics, because one hides different failures. `max_abs` catches a
/// single wrong cell that an average would bury; `rel_rms` scales to the
/// tensor's own magnitude; `cosine` catches a correct-shaped answer at the
/// wrong scale, which the other two can under-report.
fn agreement(mine: &[f32], theirs: &[f32]) -> Agreement {
    assert_eq!(mine.len(), theirs.len(), "compared tensors differ in size");
    let mut max_abs = 0.0f32;
    let (mut se, mut ss, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in mine.iter().zip(theirs) {
        max_abs = max_abs.max((a - b).abs());
        se += ((a - b) as f64).powi(2);
        ss += (*b as f64).powi(2);
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    Agreement {
        max_abs,
        rel_rms: (se / ss.max(f64::MIN_POSITIVE)).sqrt() as f32,
        cosine: (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)) as f32,
    }
}

fn check(label: &str, mine: &[f32], theirs: &[f32]) {
    let a = agreement(mine, theirs);
    println!(
        "  {label:34} max_abs {:.3e}  rel_rms {:.3e}  cos {:.7}",
        a.max_abs, a.rel_rms, a.cosine
    );
    assert!(
        a.rel_rms < MAX_REL_RMS,
        "{label}: rel_rms {:.3e} exceeds {MAX_REL_RMS:.0e}",
        a.rel_rms
    );
    assert!(
        a.cosine > MIN_COSINE,
        "{label}: cosine {:.7} below {MIN_COSINE}",
        a.cosine
    );
}

/// QW-2C gates 1-4 in one pass per layer: the sequence starts from a zero
/// state, and every position's output AND resulting state are compared.
/// Position 1 consumes the state position 0 produced, which is the only
/// arrangement that can catch a wrong state update.
#[test]
fn the_recurrence_and_its_state_match_hf_on_real_weights() {
    let Some(dir) = fixture_dir() else {
        eprintln!("QW2_FIXTURE unset — skipping the real-checkpoint parity gate");
        return;
    };
    let op = op();
    let per_token_qk = VALUE_HEADS * HEAD_DIM;

    for layer in LAYERS {
        println!("layer {layer}:");
        let q = read_f32(&dir, &format!("L{layer}_q_f32ref.f32"));
        let k = read_f32(&dir, &format!("L{layer}_k_f32ref.f32"));
        let v = read_f32(&dir, &format!("L{layer}_v_f32ref.f32"));
        let g = read_f32(&dir, &format!("L{layer}_g_f32ref.f32"));
        let beta = read_f32(&dir, &format!("L{layer}_beta_f32ref.f32"));
        let tokens = g.len() / VALUE_HEADS;
        assert!(tokens >= 2, "the state chain needs at least two positions");

        let mut state = RecurrentBuffer::zeros(
            &state_geometry(&op)
                .expect("the fixture declares a state precision")
                .buffers[crate::format::vindex3::opplan::exec::gated_delta::DELTA_MATRIX],
        );
        for t in 0..tokens {
            let qk = t * per_token_qk..(t + 1) * per_token_qk;
            let hv = t * VALUE_HEADS..(t + 1) * VALUE_HEADS;
            let step = RecurrenceStep {
                query: &q[qk.clone()],
                key: &k[qk.clone()],
                value: &v[qk],
                g: &g[hv.clone()],
                beta: &beta[hv],
            };
            let out = recurrence_step(&op, &step, &mut state);

            check(
                &format!("t{t} core_attn_out"),
                &out,
                &read_f32(&dir, &format!("L{layer}_t{t}_core_f32ref.f32")),
            );
            check(
                &format!("t{t} state_after"),
                state.cells(),
                &read_f32(&dir, &format!("L{layer}_t{t}_state_f32ref.f32")),
            );
            // And against the real bf16 forward, which is the number that
            // says how much the checkpoint's own dtype costs.
            let bf16 = agreement(&out, &read_f32(&dir, &format!("L{layer}_t{t}_core.f32")));
            println!(
                "  {:34} rel_rms {:.3e}  (real bf16 forward)",
                format!("t{t} core vs real forward"),
                bf16.rel_rms
            );
            assert!(
                bf16.rel_rms < BF16_PATH_REL_RMS,
                "t{t}: divergence from the real bf16 forward grew to {:.3e}",
                bf16.rel_rms
            );
        }
    }
}

/// The state a zero-start sequence produces is not itself zero, and the
/// second position's state differs from the first. Without this, a
/// implementation that silently produced zeros would pass every relative
/// metric above.
#[test]
fn the_state_actually_evolves() {
    let Some(dir) = fixture_dir() else {
        eprintln!("QW2_FIXTURE unset — skipping");
        return;
    };
    let s0 = read_f32(&dir, "L30_t0_state.f32");
    let s1 = read_f32(&dir, "L30_t1_state.f32");
    assert!(
        s0.iter().any(|x| x.abs() > 1e-6),
        "captured state after one position is all zeros — the fixture is not exercising anything"
    );
    let moved = agreement(&s0, &s1);
    assert!(
        moved.rel_rms > 1e-2,
        "state barely changed between positions (rel_rms {:.3e}); the chain gate would be vacuous",
        moved.rel_rms
    );
    println!(
        "  state moved between positions: rel_rms {:.3e}",
        moved.rel_rms
    );
}

const _: () = {
    assert!(HIDDEN == 5120);
};

/// QW-2D: every deliberate defect must move the gate, and the report says
/// by how much in each metric.
///
/// The point is not only that they fail. It is WHERE they fail: a defect
/// that is loud in the state and quiet in the output tells you the state
/// comparison is load-bearing and cannot be dropped when this is cut down
/// to a compact CI fixture.
#[test]
fn every_deliberate_defect_is_detected() {
    let Some(dir) = fixture_dir() else {
        eprintln!("QW2_FIXTURE unset — skipping the mutation controls");
        return;
    };
    let op = op();
    let per_token_qk = VALUE_HEADS * HEAD_DIM;
    let layer = 30;

    let q = read_f32(&dir, &format!("L{layer}_q_f32ref.f32"));
    let k = read_f32(&dir, &format!("L{layer}_k_f32ref.f32"));
    let v = read_f32(&dir, &format!("L{layer}_v_f32ref.f32"));
    let g = read_f32(&dir, &format!("L{layer}_g_f32ref.f32"));
    let beta = read_f32(&dir, &format!("L{layer}_beta_f32ref.f32"));
    let tokens = g.len() / VALUE_HEADS;

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}",
        "mutation", "out rel_rms", "out cos", "state rel_rms", "state cos"
    );
    for mutation in [
        Mutation::ScaleBeforeNorm,
        Mutation::ReadBeforeWrite,
        Mutation::NoDecay,
        Mutation::NoBeta,
        Mutation::RawGate,
    ] {
        let mut state = RecurrentBuffer::zeros(
            &state_geometry(&op)
                .expect("the fixture declares a state precision")
                .buffers[crate::format::vindex3::opplan::exec::gated_delta::DELTA_MATRIX],
        );
        let mut last_out = Vec::new();
        for t in 0..tokens {
            let qk = t * per_token_qk..(t + 1) * per_token_qk;
            let hv = t * VALUE_HEADS..(t + 1) * VALUE_HEADS;
            let step = RecurrenceStep {
                query: &q[qk.clone()],
                key: &k[qk.clone()],
                value: &v[qk],
                g: &g[hv.clone()],
                beta: &beta[hv],
            };
            last_out = recurrence_step_mutated(&op, &step, &mut state, mutation);
        }
        let last = tokens - 1;
        let o = agreement(
            &last_out,
            &read_f32(&dir, &format!("L{layer}_t{last}_core_f32ref.f32")),
        );
        let st = agreement(
            state.cells(),
            &read_f32(&dir, &format!("L{layer}_t{last}_state_f32ref.f32")),
        );
        println!(
            "{:<20} {:>12.3e} {:>12.7} {:>12.3e} {:>12.7}",
            format!("{mutation:?}"),
            o.rel_rms,
            o.cosine,
            st.rel_rms,
            st.cosine
        );
        assert!(
            o.rel_rms > MAX_REL_RMS || st.rel_rms > MAX_REL_RMS,
            "{mutation:?} was NOT detected by either gate — the comparator is blind to it"
        );
    }

    // And the unmutated path, for contrast on the same table.
    let mut state = RecurrentBuffer::zeros(
        &state_geometry(&op)
            .expect("the fixture declares a state precision")
            .buffers[crate::format::vindex3::opplan::exec::gated_delta::DELTA_MATRIX],
    );
    let mut last_out = Vec::new();
    for t in 0..tokens {
        let qk = t * per_token_qk..(t + 1) * per_token_qk;
        let hv = t * VALUE_HEADS..(t + 1) * VALUE_HEADS;
        last_out = recurrence_step(
            &op,
            &RecurrenceStep {
                query: &q[qk.clone()],
                key: &k[qk.clone()],
                value: &v[qk],
                g: &g[hv.clone()],
                beta: &beta[hv],
            },
            &mut state,
        );
    }
    let last = tokens - 1;
    let o = agreement(
        &last_out,
        &read_f32(&dir, &format!("L{layer}_t{last}_core_f32ref.f32")),
    );
    let st = agreement(
        state.cells(),
        &read_f32(&dir, &format!("L{layer}_t{last}_state_f32ref.f32")),
    );
    println!(
        "{:<20} {:>12.3e} {:>12.7} {:>12.3e} {:>12.7}",
        "None (control)", o.rel_rms, o.cosine, st.rel_rms, st.cosine
    );
}

const NORM_EPS: f32 = 1e-6;

fn weights<'a>(store: &'a [Vec<f32>]) -> GatedDeltaWeights<'a> {
    GatedDeltaWeights {
        in_proj_qkv: WeightRows::F32(&store[0]),
        in_proj_a: WeightRows::F32(&store[1]),
        in_proj_b: WeightRows::F32(&store[2]),
        in_proj_z: WeightRows::F32(&store[3]),
        conv1d: &store[4],
        a_log: &store[5],
        dt_bias: &store[6],
        norm: &store[7],
        out_proj: WeightRows::F32(&store[8]),
        norm_eps: NORM_EPS,
    }
}

fn load_weights(dir: &Path, layer: usize) -> Vec<Vec<f32>> {
    [
        "in_proj_qkv",
        "in_proj_a",
        "in_proj_b",
        "in_proj_z",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
    ]
    .iter()
    .map(|n| read_f32(dir, &format!("L{layer}_{n}.f32")))
    .collect()
}

/// QW-2E: the WHOLE operator, boundary by boundary.
///
/// Every captured plane is gated, not merely the layer output. A
/// disagreement in `q/k/v` should be reported as a convolution or
/// head-expansion fault, not chased backwards from a wrong residual.
#[test]
fn the_whole_operator_matches_hf_stage_by_stage() {
    let Some(dir) = fixture_dir() else {
        eprintln!("QW2_FIXTURE unset — skipping the whole-operator gate");
        return;
    };
    let op = op();

    for layer in LAYERS {
        println!("layer {layer}:");
        let store = load_weights(&dir, layer);
        let w = weights(&store);
        let input = read_f32(&dir, &format!("L{layer}_input.f32"));
        let tokens = input.len() / HIDDEN;
        let hidden: Vec<Vec<f32>> = (0..tokens)
            .map(|t| input[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
            .collect();

        let mut state = RecurrentState::zeros(
            &state_geometry(&op).expect("the fixture declares a state precision"),
        );
        let planes = layer_forward(&op, &w, &hidden, &mut state, Mutation::None);

        // Boundaries HF captured for the whole sequence at once.
        let per_qk = VALUE_HEADS * HEAD_DIM;
        for (name, mine, whole) in [
            (
                "q",
                &planes.query,
                read_f32(&dir, &format!("L{layer}_q_f32ref.f32")),
            ),
            (
                "k",
                &planes.key,
                read_f32(&dir, &format!("L{layer}_k_f32ref.f32")),
            ),
            (
                "v",
                &planes.value,
                read_f32(&dir, &format!("L{layer}_v_f32ref.f32")),
            ),
            (
                "z",
                &planes.z,
                read_f32(&dir, &format!("L{layer}_z_f32ref.f32")),
            ),
        ] {
            let width = whole.len() / tokens;
            let flat: Vec<f32> = mine.iter().flatten().copied().collect();
            assert_eq!(flat.len(), whole.len(), "{name}: width mismatch");
            let _ = per_qk;
            check(&format!("{name} (all positions)"), &flat, &whole);
            let _ = width;
        }
        for (name, mine, whole) in [
            (
                "g",
                &planes.g,
                read_f32(&dir, &format!("L{layer}_g_f32ref.f32")),
            ),
            (
                "beta",
                &planes.beta,
                read_f32(&dir, &format!("L{layer}_beta_f32ref.f32")),
            ),
        ] {
            let flat: Vec<f32> = mine.iter().flatten().copied().collect();
            check(&format!("{name} (all positions)"), &flat, &whole);
        }

        for t in 0..tokens {
            check(
                &format!("t{t} core_attn_out"),
                &planes.core[t],
                &read_f32(&dir, &format!("L{layer}_t{t}_core_f32ref.f32")),
            );
            check(
                &format!("t{t} layer_output"),
                &planes.output[t],
                &read_f32(&dir, &format!("L{layer}_t{t}_out_f32ref.f32")),
            );
        }
        check(
            "final state",
            state.buffer(0).cells(),
            &read_f32(&dir, &format!("L{layer}_t{}_state_f32ref.f32", tokens - 1)),
        );
    }
}

/// QW-2E's own controls: the defects that live OUTSIDE the recurrence.
///
/// Each must be caught at the earliest plane it corrupts — a convolution
/// fault has no business surviving until the layer output.
#[test]
fn outer_path_defects_are_caught_at_their_own_stage() {
    let Some(dir) = fixture_dir() else {
        eprintln!("QW2_FIXTURE unset — skipping");
        return;
    };
    let op = op();
    let layer = 30;
    let store = load_weights(&dir, layer);
    let w = weights(&store);
    let input = read_f32(&dir, &format!("L{layer}_input.f32"));
    let tokens = input.len() / HIDDEN;
    let hidden: Vec<Vec<f32>> = (0..tokens)
        .map(|t| input[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
        .collect();
    let ref_q = read_f32(&dir, &format!("L{layer}_q_f32ref.f32"));
    let ref_out = read_f32(&dir, &format!("L{layer}_t{}_out_f32ref.f32", tokens - 1));

    println!(
        "{:<22} {:>13} {:>13}",
        "mutation", "q rel_rms", "output rel_rms"
    );
    for mutation in [
        Mutation::SiluBeforeConv,
        Mutation::CentredConv,
        Mutation::TiledHeadExpansion,
        Mutation::None,
    ] {
        let mut state = RecurrentState::zeros(
            &state_geometry(&op).expect("the fixture declares a state precision"),
        );
        let planes = layer_forward(&op, &w, &hidden, &mut state, mutation);
        let flat: Vec<f32> = planes.query.iter().flatten().copied().collect();
        let q = agreement(&flat, &ref_q);
        let out = agreement(&planes.output[tokens - 1], &ref_out);
        println!(
            "{:<22} {:>13.3e} {:>13.3e}",
            format!("{mutation:?}"),
            q.rel_rms,
            out.rel_rms
        );
        if mutation == Mutation::None {
            continue;
        }
        assert!(
            q.rel_rms > MAX_REL_RMS,
            "{mutation:?} was not visible at the q plane — the stage gate is blind to it"
        );
    }
}
