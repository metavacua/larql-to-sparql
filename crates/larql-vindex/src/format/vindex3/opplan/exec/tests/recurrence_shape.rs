//! CPU-2D1: the recurrence was reshaped for the optimiser, not for the
//! arithmetic.
//!
//! `step_inner` used to read `dk`/`dv` from a `&GatedDeltaOp` and reach
//! every element through `state.cells_mut()[cell(op, h, kk, vv)]`, four
//! sweeps deep. A transcription of the same loop with compile-time
//! dimensions and a bare slice ran **1.92x faster** — same arithmetic,
//! same machine, same process. That is a code-shape cost, not an
//! algorithmic one.
//!
//! So the gate is bit-identity, not a tolerance. Nothing about this rung
//! is a numerical decision, and the moment it needs an epsilon it has
//! become a different rung.
//!
//! **Output AND state.** QW-2 already showed a recurrence can be right
//! about the current token and wrong about the state every following one
//! reads; a gate on the output alone would pass that.

use super::super::continuation::RecurrentBuffer;
use super::super::gated_delta::{
    recurrence_step_mutated, step_inner_literal, Mutation, RecurrenceStep,
};
use super::super::gated_delta::{state_geometry, DELTA_MATRIX};
use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::opplan::{GatedDeltaOp, OperandRef};

/// Qwen3.8's own head count is 48 at 128x128; the fixture keeps the
/// SHAPE (many heads, square state, `dv` a multiple of four) at a size a
/// test can run, because the shape is what the hoist is about.
const HV: usize = 6;
const DK: usize = 8;
const DV: usize = 8;

fn operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".into(),
        tensor: "fixture".into(),
        dtype: "F32".into(),
        shape: vec![1],
    }
}

fn op_with(hv: usize, dk: usize, dv: usize) -> GatedDeltaOp {
    GatedDeltaOp {
        num_key_heads: hv,
        num_value_heads: hv,
        key_head_dim: dk,
        value_head_dim: dv,
        conv_kernel: 4,
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

fn op() -> GatedDeltaOp {
    op_with(HV, DK, DV)
}

/// A state buffer seeded with values, so the comparison starts from
/// something the recurrence has to carry rather than from zeros — a bug
/// that dropped the incoming state would be invisible against a zero
/// seed.
fn seeded(op: &GatedDeltaOp, seed: u64) -> RecurrentBuffer {
    let geometry = &state_geometry(op)
        .expect("the fixture declares a state precision")
        .buffers[DELTA_MATRIX];
    let mut buffer = RecurrentBuffer::zeros(geometry);
    let values = lcg_values(buffer.cells().len(), seed);
    buffer.cells_mut().copy_from_slice(&values);
    buffer
}

struct Step {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

impl Step {
    fn new(seed: u64) -> Self {
        Self {
            query: lcg_values(HV * DK, seed),
            key: lcg_values(HV * DK, seed + 1),
            value: lcg_values(HV * DV, seed + 2),
            // `g` is exponentiated, so it must be negative for a decay in
            // (0, 1]; a positive gate would grow the state without bound
            // and the comparison would end up between two infinities.
            g: lcg_values(HV, seed + 3).iter().map(|x| -x.abs()).collect(),
            beta: lcg_values(HV, seed + 4).iter().map(|x| x.abs()).collect(),
        }
    }

    fn as_step(&self) -> RecurrenceStep<'_> {
        RecurrenceStep {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            g: &self.g,
            beta: &self.beta,
        }
    }
}

/// **The gate.** Same output, same state, bit for bit.
///
/// Every mutation the operator table carries is run too: the hoisted form
/// keeps the mutation branches, and a refactor that quietly dropped one
/// would still pass on the default path while making the whole mutation
/// table vacuous.
#[test]
fn the_hoisted_recurrence_is_bit_identical() {
    for mutation in [
        Mutation::None,
        Mutation::NoDecay,
        Mutation::RawGate,
        Mutation::NoBeta,
        Mutation::ScaleBeforeNorm,
        Mutation::ReadBeforeWrite,
    ] {
        let op = op();
        let step = Step::new(31);
        let (mut a, mut b) = (seeded(&op, 77), seeded(&op, 77));
        let out_a = recurrence_step_mutated(&op, &step.as_step(), &mut a, mutation);
        let out_b = step_inner_literal(&op, &step.as_step(), &mut b, mutation);
        assert_eq!(
            out_a, out_b,
            "{mutation:?}: the hoisted recurrence changed the OUTPUT — this rung moves no \
             arithmetic, so any difference is a defect and not a tolerance question"
        );
        assert_eq!(
            a.cells(),
            b.cells(),
            "{mutation:?}: the hoisted recurrence changed the STATE. The output above matched, \
             which is exactly how a corrupted continuation hides"
        );
    }
}

/// Bit-identity has to survive ITERATION, because that is what a decode
/// does.
///
/// One step agreeing proves the arithmetic; sixty-four steps agreeing
/// proves nothing is accumulating. A 1-ulp difference introduced at step
/// one and fed back into the state would be invisible in the test above
/// and unmistakable here.
#[test]
fn bit_identity_survives_sixty_four_continuation_steps() {
    let op = op();
    let (mut a, mut b) = (seeded(&op, 77), seeded(&op, 77));
    for t in 0..64u64 {
        let step = Step::new(100 + t * 7);
        let out_a = recurrence_step_mutated(&op, &step.as_step(), &mut a, Mutation::None);
        let out_b = step_inner_literal(&op, &step.as_step(), &mut b, Mutation::None);
        assert_eq!(out_a, out_b, "step {t}: outputs diverged");
        assert_eq!(a.cells(), b.cells(), "step {t}: state diverged");
        assert!(
            a.cells().iter().all(|x| x.is_finite()),
            "step {t}: the fixture's state left the normal range, so the comparison after this \
             point would be between two meaningless numbers"
        );
    }
}

/// **A/B/A/B in one binary.** Env-gated; never runs in CI.
///
/// One binary because a rebuild that touched only `workers_from` moved
/// this function 713 -> 615 us/call, 14%, without editing it — default
/// codegen-units, no LTO. Cross-build timing cannot see through that, and
/// alternating rounds are what stop a thermal or scheduling drift landing
/// entirely on whichever arm ran second.
///
/// ```text
/// QW_RECUR_AB=1 cargo test --release recurrence_shape -- --nocapture
/// ```
#[test]
fn hoisted_versus_literal_bench() {
    if std::env::var("QW_RECUR_AB").is_err() {
        eprintln!("SKIP hoisted_versus_literal_bench: set QW_RECUR_AB=1");
        return;
    }
    use std::time::Instant;
    // Qwen3.8's real geometry, not the fixture's.
    let (hv, dk, dv) = (48usize, 128usize, 128usize);
    let op = op_with(hv, dk, dv);

    let step = Step {
        query: lcg_values(hv * dk, 1),
        key: lcg_values(hv * dk, 2),
        value: lcg_values(hv * dv, 3),
        g: lcg_values(hv, 4).iter().map(|x| -x.abs() * 0.05).collect(),
        beta: lcg_values(hv, 5).iter().map(|x| x.abs().min(1.0)).collect(),
    };
    let iters = 200;

    let run = |f: &dyn Fn(&GatedDeltaOp, &RecurrenceStep<'_>, &mut RecurrentBuffer) -> Vec<f32>| {
        let mut state = seeded(&op, 9);
        for _ in 0..8 {
            f(&op, &step.as_step(), &mut state);
        }
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f(&op, &step.as_step(), &mut state));
        }
        t.elapsed().as_secs_f64() / iters as f64
    };

    let (mut hoisted, mut literal) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..4 {
        literal = literal.min(run(&|o, s, b| step_inner_literal(o, s, b, Mutation::None)));
        hoisted = hoisted.min(run(&|o, s, b| {
            recurrence_step_mutated(o, s, b, Mutation::None)
        }));
    }
    println!(
        "\n  Delta recurrence, {hv} heads x {dk}x{dv}, min of 4 alternating rounds\n\
         \n    literal (pre-CPU-2D1)  {:>8.1} us/layer\
         \n    hoisted                {:>8.1} us/layer   {:.2}x\
         \n\n  x{hv} layers: {:.2} ms/token against {:.2} ms\n",
        literal * 1e6,
        hoisted * 1e6,
        literal / hoisted,
        literal * 48.0 * 1e3,
        hoisted * 48.0 * 1e3,
    );
}

/// The hoisted form slices by `h * dk * dv + kk * dv + vv`; `cell` is the
/// written definition of that layout.
///
/// Pinned because the whole refactor rests on the identity. If `cell`
/// ever meant something else — a different axis order, padding between
/// heads — the hoisted loops would read the wrong cells while still
/// producing finite, plausible numbers, and the bit-identity gates above
/// would agree with each other because BOTH would be wrong.
#[test]
fn the_head_block_is_the_cell_formula() {
    let op = op();
    for h in 0..HV {
        for kk in 0..DK {
            for vv in 0..DV {
                assert_eq!(
                    super::super::gated_delta::cell(&op, h, kk, vv),
                    h * DK * DV + kk * DV + vv,
                    "cell({h}, {kk}, {vv}) is not the block the hoisted loops slice"
                );
            }
        }
    }
}
