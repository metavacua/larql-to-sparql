//! Differential test: `ExpertWeightFfn` against the GPT-OSS reference.
//!
//! Pins the whole MoE block — fused-tensor de-interleaving, router bias,
//! select-then-normalise routing, the clamped GLU, and both expert biases —
//! against values produced by an independent transcription of
//! `transformers/models/gpt_oss/modeling_gpt_oss.py`
//! (`GptOssTopKRouter.forward` + `GptOssExperts._apply_gate`).
//!
//! **No fixture file.** Both sides draw their weights from the same LCG in the
//! same order, so the Python script and this test see bit-identical inputs.
//! The generator is `scripts/moe_reference_gpt_oss.py`; re-run it and paste the
//! result into [`EXPECTED`] if the draw order below ever changes.
//!
//! Why this test is the load-bearing one: every individual piece has a unit
//! test, but the bug this replaced was a *composition* error — a plausible
//! decode feeding a plausible split feeding a plausible activation, each
//! defensible alone and collectively wrong. Only an end-to-end diff against
//! the reference catches that. See `docs/k3-funnel.md` §4.7.

use larql_compute::ffn::{ExpertWeightFfn, FfnBackend};
use larql_models::quant::mxfp4::{deinterleave_fused_half, FusedHalf};
use larql_models::{ModelWeights, WeightArray};
use ndarray::Array2;

const HIDDEN: usize = 16;
const INTER: usize = 16;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const TOKENS: usize = 3;
const FUSED_ROWS: usize = 2 * INTER;

const SEED: u64 = 0x5EED_1234;

/// Tolerance on the block output. The two sides sum in different orders
/// (ndarray/BLAS vs NumPy), so this is fp-reordering slack, not modelling
/// slack — the values are O(0.1), so this is ~1e-4 relative.
const TOLERANCE: f32 = 1e-5;

/// The same LCG the Python reference script uses.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
    }

    fn flat(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next(scale)).collect()
    }

    fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_vec((rows, cols), self.flat(rows * cols, scale))
            .expect("shape matches element count")
    }
}

fn shared(a: Array2<f32>) -> WeightArray {
    a.into_shared()
}

/// Reference output, row-major `[TOKENS × HIDDEN]`, computed by the Python
/// transcription of the reference implementation on these exact weights.
///
/// The three tokens do not all route to the same experts (token 2 selects a
/// different pair), so this also pins per-token routing rather than a single
/// shared expert set.
#[rustfmt::skip]
const EXPECTED: [f32; TOKENS * HIDDEN] = [
    0.0344635, -0.0400323, 0.0006023, 0.1556825,
    0.339892, -0.1307565, 0.0531546, 0.0065341,
    -0.030165, -0.2066685, -0.0477355, -0.0586864,
    -0.1378887, -0.0682759, 0.066816, 0.2042877,
    0.0570752, -0.0207001, 0.0021008, 0.1934435,
    0.232725, -0.1195951, 0.0449907, -0.0104086,
    -0.0585297, -0.2317356, -0.1011433, -0.0392903,
    -0.1277727, 0.0283705, 0.0246793, 0.1729313,
    -0.0957333, -0.0193308, 0.1625361, 0.0848202,
    0.0952437, -0.0922684, 0.0869211, 0.1446012,
    -0.0495669, -0.1051358, -0.1286211, 0.0502055,
    -0.0381496, -0.1930531, -0.113367, 0.0628057,
];

/// Build the GPT-OSS-shaped weights, drawing in the reference script's order.
///
/// Gate and up are split out of a **fused** tensor via
/// [`deinterleave_fused_half`], exactly as the MXFP4 loader does — so this
/// test covers the de-interleaving convention, not just the arithmetic after
/// it. Feeding the contiguous-halves split here instead would reproduce the
/// original bug and fail.
fn build_weights(x: &mut Array2<f32>) -> ModelWeights {
    let mut rng = Lcg(SEED);

    *x = rng.mat(TOKENS, HIDDEN, 0.5);
    let router_w = rng.mat(EXPERTS, HIDDEN, 0.4);
    let router_b = rng.flat(EXPERTS, 0.3);

    let fused: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|_| rng.flat(FUSED_ROWS * HIDDEN, 0.3))
        .collect();
    let fused_bias: Vec<Vec<f32>> = (0..EXPERTS).map(|_| rng.flat(FUSED_ROWS, 0.2)).collect();
    let down: Vec<Array2<f32>> = (0..EXPERTS).map(|_| rng.mat(HIDDEN, INTER, 0.3)).collect();
    let down_bias: Vec<Vec<f32>> = (0..EXPERTS).map(|_| rng.flat(HIDDEN, 0.2)).collect();

    let mut weights = larql_models::test_fixtures::make_test_weights();
    weights.arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "num_hidden_layers": 1,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": HIDDEN,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "swiglu_limit": 7.0,
    }));
    weights.hidden_size = HIDDEN;
    weights.intermediate_size = INTER;
    weights.num_layers = 1;

    let arch_keys: Vec<(String, String, String)> = (0..EXPERTS)
        .map(|e| {
            (
                weights.arch.expert_ffn_gate_key(0, e).unwrap(),
                weights.arch.expert_ffn_up_key(0, e).unwrap(),
                weights.arch.expert_ffn_down_key(0, e).unwrap(),
            )
        })
        .collect();
    let router_key = weights.arch.moe_router_key(0).unwrap();
    let router_bias_key = weights.arch.moe_router_bias_key(0).unwrap();
    let gate_up_bias_key = weights.arch.packed_gate_up_bias_key(0).unwrap();
    let down_bias_key = weights.arch.packed_down_bias_key(0).unwrap();

    weights.tensors.insert(router_key, shared(router_w));
    weights.vectors.insert(router_bias_key, router_b);

    for (e, (gate_key, up_key, down_key)) in arch_keys.into_iter().enumerate() {
        let gate = deinterleave_fused_half(&fused[e], FUSED_ROWS, HIDDEN, FusedHalf::Gate);
        let up = deinterleave_fused_half(&fused[e], FUSED_ROWS, HIDDEN, FusedHalf::Up);
        weights.tensors.insert(
            gate_key,
            shared(Array2::from_shape_vec((INTER, HIDDEN), gate).unwrap()),
        );
        weights.tensors.insert(
            up_key,
            shared(Array2::from_shape_vec((INTER, HIDDEN), up).unwrap()),
        );
        weights.tensors.insert(down_key, shared(down[e].clone()));
    }

    // Per-expert bias rows, stacked exactly as the checkpoint stores them:
    // `[num_experts, 2 * intermediate]` and `[num_experts, hidden]`.
    let stack = |rows: &[Vec<f32>], cols: usize| {
        shared(
            Array2::from_shape_vec(
                (EXPERTS, cols),
                rows.iter().flat_map(|r| r.iter().copied()).collect(),
            )
            .unwrap(),
        )
    };
    weights
        .tensors
        .insert(gate_up_bias_key, stack(&fused_bias, FUSED_ROWS));
    weights
        .tensors
        .insert(down_bias_key, stack(&down_bias, HIDDEN));

    weights
}

#[test]
fn moe_block_matches_the_gpt_oss_reference_implementation() {
    let mut x = Array2::<f32>::zeros((TOKENS, HIDDEN));
    let weights = build_weights(&mut x);

    assert!(
        larql_compute::ffn::expert_weight::resolvable(&weights, 0),
        "fixture must present a resolvable per-expert MoE layer"
    );

    let got = ExpertWeightFfn { weights: &weights }.forward(0, &x);
    assert_eq!(got.shape(), &[TOKENS, HIDDEN]);

    let mut max_diff = 0.0f32;
    for (i, (&g, &want)) in got.iter().zip(EXPECTED.iter()).enumerate() {
        let diff = (g - want).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < TOLERANCE,
            "element {i} (token {}, dim {}): got {g}, reference says {want} (diff {diff})",
            i / HIDDEN,
            i % HIDDEN
        );
    }
    assert!(max_diff < TOLERANCE, "max divergence {max_diff}");
}

/// The contiguous-halves split this replaced must not reproduce the reference.
///
/// Without this, the parity test above could pass for the wrong reason if the
/// two splits happened to coincide — as they do at `out_features = 2`, which
/// is exactly why the original unit tests never caught the bug.
#[test]
fn the_contiguous_halves_split_would_fail_this_test() {
    let mut x = Array2::<f32>::zeros((TOKENS, HIDDEN));
    let mut weights = build_weights(&mut x);

    // Re-derive the fused tensors and install the WRONG split.
    let mut rng = Lcg(SEED);
    let _ = rng.flat(TOKENS * HIDDEN, 0.5);
    let _ = rng.flat(EXPERTS * HIDDEN, 0.4);
    let _ = rng.flat(EXPERTS, 0.3);
    let fused: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|_| rng.flat(FUSED_ROWS * HIDDEN, 0.3))
        .collect();

    let half = INTER * HIDDEN;
    for (e, expert) in fused.iter().enumerate() {
        let gate_key = weights.arch.expert_ffn_gate_key(0, e).unwrap();
        let up_key = weights.arch.expert_ffn_up_key(0, e).unwrap();
        weights.tensors.insert(
            gate_key,
            shared(Array2::from_shape_vec((INTER, HIDDEN), expert[..half].to_vec()).unwrap()),
        );
        weights.tensors.insert(
            up_key,
            shared(Array2::from_shape_vec((INTER, HIDDEN), expert[half..].to_vec()).unwrap()),
        );
    }

    let got = ExpertWeightFfn { weights: &weights }.forward(0, &x);
    let max_diff = got
        .iter()
        .zip(EXPECTED.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "the wrong split should diverge visibly, but max diff was only {max_diff}"
    );
}
