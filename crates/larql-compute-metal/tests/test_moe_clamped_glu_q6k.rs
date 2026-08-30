//! GPT-OSS-shaped MoE block parity: Q6_K experts at a writer-padded row
//! width, ClampedGlu gating, gate/up/down + router biases — the Metal
//! dispatch against `cpu_moe_forward` on the SAME `MoeLayerWeights`.
//!
//! This is the configuration the Metal path refused until the
//! `clamped_glu_bias` kernel and format-aware scratch existed; the test
//! pins that the whole block (routing included) agrees with the CPU
//! integer path, not merely that the shader math is plausible.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;

use common::get_metal;
use larql_compute::cpu::ops::moe::cpu_moe_forward;
use larql_compute_metal::MoeScratch;

// Deliberately NOT a super-block multiple: 288 stores as 512 columns, so a
// stride slip between the padded store and either backend moves every row
// after the first (the GPT-OSS 2880→3072 geometry, scaled down).
const HIDDEN: usize = 288;
const WEIGHT_COLS: usize = 512;
const INTER: usize = 320;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 4;
const EPS: f32 = 1e-6;
const LIMIT: f32 = 7.0;
const ALPHA: f32 = 1.702;

/// Hash-decorrelated fill — a smooth ramp survives permutations (M7).
fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((h >> 16) as f32 / 65536.0) - 0.5
        })
        .collect()
}

fn pad_rows(data: &[f32], rows: usize, cols: usize, padded: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * padded];
    for r in 0..rows {
        out[r * padded..r * padded + cols].copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    out
}

struct Fixture {
    experts_gate_up: Vec<Vec<u8>>,
    experts_down: Vec<Vec<u8>>,
    router: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
    pre_norm: Vec<f32>,
}

fn fixture() -> Fixture {
    let inter_padded = INTER.div_ceil(256) * 256;
    let mut experts_gate_up = Vec::with_capacity(NUM_EXPERTS);
    let mut experts_down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        // Writer layout: gate rows then up rows, each row padded to
        // WEIGHT_COLS; down padded to inter_padded columns.
        let gate = fill(INTER * HIDDEN, 1000 + e as u32);
        let up = fill(INTER * HIDDEN, 2000 + e as u32);
        let mut gu = pad_rows(&gate, INTER, HIDDEN, WEIGHT_COLS);
        gu.extend(pad_rows(&up, INTER, HIDDEN, WEIGHT_COLS));
        experts_gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q6_k(&gu));
        let down = fill(HIDDEN * INTER, 3000 + e as u32);
        let down_padded = pad_rows(&down, HIDDEN, INTER, inter_padded);
        experts_down.push(larql_compute::cpu::ops::q4_common::quantize_q6_k(
            &down_padded,
        ));
    }
    Fixture {
        experts_gate_up,
        experts_down,
        router: fill(NUM_EXPERTS * HIDDEN, 7),
        // Large enough to move selections for some tokens.
        router_bias: (0..NUM_EXPERTS).map(|e| (e as f32 - 3.5) * 0.3).collect(),
        gate_up_bias: fill(NUM_EXPERTS * 2 * INTER, 11),
        down_bias: fill(NUM_EXPERTS * HIDDEN, 13),
        pre_norm: (0..HIDDEN).map(|i| 1.0 + (i % 7) as f32 * 0.2).collect(),
    }
}

fn moe_weights(f: &Fixture) -> larql_compute::MoeLayerWeights<'_> {
    larql_compute::MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: f.experts_gate_up.iter().map(|v| v.as_slice()).collect(),
        experts_down: f.experts_down.iter().map(|v| v.as_slice()).collect(),
        routing_policy: larql_compute::MoeRoutingPolicy::top_k_then_softmax(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        expert_data_format: larql_compute::QuantFormat::Q6_K,
        router_proj: &f.router,
        router_bias: &f.router_bias,
        experts_gate_up_bias: &f.gate_up_bias,
        experts_down_bias: &f.down_bias,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &f.pre_norm,
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        intermediate_size: INTER,
        gate_rule: larql_compute::MoeGateRule::ClampedGlu {
            limit: LIMIT,
            alpha: ALPHA,
        },
    }
}

/// f32 reference over the DEQUANTISED store: same routing as production
/// (`cpu_moe_route`), then plain f32 expert math. This isolates the GPU
/// arm — no Q8_K activation step — so its bound can be tight.
fn f32_reference(h: &[f32], f: &Fixture, moe: &larql_compute::MoeLayerWeights<'_>) -> Vec<f32> {
    let inter_padded = INTER.div_ceil(256) * 256;
    let (indices, weights) = larql_compute::cpu::ops::moe::cpu_moe_route(h, moe, EPS);
    // moe_expert_input under PreExpertsNorm: rms_norm(h) ⊙ pre_norm.
    let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32 + EPS).sqrt();
    let mut xn: Vec<f32> = h
        .iter()
        .zip(&f.pre_norm)
        .map(|(v, w)| v / rms * w)
        .collect();
    xn.resize(WEIGHT_COLS, 0.0);
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut out = vec![0.0f32; HIDDEN];
    for (&ei, &w) in indices.iter().zip(&weights) {
        let gu = larql_models::quant::ggml::dequantize_q6_k(
            &f.experts_gate_up[ei],
            2 * INTER * WEIGHT_COLS,
        )
        .unwrap();
        let dn =
            larql_models::quant::ggml::dequantize_q6_k(&f.experts_down[ei], HIDDEN * inter_padded)
                .unwrap();
        let mut act = vec![0.0f32; inter_padded];
        for j in 0..INTER {
            let g: f32 = (0..WEIGHT_COLS)
                .map(|c| gu[j * WEIGHT_COLS + c] * xn[c])
                .sum::<f32>()
                + f.gate_up_bias[ei * 2 * INTER + 2 * j];
            let u: f32 = (0..WEIGHT_COLS)
                .map(|c| gu[(INTER + j) * WEIGHT_COLS + c] * xn[c])
                .sum::<f32>()
                + f.gate_up_bias[ei * 2 * INTER + 2 * j + 1];
            let g = g.min(LIMIT);
            let u = u.clamp(-LIMIT, LIMIT);
            act[j] = (u + 1.0) * (g * sigmoid(g * ALPHA));
        }
        for r in 0..HIDDEN {
            let d: f32 = (0..inter_padded)
                .map(|c| dn[r * inter_padded + c] * act[c])
                .sum();
            out[r] += w * (d + f.down_bias[ei * HIDDEN + r]);
        }
    }
    out
}

fn cos_and_rel_rms(a: &[f32], b: &[f32]) -> (f32, f32) {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    (dot / (na * nb).max(1e-30), num / nb.max(1e-30))
}

#[test]
fn metal_clamped_glu_q6k_block_matches_f32_reference() {
    let metal = get_metal();
    let f = fixture();
    let moe = moe_weights(&f);
    assert_eq!(
        moe.gate_up_cols(HIDDEN),
        WEIGHT_COLS,
        "fixture must exercise the padded-row geometry"
    );

    let scratch = MoeScratch::new_public_with_format(
        &metal,
        TOP_K,
        HIDDEN,
        INTER,
        larql_compute::QuantFormat::Q6_K,
        WEIGHT_COLS,
    );

    // Several residuals so more than one expert set is exercised.
    for seed in [21u32, 22, 23] {
        let h: Vec<f32> = fill(HIDDEN, seed).iter().map(|v| v * 2.5).collect();
        let gpu = metal.moe_block_for_layer(&h, &moe, EPS, &scratch, |ei| {
            Some((
                f.experts_gate_up[ei].as_slice(),
                f.experts_down[ei].as_slice(),
            ))
        });

        // Tight bound: GPU vs the f32-dequant reference — both read the
        // same Q6_K bytes and f32 activations, so only reduction order
        // separates them.
        let reference = f32_reference(&h, &f, &moe);
        let (cos_ref, rel_ref) = cos_and_rel_rms(&gpu, &reference);
        assert!(
            cos_ref > 0.99999 && rel_ref < 2e-3,
            "seed {seed}: Metal diverges from the f32 reference:              cos={cos_ref} rel_rms={rel_ref}"
        );

        // Loose bound: the CPU integer path Q8_K-quantises activations
        // (twice), which the GPU arm does not — ~1% rel_rms of documented
        // activation-quantisation noise, not a layout defect. A stride
        // slip would decorrelate the vectors (rel_rms ~ O(1)).
        let cpu = cpu_moe_forward(&h, &moe, 0.0, EPS);
        let (cos_cpu, rel_cpu) = cos_and_rel_rms(&gpu, &cpu);
        assert!(
            cos_cpu > 0.999 && rel_cpu < 0.03,
            "seed {seed}: Metal vs CPU-integer beyond activation-noise              band: cos={cos_cpu} rel_rms={rel_cpu}"
        );

        let na: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(na > 0.05, "fixture degenerated to ~zero output (na={na})");
    }
}

/// The bias plumbing is discriminating: zeroing the staged biases must
/// change the Metal output. A test that passes with and without biases
/// is an absent test (the fixture-adequacy rule).
#[test]
fn metal_biases_are_load_bearing() {
    let metal = get_metal();
    let f = fixture();
    let moe = moe_weights(&f);
    let no_bias = larql_compute::MoeLayerWeights {
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        ..moe_weights(&f)
    };
    let scratch = MoeScratch::new_public_with_format(
        &metal,
        TOP_K,
        HIDDEN,
        INTER,
        larql_compute::QuantFormat::Q6_K,
        WEIGHT_COLS,
    );
    let h: Vec<f32> = fill(HIDDEN, 31).iter().map(|v| v * 2.5).collect();
    let get = |ei: usize| {
        Some((
            f.experts_gate_up[ei].as_slice(),
            f.experts_down[ei].as_slice(),
        ))
    };
    let with = metal.moe_block_for_layer(&h, &moe, EPS, &scratch, get);
    let without = metal.moe_block_for_layer(&h, &no_bias, EPS, &scratch, get);
    let diff: f32 = with
        .iter()
        .zip(&without)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        diff > 1e-3,
        "biases did not change the Metal output (max|Δ|={diff}) — the slots are not wired"
    );
}
