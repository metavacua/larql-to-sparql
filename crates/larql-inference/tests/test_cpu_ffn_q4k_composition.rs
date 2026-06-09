//! End-to-end production CPU FFN composition test.
//!
//! Replicates the operation sequence that `q4k_ffn_forward_layer_q8k`
//! (the walk-ffn-q8k server endpoint's CPU fallback) runs per layer,
//! without the `VectorIndex` dependency. Validates that the composition
//! of:
//!
//! 1. `quantize_x_to_q8k` (f32 → Q8_K activation)
//! 2. `q4k_q8k_gate_up_into` (fused Q4_K × Q8_K gate+up matvec)
//! 3. `silu_gate_up` (SiLU activation × up gating)
//! 4. `quantize_x_to_q8k` again (activated → Q8_K)
//! 5. `q4k_q8k_matvec_into` (Q4_K × Q8_K down projection)
//!
//! agrees with a dequant + f32 reference within the Q8_K activation
//! noise envelope. This is the smallest end-to-end test that covers
//! every AVX2 K-quant production path that landed in PRs #102–#116
//! without needing a real vindex on disk.

use larql_compute::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};
use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_gate_up_into, q4k_q8k_matvec_into, quantize_x_to_q8k,
};
use larql_inference::ffn::silu_gate_up;
use ndarray::Array2;

/// Reference FFN forward using f32 dequant + ndarray dot. This is the
/// numerical ground truth that the production AVX2 path must match
/// within Q8_K activation noise.
fn ffn_reference_f32(
    x: &[f32],
    gate_f32: &[f32],
    up_f32: &[f32],
    down_f32: &[f32],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let x_arr = Array2::from_shape_vec((1, hidden), x.to_vec()).unwrap();
    // gate / up are stored as [inter rows × hidden cols]; FFN forward
    // computes `gate = x · gate_w^T` (1 × inter).
    let gate_w = Array2::from_shape_vec((inter, hidden), gate_f32.to_vec()).unwrap();
    let up_w = Array2::from_shape_vec((inter, hidden), up_f32.to_vec()).unwrap();
    let gate = x_arr.dot(&gate_w.t());
    let up = x_arr.dot(&up_w.t());
    let activated = silu_gate_up(&gate, &up);
    // down stored as [hidden rows × inter cols]; `out = activated · down_w^T`.
    let down_w = Array2::from_shape_vec((hidden, inter), down_f32.to_vec()).unwrap();
    let out = activated.dot(&down_w.t());
    out.into_raw_vec_and_offset().0
}

/// Production FFN forward — the AVX2 K-quant family's full composition,
/// matching what `q4k_ffn_forward_layer_q8k` calls per layer in
/// production.
fn ffn_production_q4k_q8k(
    x: &[f32],
    gate_q4k: &[u8],
    up_q4k: &[u8],
    down_q4k: &[u8],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    // 1. Quantise input to Q8_K.
    let q8k_x = quantize_x_to_q8k(x);

    // 2. Fused gate+up matvec (AVX2 on x86_64 post-#112).
    let mut gate_flat = vec![0.0f32; inter];
    let mut up_flat = vec![0.0f32; inter];
    q4k_q8k_gate_up_into(
        &mut gate_flat,
        &mut up_flat,
        &q8k_x,
        gate_q4k,
        up_q4k,
        inter,
        hidden,
    );

    // 3. SiLU(gate) ⊙ up — same activation `q4k_ffn_forward_layer_q8k`
    //    applies (`crate::ffn::silu_gate_up`).
    let gate = Array2::from_shape_vec((1, inter), gate_flat).unwrap();
    let up = Array2::from_shape_vec((1, inter), up_flat).unwrap();
    let activated = silu_gate_up(&gate, &up);

    // 4. Re-quantise activated to Q8_K (production allocates once per
    //    layer, here we just call the standard helper).
    let activation_flat = activated.as_slice().unwrap();
    let act_q8k = quantize_x_to_q8k(activation_flat);

    // 5. Down projection (AVX2 on x86_64).
    let mut out = vec![0.0f32; hidden];
    q4k_q8k_matvec_into(&mut out, &act_q8k, down_q4k, hidden, inter);
    out
}

/// Q4_K-only FFN composition end-to-end. Uses Q4_K for all three FFN
/// matrices (gate / up / down) — covers the common "all-Q4_K" vindex
/// extracted with `--down-q4k`. The Q6_K-down variant is covered
/// separately in `test_cpu_ffn_q4k_q6k_down_composition`.
#[test]
fn cpu_ffn_q4k_composition_matches_dequant_f32_reference() {
    // Small shape: hidden=256 (one super-block), inter=512 (two
    // super-blocks). Q4_K requires K to be a multiple of 256 in both
    // matrix dims; pick the smallest shapes that still exercise
    // multi-superblock paths in down.
    let hidden = 256usize;
    let inter = 512usize;

    // Synthetic but non-trivial weights: smooth oscillations with
    // moderate amplitude, avoid outliers that would dominate the
    // amax scaling and produce trivial dequant.
    let gate_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.011).sin() * 0.4)
        .collect();
    let up_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.013).cos() * 0.4)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| (i as f32 * 0.009).sin() * 0.3)
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.017).sin() * 1.2)
        .collect();

    // Quantise each weight matrix to Q4_K.
    let gate_q4k = quantize_q4_k(&gate_f32);
    let up_q4k = quantize_q4_k(&up_f32);
    let down_q4k = quantize_q4_k(&down_f32);

    // Reference: f32 throughout, but using the same dequantised
    // weights the production path will see. This isolates Q8_K
    // activation noise from Q4_K weight noise (which both paths share).
    let gate_deq = dequantize_q4_k(&gate_q4k, inter * hidden);
    let up_deq = dequantize_q4_k(&up_q4k, inter * hidden);
    let down_deq = dequantize_q4_k(&down_q4k, hidden * inter);

    let ref_out = ffn_reference_f32(&x, &gate_deq, &up_deq, &down_deq, hidden, inter);
    let got = ffn_production_q4k_q8k(&x, &gate_q4k, &up_q4k, &down_q4k, hidden, inter);

    assert_eq!(ref_out.len(), hidden);
    assert_eq!(got.len(), hidden);

    // Cosine similarity is the primary correctness gate — direction
    // must match almost exactly even though magnitudes drift under
    // compound Q8_K activation noise (gate/up + SiLU + down chains
    // three quantised dots, so per-element noise can reach 5-6 % on
    // smooth inputs whose outputs cluster near zero amplitude).
    let dot: f32 = ref_out.iter().zip(&got).map(|(a, b)| a * b).sum();
    let na = (ref_out.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let nb = (got.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let cos = dot / (na * nb);
    assert!(
        cos > 0.9995,
        "FFN composition cosine similarity {cos:.6} < 0.9995"
    );
}

/// Mixed Q4_K (gate/up) + Q6_K (down) — the *default* vindex
/// extraction layout when `--down-q4k=false`. Exercises the format
/// dispatch added in #103 indirectly: down is Q6_K-quantised here
/// and processed via the Q6_K AVX2 kernel.
#[test]
fn cpu_ffn_q4k_q6k_down_composition_matches_dequant_f32_reference() {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use larql_compute::cpu::ops::q4k_q8k_dot::q6k_q8k_matvec_into;
    use larql_models::quant::ggml::dequantize_q6_k;

    let hidden = 256usize;
    let inter = 512usize;

    let gate_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.011).sin() * 0.4)
        .collect();
    let up_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.013).cos() * 0.4)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| (i as f32 * 0.009).sin() * 0.3)
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.017).sin() * 1.2)
        .collect();

    let gate_q4k = quantize_q4_k(&gate_f32);
    let up_q4k = quantize_q4_k(&up_f32);
    let down_q6k = quantize_q6_k(&down_f32);

    let gate_deq = dequantize_q4_k(&gate_q4k, inter * hidden);
    let up_deq = dequantize_q4_k(&up_q4k, inter * hidden);
    let down_deq = dequantize_q6_k(&down_q6k, hidden * inter).expect("dequant q6_k");

    let ref_out = ffn_reference_f32(&x, &gate_deq, &up_deq, &down_deq, hidden, inter);

    // Production path with Q6_K down — replicates the post-#103
    // walk-ffn-q8k Q6_K branch.
    let q8k_x = quantize_x_to_q8k(&x);
    let mut gate_flat = vec![0.0f32; inter];
    let mut up_flat = vec![0.0f32; inter];
    q4k_q8k_gate_up_into(
        &mut gate_flat,
        &mut up_flat,
        &q8k_x,
        &gate_q4k,
        &up_q4k,
        inter,
        hidden,
    );
    let gate = Array2::from_shape_vec((1, inter), gate_flat).unwrap();
    let up = Array2::from_shape_vec((1, inter), up_flat).unwrap();
    let activated = silu_gate_up(&gate, &up);
    let act_q8k = quantize_x_to_q8k(activated.as_slice().unwrap());
    let mut got = vec![0.0f32; hidden];
    q6k_q8k_matvec_into(&mut got, &act_q8k, &down_q6k, hidden, inter);

    let dot: f32 = ref_out.iter().zip(&got).map(|(a, b)| a * b).sum();
    let na = (ref_out.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let nb = (got.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let cos = dot / (na * nb);
    assert!(
        cos > 0.9995,
        "Mixed Q4_K+Q6_K FFN composition cosine similarity {cos:.6} < 0.9995"
    );
}
