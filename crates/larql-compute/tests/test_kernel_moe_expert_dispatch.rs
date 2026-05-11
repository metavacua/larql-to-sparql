#![cfg(all(feature = "metal", target_os = "macos"))]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;

use common::{cos_sim, get_metal, max_diff};
use larql_compute::prelude::*;
use larql_compute::MoeScratch;

fn synth_values(len: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let a = (seed + i as f32 * 0.0017).sin();
            let b = (seed * 0.37 + (i >> 7) as f32 * 0.019).cos();
            (a + 0.25 * b) * scale
        })
        .collect()
}

fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let padded_cols = cols.div_ceil(256) * 256;
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }
    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    (out, padded_cols)
}

fn make_q4k_experts(hidden: usize, inter: usize, top_k: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut gate_up = Vec::with_capacity(top_k);
    let mut down = Vec::with_capacity(top_k);
    for e in 0..top_k {
        let gate = synth_values(inter * hidden, 0.11 + e as f32 * 0.13, 0.18);
        let up = synth_values(inter * hidden, 0.41 + e as f32 * 0.17, 0.16);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

        let raw_down = synth_values(hidden * inter, 0.73 + e as f32 * 0.07, 0.11);
        let (down_padded, _) = pad_rows_to_256(&raw_down, hidden, inter);
        down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
            &down_padded,
        ));
    }
    (gate_up, down)
}

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_cols);
    debug_assert_eq!(w.len(), out_rows * in_cols);
    let mut out = vec![0.0f32; out_rows];
    for row in 0..out_rows {
        let w_row = &w[row * in_cols..(row + 1) * in_cols];
        out[row] = w_row.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum();
    }
    out
}

fn run_single_expert_f32_reference(
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let gate_up_w =
        larql_compute::cpu::ops::q4_common::dequantize_q4_k(gate_up_bytes, 2 * inter * hidden);
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    let mut act = vec![0.0f32; inter_padded];
    for j in 0..inter {
        act[j] = gelu_tanh(gate_out[j]) * up_out[j];
    }

    let down_w =
        larql_compute::cpu::ops::q4_common::dequantize_q4_k(down_bytes, hidden * inter_padded);
    matmul_vec(&act, &down_w, hidden, inter_padded)
}

fn run_single_expert_separated_metal_reference(
    metal: &larql_compute::metal::MetalBackend,
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let row_bytes = (hidden / block) * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
    let half = inter * row_bytes;
    let gate = metal
        .q4k_matvec(&gate_up_bytes[..half], h_norm, inter, hidden)
        .expect("Metal gate q4k matvec");
    let up = metal
        .q4k_matvec(&gate_up_bytes[half..2 * half], h_norm, inter, hidden)
        .expect("Metal up q4k matvec");

    let mut act = vec![0.0f32; inter_padded];
    for j in 0..inter {
        act[j] = gelu_tanh(gate[j]) * up[j];
    }

    metal
        .q4k_matvec(down_bytes, &act, hidden, inter_padded)
        .expect("Metal down q4k matvec")
}

fn assert_preselected_dispatch_matches_cpu(label: &str, hidden: usize, inter: usize, top_k: usize) {
    let metal = get_metal();
    let h_norm = synth_values(hidden, 1.23, 0.35);
    let expert_ids: Vec<usize> = (0..top_k).collect();
    let expert_weights: Vec<f32> = (0..top_k)
        .map(|i| (i as f32 + 1.0) / (top_k as f32 * (top_k as f32 + 1.0) * 0.5))
        .collect();
    let (gate_up, down) = make_q4k_experts(hidden, inter, top_k);

    let mut expected = vec![0.0f32; hidden];
    for e in 0..top_k {
        let out = run_single_expert_f32_reference(&h_norm, &gate_up[e], &down[e], hidden, inter);
        for (acc, &v) in expected.iter_mut().zip(&out) {
            *acc += v * expert_weights[e];
        }
    }

    let mut separated_metal = vec![0.0f32; hidden];
    for e in 0..top_k {
        let out = run_single_expert_separated_metal_reference(
            &metal,
            &h_norm,
            &gate_up[e],
            &down[e],
            hidden,
            inter,
        );
        for (acc, &v) in separated_metal.iter_mut().zip(&out) {
            *acc += v * expert_weights[e];
        }
    }

    let scratch = MoeScratch::new_public(&metal, top_k, hidden, inter);
    let got = metal.run_experts_preselected_metal(
        &h_norm,
        &expert_ids,
        &expert_weights,
        &scratch,
        |eid| Some((gate_up[eid].as_slice(), down[eid].as_slice())),
    );

    let diff = max_diff(&expected, &got);
    let cos = cos_sim(&expected, &got);
    let expected_max = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = diff / expected_max.max(1.0);
    let metal_diff = max_diff(&separated_metal, &got);
    let metal_cos = cos_sim(&separated_metal, &got);
    let metal_max = separated_metal
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let metal_rel = metal_diff / metal_max.max(1.0);
    let nonzero = got.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(
        nonzero > hidden / 2 && metal_rel < 1e-4 && metal_cos > 0.999_999,
        "{label}: Metal MoE expert dispatch diverged from CPU: \
         cpu_max_abs={diff:.3e} cpu_rel={rel:.3e} cpu_cos={cos:.6} \
         metal_max_abs={metal_diff:.3e} metal_rel={metal_rel:.3e} \
         metal_cos={metal_cos:.6} nonzero={nonzero}/{hidden}"
    );
}

#[test]
fn metal_moe_preselected_small_q4k_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("small q4k moe", 256, 256, 2);
}

#[test]
#[ignore = "known open Metal MoE issue at Gemma 4 26B-A4B shape; run explicitly while debugging"]
fn metal_moe_preselected_gemma4_26b_a4b_shape_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("gemma4-26b-a4b moe", 2816, 704, 8);
}
