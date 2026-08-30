//! A-5b rung 2 kernels against the separate dispatches they replace:
//! - `x2r` (residual folded into the write) — bit-identical to x2 then
//!   `residual_add`;
//! - `rms_norm_multi3` — bit-identical to three `rms_norm` dispatches;
//! - `x2n` / `x2m` (pre-norm folded in, both forms) — to fp32 rounding
//!   (different sum-of-squares order); both measured slower and are
//!   retained as arms, so the contract is pinned while they exist.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::{MatvecOperands, NormOutput, Nvfp4Kernel, PreNorm};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const K: usize = 2816;
const N: usize = 2113;

fn fixture(gpu: &MetalBackend) -> (nvfp4::Nvfp4Matrix, Vec<f32>, metal::Buffer) {
    let values: Vec<f32> = (0..N * K)
        .map(|i| (((i * 13) % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, N, K).expect("quantise");
    let x: Vec<f32> = (0..K).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    (m, x, xb)
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn x2r_equals_x2_then_residual_add_bit_for_bit() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (m, _, xb) = fixture(&gpu);
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let res: Vec<f32> = (0..N).map(|i| (i % 7) as f32 - 3.0).collect();
    let rb = gpu.lowering_upload(&res).expect("residual");
    let tmp = gpu.lowering_scratch(N);
    let want = gpu.lowering_scratch(N);
    let got = gpu.lowering_scratch(N);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let op = |out| MatvecOperands {
        packed: &packed,
        scales: &scales,
        x: &xb,
        out,
        out_offset: 0,
        n: N,
        k: K,
    };
    gpu.encode_nvfp4_kernel(Nvfp4Kernel::X2, enc, &op(&tmp), m.tensor_scale);
    gpu.encode_residual_add(enc, &rb, &tmp, &want, N, 1.0);
    gpu.encode_nvfp4_matvec_residual(enc, &op(&got), m.tensor_scale, &rb);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let w = gpu.lowering_readback(&want, N).unwrap();
    let g = gpu.lowering_readback(&got, N).unwrap();
    assert_eq!(w, g);
}

#[test]
fn multi_norm_equals_three_norms_bit_for_bit() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (_, _, xb) = fixture(&gpu);
    let weights: Vec<Vec<f32>> = (0..3)
        .map(|j| (0..K).map(|i| ((i + j) % 5) as f32 * 0.3).collect())
        .collect();
    let wb: Vec<_> = weights
        .iter()
        .map(|w| gpu.lowering_upload(w).unwrap())
        .collect();
    let offsets = [1.0f32, 0.0, 1.0];
    let eps = 1e-6f32;
    let want: Vec<_> = (0..3).map(|_| gpu.lowering_scratch(K)).collect();
    let got: Vec<_> = (0..3).map(|_| gpu.lowering_scratch(K)).collect();
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for j in 0..3 {
        larql_compute_metal::stages::input_norm::encode_f32(
            enc,
            &gpu.norms.rms_norm_pipeline,
            &xb,
            0,
            &wb[j],
            &want[j],
            0,
            K,
            eps,
            offsets[j],
        );
    }
    let outs: Vec<NormOutput<'_>> = (0..3)
        .map(|j| NormOutput {
            weight: &wb[j],
            offset: offsets[j],
            out: &got[j],
        })
        .collect();
    gpu.encode_rms_norm_multi(enc, &xb, K, eps, &outs);
    // One and two outputs also run (the absent slots alias the first).
    gpu.encode_rms_norm_multi(enc, &xb, K, eps, &outs[..1]);
    gpu.encode_rms_norm_multi(enc, &xb, K, eps, &outs[..2]);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    for j in 0..3 {
        let w = gpu.lowering_readback(&want[j], K).unwrap();
        let g = gpu.lowering_readback(&got[j], K).unwrap();
        assert_eq!(w, g, "output {j}");
    }
}

#[test]
fn prenorm_forms_match_norm_then_x2_to_fp32_rounding() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (m, _, xb) = fixture(&gpu);
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let wn: Vec<f32> = (0..K).map(|i| (i % 9) as f32 * 0.25).collect();
    let wnb = gpu.lowering_upload(&wn).unwrap();
    let normed = gpu.lowering_scratch(K);
    let want = gpu.lowering_scratch(N);
    let got_a = gpu.lowering_scratch(N);
    let got_b = gpu.lowering_scratch(N);
    let (eps, off) = (1e-6f32, 1.0f32);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    larql_compute_metal::stages::input_norm::encode_f32(
        enc,
        &gpu.norms.rms_norm_pipeline,
        &xb,
        0,
        &wnb,
        &normed,
        0,
        K,
        eps,
        off,
    );
    gpu.encode_nvfp4_kernel(
        Nvfp4Kernel::X2,
        enc,
        &MatvecOperands {
            packed: &packed,
            scales: &scales,
            x: &normed,
            out: &want,
            out_offset: 0,
            n: N,
            k: K,
        },
        m.tensor_scale,
    );
    let norm = PreNorm {
        weight: &wnb,
        eps,
        offset: off,
    };
    let op = |out| MatvecOperands {
        packed: &packed,
        scales: &scales,
        x: &xb,
        out,
        out_offset: 0,
        n: N,
        k: K,
    };
    gpu.encode_nvfp4_matvec_prenorm(enc, &op(&got_a), m.tensor_scale, &norm);
    gpu.encode_nvfp4_matvec_prenorm_staged(enc, &op(&got_b), m.tensor_scale, &norm);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let w = gpu.lowering_readback(&want, N).unwrap();
    let a = gpu.lowering_readback(&got_a, N).unwrap();
    let b = gpu.lowering_readback(&got_b, N).unwrap();
    assert!(rel_rms(&w, &a) < 1e-5, "form A {}", rel_rms(&w, &a));
    assert!(rel_rms(&w, &b) < 1e-5, "form B {}", rel_rms(&w, &b));
}
