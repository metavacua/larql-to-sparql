//! `nvfp4_matvec_v2` against v1: the same contract to fp32 rounding.
//!
//! v2 was the A-5 hypothesis "the v1 kernel is issue-bound on its
//! constant-memory LUT decode" — arithmetic decode, vector loads. The
//! single-command-buffer shape bench (`examples/nvfp4_gemv_shapes.rs`)
//! falsified it: v2 is numerically sound and not faster. It stays as an
//! explicit arm (`LARQL_NVFP4_KERNEL=v2`, default v1) under the shader
//! retention policy; this test keeps it honest while it is retained.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::{nvfp4_kernel_choice, MatvecOperands, Nvfp4Kernel};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

fn run_all(gpu: &MetalBackend, n: usize, k: usize, seed: usize) -> Vec<Vec<f32>> {
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i * 31 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, n, k).expect("quantise");
    let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01 - 0.05).collect();
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let xb = gpu.lowering_upload(&x).expect("x");
    let mut outs = Vec::new();
    for which in Nvfp4Kernel::ALL {
        let out = gpu.lowering_scratch(n);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let op = MatvecOperands {
            packed: &packed,
            scales: &scales,
            x: &xb,
            out: &out,
            out_offset: 0,
            n,
            k,
        };
        gpu.encode_nvfp4_kernel(which, enc, &op, m.tensor_scale);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        outs.push(gpu.lowering_readback(&out, n).expect("readback"));
        gpu.recycle_lowering_scratch(out);
    }
    gpu.recycle_lowering_scratch(xb);
    outs
}

fn run(gpu: &MetalBackend, n: usize, k: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i * 31 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, n, k).expect("quantise");
    let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01 - 0.05).collect();
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let xb = gpu.lowering_upload(&x).expect("x");
    let mut outs = Vec::new();
    for v2 in [false, true] {
        let out = gpu.lowering_scratch(n);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let op = MatvecOperands {
            packed: &packed,
            scales: &scales,
            x: &xb,
            out: &out,
            out_offset: 0,
            n,
            k,
        };
        if v2 {
            gpu.encode_nvfp4_matvec_v2(enc, &op, m.tensor_scale);
        } else {
            gpu.encode_nvfp4_matvec_v1(enc, &op, m.tensor_scale);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        outs.push(gpu.lowering_readback(&out, n).expect("readback"));
        gpu.recycle_lowering_scratch(out);
    }
    gpu.recycle_lowering_scratch(xb);
    let v1 = outs.remove(0);
    let v2 = outs.remove(0);
    (v1, v2)
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
fn v2_matches_v1_to_fp32_rounding_on_ledger_shapes() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Rows not a multiple of either geometry (8 or 4 rows/TG) included.
    for &(n, k) in &[
        (2560usize, 2560usize),
        (2112, 2816),
        (4099, 2880),
        (7, 6656),
    ] {
        let (v1, v2) = run(&gpu, n, k, n + k);
        let e = rel_rms(&v1, &v2);
        assert!(e < 1e-5, "[{n},{k}] rel_rms {e}");
        assert!(v1.iter().all(|v| v.is_finite()) && v2.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn every_sweep_arm_matches_v1_to_fp32_rounding() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // K not a multiple of 32·G groups and rows not a multiple of 8, so
    // the strided lane walk and the row guard are both exercised.
    for &(n, k) in &[(2113usize, 2816usize), (37, 1040)] {
        let outs = run_all(&gpu, n, k, n * 3 + k);
        for (i, which) in Nvfp4Kernel::ALL.iter().enumerate() {
            let e = rel_rms(&outs[0], &outs[i]);
            assert!(e < 1e-5, "{which:?} [{n},{k}] rel_rms {e}");
        }
    }
    // Names round-trip through the env spelling.
    for which in Nvfp4Kernel::ALL {
        assert_eq!(Nvfp4Kernel::parse(which.name()), Some(which));
    }
    assert_eq!(Nvfp4Kernel::parse("v9"), None);
}

#[test]
fn default_kernel_is_x2_unless_overridden() {
    // The choice is read once per process; with the variable unset the
    // default holds. (Tests that set the variable would need a fresh
    // process, so this pins only the default.)
    if std::env::var_os("LARQL_NVFP4_KERNEL").is_none() {
        assert_eq!(nvfp4_kernel_choice(), Nvfp4Kernel::X2);
    }
}

#[test]
fn x2_is_bit_identical_to_v1() {
    // Same per-row element order, same scale fold, nothing reassociated:
    // x2 reproduces v1 to the bit (the byte-LUT arms differ at ~1e-8).
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for &(n, k) in &[(2113usize, 2816usize), (37, 1040), (2560, 2560)] {
        let outs = run_all(&gpu, n, k, n + 7 * k);
        let v1 = &outs[0];
        let x2 = &outs[Nvfp4Kernel::ALL
            .iter()
            .position(|k| *k == Nvfp4Kernel::X2)
            .expect("x2 arm")];
        assert_eq!(v1, x2, "[{n},{k}] x2 diverged from v1");
    }
}
