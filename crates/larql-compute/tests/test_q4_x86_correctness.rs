//! Numerical correctness for the scalar Q4_0 kernels in csrc/q4_dot.c.
//!
//! Compares `q4_matvec::dispatch` and `q4_vecmat::dispatch` output against a
//! pure-Rust dequantize-and-compute reference. Q4/Q8 are lossy, so we check
//! relative error and cosine similarity rather than exact agreement.

use larql_compute::cpu::q4::{q4_matvec, q4_vecmat, quantize_q4_0, quantize_to_q8};

/// Local f16→f32 (mirrors the decoder in q4_common.rs, not re-exported).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 { -val } else { val }
}

/// Dequantize a single Q4_0 row (blocks_per_row * 18 bytes) into f32.
fn dequantize_q4_0_row(row: &[u8], hidden: usize) -> Vec<f32> {
    let blocks = hidden / 32;
    let mut out = vec![0.0f32; hidden];
    for b in 0..blocks {
        let block = &row[b * 18..(b + 1) * 18];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[b * 32 + 2 * j]     = lo as f32 * scale;
            out[b * 32 + 2 * j + 1] = hi as f32 * scale;
        }
    }
    out
}

fn dequantize_q8(q8: &[i8], scales: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; q8.len()];
    for b in 0..scales.len() {
        let s = scales[b];
        for j in 0..32 {
            out[b * 32 + j] = q8[b * 32 + j] as f32 * s;
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_rel_err(kernel: &[f32], reference: &[f32]) -> f32 {
    let scale: f32 = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let denom = scale.max(1e-6);
    kernel.iter().zip(reference)
        .map(|(k, r)| (k - r).abs() / denom)
        .fold(0.0f32, f32::max)
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }).collect()
}

#[test]
fn q4_matvec_matches_dequant_reference() {
    let rows = 4;
    let hidden = 64;
    let matrix = synth(rows * hidden, 0xC0FFEE);
    let x = synth(hidden, 0xBEEF);

    let q4 = quantize_q4_0(&matrix);
    let kernel_out = q4_matvec(&q4, &x, rows, hidden);

    // Reference: dequantize both Q4 weights and Q8 input, do plain f32 matvec.
    // Using the Q8-requantized x (not the raw x) isolates the kernel's arithmetic
    // from the quantize_to_q8 step, which the kernel applies implicitly.
    let (q8_x, q8_scales) = quantize_to_q8(&x);
    let x_deq = dequantize_q8(&q8_x, &q8_scales);

    let bytes_per_row = (hidden / 32) * 18;
    let mut ref_out = vec![0.0f32; rows];
    for r in 0..rows {
        let row_deq = dequantize_q4_0_row(&q4[r * bytes_per_row..(r + 1) * bytes_per_row], hidden);
        ref_out[r] = row_deq.iter().zip(&x_deq).map(|(a, b)| a * b).sum();
    }

    let rel = max_rel_err(&kernel_out, &ref_out);
    let cos = cosine_similarity(&kernel_out, &ref_out);
    eprintln!("q4_matvec: max_rel_err={rel:.6e}, cos={cos:.6}");
    eprintln!("  kernel: {kernel_out:?}");
    eprintln!("  ref:    {ref_out:?}");

    // Dequant-and-multiply reference should agree with the kernel to within f32
    // rounding — both are doing the same math, just in different orders.
    assert!(rel < 1e-4, "max rel err {rel} exceeds 1e-4");
    assert!(cos > 0.9999, "cosine {cos} too low");
}

#[test]
fn q4_vecmat_matches_dequant_reference() {
    let intermediate = 8;
    let hidden = 64;
    let activation = synth(intermediate, 0xDEADBEEF);
    let matrix = synth(intermediate * hidden, 0xFEEDFACE);

    let q4 = quantize_q4_0(&matrix);
    let kernel_out = q4_vecmat(&activation, &q4, intermediate, hidden);

    // Reference: dequantize Q4, then do activation @ dequantized_matrix.
    let bytes_per_row = (hidden / 32) * 18;
    let mut ref_out = vec![0.0f32; hidden];
    for r in 0..intermediate {
        let row_deq = dequantize_q4_0_row(&q4[r * bytes_per_row..(r + 1) * bytes_per_row], hidden);
        let a = activation[r];
        for j in 0..hidden {
            ref_out[j] += a * row_deq[j];
        }
    }

    let rel = max_rel_err(&kernel_out, &ref_out);
    let cos = cosine_similarity(&kernel_out, &ref_out);
    eprintln!("q4_vecmat: max_rel_err={rel:.6e}, cos={cos:.6}");

    assert!(rel < 1e-4, "max rel err {rel} exceeds 1e-4");
    assert!(cos > 0.9999, "cosine {cos} too low");
}

#[test]
fn q4_matvec_vs_raw_f32_matvec_quant_noise() {
    // Looser bound: compare kernel output against the *original* f32 matvec
    // (before quantization). This captures total Q4/Q8 quantization noise.
    let rows = 4;
    let hidden = 64;
    let matrix = synth(rows * hidden, 0x1234);
    let x = synth(hidden, 0x5678);

    let q4 = quantize_q4_0(&matrix);
    let kernel_out = q4_matvec(&q4, &x, rows, hidden);

    let mut ref_out = vec![0.0f32; rows];
    for r in 0..rows {
        ref_out[r] = (0..hidden).map(|j| matrix[r * hidden + j] * x[j]).sum();
    }

    let cos = cosine_similarity(&kernel_out, &ref_out);
    eprintln!("q4_matvec vs raw f32: cos={cos:.6}");
    eprintln!("  kernel: {kernel_out:?}");
    eprintln!("  raw f32:{ref_out:?}");

    // Q4 (4-bit) + Q8 (8-bit) with random inputs — expect high cosine,
    // but not tight elementwise agreement.
    assert!(cos > 0.99, "cosine {cos} indicates kernel disagrees with f32 reference");
}
