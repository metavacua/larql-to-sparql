//! Diagnostic: take a real Q4_K weight (Qwen 3.6 35B-A3B layer 0
//! attn_qkv.weight), a bit-exact x input (our dumped `dn_x_norm_l00.bin`),
//! compute the matmul three ways:
//!
//! 1. **F32 dequant + ndarray dot** (this is our reference)
//! 2. **q4k_row_dot** (the f32-activation precise kernel)
//! 3. **q4k_q8k_row_dot** (the Q8K-activation fast kernel)
//!
//! Then compares all three against llama.cpp's
//! `linear_attn_qkv_mixed-0.bin` (token 0 slice) to see which has the
//! 0.1% precision loss and where it comes from.
//!
//! Usage:
//!   target/release/examples/q4k_matmul_parity \
//!     /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v5 \
//!     /tmp/larql-dump/tok0/dn_x_norm_l00.bin \
//!     /tmp/llama-dump/linear_attn_qkv_mixed-0.bin

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use larql_models::quant::ggml::{dequantize_q4_k, q4k_q8k_row_dot, q4k_row_dot, quantize_to_q8_k};

fn read_f32_bin(path: &str) -> Vec<f32> {
    let mut f = fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).unwrap();
    let ne0 = i64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
    let ne1 = i64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let ne2 = i64::from_le_bytes(hdr[16..24].try_into().unwrap()) as usize;
    let ne3 = i64::from_le_bytes(hdr[24..32].try_into().unwrap()) as usize;
    let total = ne0 * ne1 * ne2 * ne3;
    let mut buf = vec![0u8; total * 4];
    f.read_exact(&mut buf).unwrap();
    let mut out = Vec::with_capacity(total);
    for chunk in buf.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    out
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let dx: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let dy: f32 = b.iter().map(|x| (x - mb).powi(2)).sum::<f32>().sqrt();
    if dx > 0.0 && dy > 0.0 {
        num / (dx * dy)
    } else {
        0.0
    }
}

fn stats(label: &str, ours: &[f32], lc: &[f32]) {
    let l2_o: f32 = ours.iter().map(|v| v * v).sum::<f32>().sqrt();
    let l2_l: f32 = lc.iter().map(|v| v * v).sum::<f32>().sqrt();
    let pr = pearson(ours, lc);
    let mut max_d = 0.0_f32;
    let mut sum_d = 0.0_f32;
    for (a, b) in ours.iter().zip(lc) {
        let d = (a - b).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
    }
    let mean_d = sum_d / ours.len() as f32;
    println!(
        "  {label}: l2_o={l2_o:.4} l2_l={l2_l:.4} ratio={:.6} pearson={pr:.6} max|d|={max_d:.4} mean|d|={mean_d:.4}",
        l2_o / l2_l
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <vindex_dir> <x_norm.bin> <llama_qkv_mixed.bin>",
            args[0]
        );
        std::process::exit(2);
    }
    let vindex_dir = PathBuf::from(&args[1]);
    let x_path = &args[2];
    let lc_path = &args[3];

    // Read layer 0 attn_qkv (Q4_K, shape [8192, 2048])
    let dn_q4k = fs::read(vindex_dir.join("deltanet_weights_q4k.bin"))
        .expect("read deltanet_weights_q4k.bin");
    // From manifest: layers.0.self_attn.qkv_proj.weight at offset 0, length 9437184
    let rows = 8192_usize;
    let cols = 2048_usize;
    let length = 9437184_usize;
    assert_eq!(rows * cols * 144 / 256, length, "size check");
    let weight_bytes = &dn_q4k[0..length];

    // Read inputs
    let x = read_f32_bin(x_path);
    assert_eq!(x.len(), cols, "x_norm size {}", x.len());
    let lc_full = read_f32_bin(lc_path);
    let lc_t0 = &lc_full[0..rows];
    println!("=== L0 attn_qkv matmul parity ===");
    println!(
        "x ({} f32, l2={:.4})",
        x.len(),
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    );
    println!(
        "W is [{rows}, {cols}] Q4_K, total {} bytes",
        weight_bytes.len()
    );

    // Path 1: dequantize whole W to F32, then ndarray dot.
    let w_f32 = dequantize_q4_k(weight_bytes, rows * cols).expect("dequant");
    assert_eq!(w_f32.len(), rows * cols);
    let mut y_f32 = vec![0.0_f32; rows];
    for r in 0..rows {
        let row = &w_f32[r * cols..(r + 1) * cols];
        let mut acc = 0.0_f32;
        for j in 0..cols {
            acc += row[j] * x[j];
        }
        y_f32[r] = acc;
    }
    stats("F32 dequant + naive scalar dot", &y_f32, lc_t0);

    // Path 2: q4k_row_dot (uses Q4_K bytes + f32 x)
    let row_bytes = (cols / 256) * 144;
    let mut y_q4 = vec![0.0_f32; rows];
    for r in 0..rows {
        let q_row = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        y_q4[r] = q4k_row_dot(q_row, &x).expect("q4k_row_dot");
    }
    stats("q4k_row_dot (Q4_K + f32 x)", &y_q4, lc_t0);

    // Path 3: q4k_q8k_row_dot (Q4_K weight + Q8_K activation)
    let q8k_x = quantize_to_q8_k(&x);
    let mut y_q8 = vec![0.0_f32; rows];
    for r in 0..rows {
        let q_row = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        y_q8[r] = q4k_q8k_row_dot(q_row, &q8k_x).expect("q4k_q8k_row_dot");
    }
    stats("q4k_q8k_row_dot (Q4_K + Q8_K x)", &y_q8, lc_t0);

    // Path 1 vs Path 2 internal consistency
    println!();
    println!("=== Internal consistency (us vs us) ===");
    stats("F32 naive vs q4k_row_dot", &y_f32, &y_q4);
    stats("F32 naive vs q4k_q8k_row_dot", &y_f32, &y_q8);
    stats("q4k_row_dot vs q4k_q8k_row_dot", &y_q4, &y_q8);
}
