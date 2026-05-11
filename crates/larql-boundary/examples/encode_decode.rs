//! Encode and decode a synthetic residual with both bf16 and int8_clip3sigma.
//!
//! Run with:
//!   cargo run -p larql-boundary --example encode_decode

use larql_boundary::codec::{bf16, int8};

fn main() {
    let d = 2560_usize;

    // Synthetic residual with outliers (resembles Gemma 3 4B final-layer residuals).
    let mut residual: Vec<f32> = (0..d).map(|i| (i as f32 * 0.01).sin() * 80.0).collect();
    residual[100] = 94_208.0; // outlier — absmax for Gemma-class residuals
    residual[500] = -60_000.0; // outlier

    let bf16_bytes = bf16::encode(&residual);
    let bf16_decoded = bf16::decode(&bf16_bytes);

    let int8_payload = int8::encode(&residual);
    let int8_bytes = int8_payload.to_bytes();
    let int8_decoded = int8::decode(&int8_payload);

    let bf16_mse: f32 = residual
        .iter()
        .zip(bf16_decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / d as f32;

    let int8_mse: f32 = residual
        .iter()
        .zip(int8_decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / d as f32;

    println!("d = {d}");
    println!();
    println!(
        "bf16:           {} bytes  ({:.1}× vs bf16)  MSE = {bf16_mse:.4}",
        bf16_bytes.len(),
        1.0_f32
    );
    println!(
        "int8_clip3σ:    {} bytes  ({:.1}× vs bf16)  MSE = {int8_mse:.4}",
        int8_bytes.len(),
        bf16_bytes.len() as f32 / int8_bytes.len() as f32
    );
    println!();
    println!("Note: int8_clip3σ saturates outliers — MSE is dominated by");
    println!("the two outlier elements at indices 100 and 500.");
    println!();
    println!("Non-outlier MSE (excluding indices 100, 500):");
    let no_outlier_mse: f32 = residual
        .iter()
        .zip(int8_decoded.iter())
        .enumerate()
        .filter(|(i, _)| *i != 100 && *i != 500)
        .map(|(_, (a, b))| (a - b).powi(2))
        .sum::<f32>()
        / (d - 2) as f32;
    println!("  int8_clip3σ non-outlier MSE = {no_outlier_mse:.4}");
}
