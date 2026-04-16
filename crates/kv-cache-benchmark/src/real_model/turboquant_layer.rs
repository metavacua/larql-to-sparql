//! TurboQuant applied to real K/V tensors from the forward pass.
//!
//! Intercepts K/V capture, quantizes each head vector via WHT + Lloyd-Max,
//! then dequantizes on read. Measures MSE, cosine, and compression vs FP16.

use ndarray::Array2;
use crate::turboquant::TurboQuant;
use crate::metrics::Metrics;
use super::kv_capture::KvCapture;

/// Result of applying TurboQuant to captured K/V.
pub struct TurboQuantResult {
    /// Compressed bytes for all K/V across all layers.
    pub compressed_bytes: usize,
    /// Original FP16 bytes.
    pub original_bytes: usize,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Mean MSE across all vectors.
    pub mse: f64,
    /// Mean cosine similarity across all vectors.
    pub cosine_sim: f64,
    /// Encode time in microseconds.
    pub encode_us: f64,
    /// Decode time in microseconds.
    pub decode_us: f64,
    /// Reconstructed K tensors (for downstream accuracy comparison).
    pub decoded_keys: Vec<Array2<f32>>,
    /// Reconstructed V tensors.
    pub decoded_values: Vec<Array2<f32>>,
}

/// Apply TurboQuant to captured K/V tensors.
/// Quantizes each per-head vector independently (matching the paper's algorithm).
pub fn apply_turboquant(capture: &KvCapture, tq: &TurboQuant) -> TurboQuantResult {
    let num_layers = capture.num_layers;
    let mut total_compressed = 0usize;
    let mut total_original = 0usize;
    let mut total_mse = 0.0f64;
    let mut total_cosine = 0.0f64;
    let mut vector_count = 0usize;
    let mut total_encode_us = 0.0f64;
    let mut total_decode_us = 0.0f64;

    let mut decoded_keys = Vec::with_capacity(num_layers);
    let mut decoded_values = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let k = &capture.keys[layer];
        let v = &capture.values[layer];

        let (dk, enc_bytes_k, enc_us_k, dec_us_k, mse_k, cos_k, count_k) =
            quantize_tensor(k, tq);
        let (dv, enc_bytes_v, enc_us_v, dec_us_v, mse_v, cos_v, count_v) =
            quantize_tensor(v, tq);

        total_compressed += enc_bytes_k + enc_bytes_v;
        total_original += (k.len() + v.len()) * 2; // FP16
        total_mse += mse_k + mse_v;
        total_cosine += cos_k + cos_v;
        vector_count += count_k + count_v;
        total_encode_us += enc_us_k + enc_us_v;
        total_decode_us += dec_us_k + dec_us_v;

        decoded_keys.push(dk);
        decoded_values.push(dv);
    }

    let avg_mse = if vector_count > 0 { total_mse / vector_count as f64 } else { 0.0 };
    let avg_cosine = if vector_count > 0 { total_cosine / vector_count as f64 } else { 0.0 };
    let compression = if total_compressed > 0 {
        total_original as f64 / total_compressed as f64
    } else {
        0.0
    };

    TurboQuantResult {
        compressed_bytes: total_compressed,
        original_bytes: total_original,
        compression_ratio: compression,
        mse: avg_mse,
        cosine_sim: avg_cosine,
        encode_us: total_encode_us,
        decode_us: total_decode_us,
        decoded_keys,
        decoded_values,
    }
}

/// Quantize a 2D tensor row-by-row, return decoded tensor + metrics.
fn quantize_tensor(
    tensor: &Array2<f32>,
    tq: &TurboQuant,
) -> (Array2<f32>, usize, f64, f64, f64, f64, usize) {
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);

    // We need power-of-2 dimension for WHT. head_dim is typically 128 or 256.
    // If cols isn't directly a power of 2, we work per-head.
    // For GQA: tensor is [seq_len, num_kv_heads * head_dim], process each head_dim chunk.
    let head_dim = find_power_of_two_dim(cols);
    let num_heads = cols / head_dim;

    let mut decoded = Array2::<f32>::zeros((rows, cols));
    let mut total_encoded_bytes = 0usize;
    let mut total_mse = 0.0f64;
    let mut total_cosine = 0.0f64;
    let mut count = 0usize;
    let mut encode_us = 0.0f64;
    let mut decode_us = 0.0f64;

    for row in 0..rows {
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let vec: Vec<f32> = tensor.row(row).slice(ndarray::s![start..end]).to_vec();

            let t0 = std::time::Instant::now();
            let encoded = tq.encode_vector(&vec);
            encode_us += t0.elapsed().as_secs_f64() * 1e6;

            total_encoded_bytes += encoded.len();

            let t0 = std::time::Instant::now();
            let decoded_vec = tq.decode_vector(&encoded, head_dim);
            decode_us += t0.elapsed().as_secs_f64() * 1e6;

            total_mse += Metrics::compute_mse(&vec, &decoded_vec);
            total_cosine += Metrics::compute_cosine(&vec, &decoded_vec);
            count += 1;

            for (j, &val) in decoded_vec.iter().enumerate() {
                decoded[[row, start + j]] = val;
            }
        }
    }

    (decoded, total_encoded_bytes, encode_us, decode_us, total_mse, total_cosine, count)
}

/// Find the largest power-of-2 that divides cols (for WHT compatibility).
fn find_power_of_two_dim(cols: usize) -> usize {
    // Common head dims: 64, 128, 256
    for &candidate in &[256, 128, 64, 32] {
        if cols % candidate == 0 {
            return candidate;
        }
    }
    // Fallback: use cols if it's a power of 2
    if cols.is_power_of_two() {
        return cols;
    }
    panic!("Cannot find power-of-2 head dimension for cols={cols}");
}
