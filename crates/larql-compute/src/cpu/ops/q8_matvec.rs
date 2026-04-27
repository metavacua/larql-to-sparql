//! Q8 matrix-vector multiply.
//!
//! scores[N] = Q8_weights[N, K] @ Q8_x[K]
//!
//! Simpler than Q4 — no nibble unpacking. Each weight is one signed byte.
//! Used for V projection where Q4 accuracy is insufficient.


/// Quantize a weight matrix to Q8 format: int8 values + per-block f32 scales.
/// Returns (int8_data[N*K], scales[N * K/32]).
pub fn quantize_weights_q8(weights: &[f32], num_rows: usize, hidden: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(weights.len(), num_rows * hidden);
    assert!(hidden.is_multiple_of(32));

    let blocks_per_row = hidden / 32;
    let mut q8_data = vec![0i8; num_rows * hidden];
    let mut scales = vec![0.0f32; num_rows * blocks_per_row];

    for r in 0..num_rows {
        for b in 0..blocks_per_row {
            let off = r * hidden + b * 32;
            let block = &weights[off..off + 32];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales[r * blocks_per_row + b] = scale;
            for j in 0..32 {
                q8_data[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    (q8_data, scales)
}

/// Q8 matvec on CPU: scores[N] = Q8_w[N,K] @ Q8_x[K].
pub fn dispatch(
    w_q8: &[i8], w_scales: &[f32],
    x_q8: &[i8], x_scales: &[f32],
    num_rows: usize, hidden: usize,
) -> Vec<f32> {
    let blocks = hidden / 32;
    let mut scores = vec![0.0f32; num_rows];

    for r in 0..num_rows {
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let combined_scale = w_scales[r * blocks + b] * x_scales[b];
            let w_off = r * hidden + b * 32;
            let x_off = b * 32;
            let mut isum = 0i32;
            for j in 0..32 {
                isum += w_q8[w_off + j] as i32 * x_q8[x_off + j] as i32;
            }
            acc += isum as f32 * combined_scale;
        }
        scores[r] = acc;
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::q4::quantize_to_q8;

    #[test]
    fn q8_matvec_produces_output() {
        let hidden = 256;
        let rows = 64;
        let weights: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

        let (w_q8, w_scales) = quantize_weights_q8(&weights, rows, hidden);
        let (x_q8, x_scales) = quantize_to_q8(&x);

        let result = dispatch(&w_q8, &w_scales, &x_q8, &x_scales, rows, hidden);
        assert_eq!(result.len(), rows);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q8_vs_f32_high_cosine() {
        let hidden = 256;
        let rows = 32;
        let weights: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

        // f32 reference
        let mut f32_result = vec![0.0f32; rows];
        for r in 0..rows {
            for c in 0..hidden {
                f32_result[r] += weights[r * hidden + c] * x[c];
            }
        }

        // Q8
        let (w_q8, w_scales) = quantize_weights_q8(&weights, rows, hidden);
        let (x_q8, x_scales) = quantize_to_q8(&x);
        let q8_result = dispatch(&w_q8, &w_scales, &x_q8, &x_scales, rows, hidden);

        // Cosine similarity
        let dot: f32 = f32_result.iter().zip(q8_result.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = f32_result.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = q8_result.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);

        assert!(cos > 0.999, "Q8 cosine {cos} should be > 0.999");
    }
}
