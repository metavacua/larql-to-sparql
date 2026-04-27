//! Q4×Q8 matrix-vector multiply via C kernel.
//!
//! scores[N] = Q4[N, K] @ x[K]
//!
//! Internally quantizes x to Q8, then calls the C kernel with
//! ARM vdotq_s32 intrinsics. 0.95ms on 14.7MB matrix (M3 Max).

use super::q4_common::{q4_0_matvec_c, quantize_to_q8};

/// Q4 matvec: scores = Q4_matrix @ x.
/// Pre-quantizes x to Q8 internally.
pub fn dispatch(q4_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let (q8_x, q8_scales) = quantize_to_q8(x);
    dispatch_q8(q4_data, &q8_x, &q8_scales, num_rows, hidden)
}

/// Q4 matvec with pre-quantized Q8 input (avoids re-quantizing).
pub fn dispatch_q8(q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; num_rows];
    unsafe {
        q4_0_matvec_c(
            q4_data.as_ptr(), q8_x.as_ptr(), q8_scales.as_ptr(),
            scores.as_mut_ptr(), num_rows, hidden,
        );
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::q4_common::quantize_q4_0;

    #[test]
    fn q4_matvec_produces_output() {
        let hidden = 256;
        let rows = 64;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert_eq!(result.len(), rows);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_matvec_zero_input() {
        let hidden = 256;
        let rows = 32;
        let x = vec![0.0f32; hidden];
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }
}
