//! Q4 vector-matrix multiply via C kernel (scatter-accumulate).
//!
//! out[K] = activation[N] @ Q4[N, K]

use super::q4_common::q4_0_vecmat_c;

/// Q4 vecmat: out = activation @ Q4_matrix.
pub fn dispatch(activation: &[f32], q4_data: &[u8], intermediate: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    unsafe {
        q4_0_vecmat_c(
            activation.as_ptr(), q4_data.as_ptr(),
            out.as_mut_ptr(), intermediate, hidden,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::q4_common::quantize_q4_0;

    #[test]
    fn q4_vecmat_produces_output() {
        let hidden = 256;
        let inter = 128;
        let act: Vec<f32> = (0..inter).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert_eq!(result.len(), hidden);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_vecmat_zero_activation() {
        let hidden = 256;
        let inter = 64;
        let act = vec![0.0f32; inter];
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }
}
