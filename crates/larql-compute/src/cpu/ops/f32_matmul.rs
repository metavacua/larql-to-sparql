//! f32 matrix multiply via BLAS.
//!
//! On macOS: dispatches through Accelerate → AMX coprocessor.
//! On Linux: OpenBLAS or equivalent.
//! Single-core at 117 GB/s on M3 Max.

use ndarray::{Array2, ArrayView2};

/// C = A × B via BLAS sgemm.
pub fn matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.dot(&b)
}

/// C = A × B^T via BLAS sgemm.
pub fn matmul_transb(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.dot(&b.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[cfg(all(target_arch = "wasm32", feature = "browser-tests"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut s = seed;
        Array2::from_shape_fn((rows, cols), |_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn matmul_correct_shape() {
        let a = synth(6, 256, 42);
        let b = synth(256, 128, 43);
        let c = matmul(a.view(), b.view());
        assert_eq!(c.shape(), &[6, 128]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn matmul_transb_correct_shape() {
        let a = synth(6, 256, 42);
        let b = synth(128, 256, 43);
        let c = matmul_transb(a.view(), b.view());
        assert_eq!(c.shape(), &[6, 128]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn matmul_identity() {
        let a = synth(4, 4, 42);
        let eye = Array2::eye(4);
        let c = matmul(a.view(), eye.view());
        let diff: f32 = a.iter().zip(c.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff < 1e-5);
    }
}
