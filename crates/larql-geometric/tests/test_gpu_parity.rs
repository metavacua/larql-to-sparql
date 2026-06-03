// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu"))]
mod native_gpu {
    use larql_geometric::attention::{AttnInput, AttnBackend};
    use larql_geometric::attention::linear_complex::ComplexLinearAttn;
    use larql_geometric::gpu::linear_complex_gpu::ComplexLinearAttnGpu;

    #[test]
    fn test_complex_linear_cpu_gpu_parity() {
        let n = 8usize; let d = 16usize;
        let q: Vec<f32> = (0..n*d).map(|i| ((i as f32)*0.11).sin()).collect();
        let k: Vec<f32> = (0..n*d).map(|i| ((i as f32)*0.07+0.3).cos()).collect();
        let v: Vec<f32> = (0..n*d).map(|i| ((i as f32)*0.05).sin()*0.5).collect();
        let input = AttnInput { q: &q, k: &k, v: &v, seq_len: n, head_dim: d };

        let gpu = match ComplexLinearAttnGpu::new() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available — skipping GPU parity test");
                return;
            }
        };
        let cpu_out = ComplexLinearAttn::new().forward(&input);
        let gpu_out = gpu.forward(&input);
        for (c, g) in cpu_out.out.iter().zip(gpu_out.out.iter()) {
            assert!((c - g).abs() < 1e-3, "CPU={c} GPU={g} diff too large");
        }
    }
}
