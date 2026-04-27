//! Auto-calibration: benchmark CPU vs Metal to find the FLOP threshold
//! where GPU dispatch starts beating CPU BLAS.

use ndarray::Array2;
use std::time::Instant;

use super::f32_ops::F32Ops;
use super::buffers::BufferCache;
use metal::CommandQueue;

/// Conservative default before calibration runs.
pub const DEFAULT_FLOP_THRESHOLD: usize = 500_000_000;

/// Absolute floor: never dispatch to GPU below this.
pub const MIN_FLOP_FLOOR: usize = 100_000;

/// Run calibration and return the optimal FLOP threshold.
pub fn calibrate(
    f32_ops: &F32Ops,
    queue: &CommandQueue,
    bufs: &BufferCache,
) -> usize {
    let test_cases: &[(usize, usize, usize)] = &[
        (6, 256, 256),       // ~800K FLOPs
        (6, 2560, 512),      // ~15M FLOPs
        (6, 2560, 2560),     // ~79M FLOPs — attention projection
        (6, 10240, 2560),    // ~315M FLOPs — FFN gate/up
    ];

    let mut best = DEFAULT_FLOP_THRESHOLD;

    for &(m, n, k) in test_cases {
        let flops = 2 * m * n * k;
        let a = synth_matrix(m, k, 42);
        let b = synth_matrix(n, k, 43);

        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();

        // Warm Metal buffer cache
        let _ = f32_ops.dispatch_transb(queue, bufs, a_slice, b_slice, m, n, k);

        let cpu_us = bench_median(5, || { let _ = a.dot(&b.t()); });
        let metal_us = bench_median(5, || {
            let _ = f32_ops.dispatch_transb(queue, bufs, a_slice, b_slice, m, n, k);
        });

        if metal_us < cpu_us {
            best = best.min(flops);
        }
    }

    best
}

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn bench_median<F: FnMut()>(n: usize, mut f: F) -> u64 {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_micros() as u64);
    }
    times.sort_unstable();
    times[n / 2]
}
