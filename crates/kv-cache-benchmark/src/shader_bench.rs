//! Phase 3: Shader benchmark harness.
//!
//! Measures per-operation GPU timing across the benchmark matrix:
//!   Operation              CPU f32    Metal fused    Graph only
//!   WHT encode             ✓          ✓              n/a
//!   Lloyd-Max quant        ✓          ✓              n/a
//!   TQ encode (fused)      ✓          ✓              n/a
//!   TQ decode (fused)      ✓          ✓              n/a
//!   Gate KNN               ✓          ✓              ✓
//!   Sparse FFN walk        ✓          ✓              n/a

use crate::turboquant::TurboQuant;
use crate::turboquant::rotation;
use crate::metrics::Metrics;

/// Benchmark result for a single operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShaderBenchResult {
    pub operation: String,
    pub backend: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub time_us: f64,
    pub throughput_ops_per_sec: f64,
}

/// Run CPU WHT benchmark at given dimension.
pub fn bench_wht_cpu(dim: usize, iterations: usize) -> ShaderBenchResult {
    let x: Vec<f32> = (0..dim).map(|i| (i as f32 - dim as f32 / 2.0) / 100.0).collect();

    let t0 = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = rotation::wht(&x);
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_op = elapsed_us / iterations as f64;

    ShaderBenchResult {
        operation: format!("WHT d={dim}"),
        backend: "CPU f32".to_string(),
        dimension: dim,
        batch_size: 1,
        time_us: per_op,
        throughput_ops_per_sec: 1e6 / per_op,
    }
}

/// Run CPU TurboQuant encode benchmark.
pub fn bench_tq_encode_cpu(dim: usize, bits: u8, batch: usize) -> ShaderBenchResult {
    let tq = TurboQuant::new(bits);
    let vectors: Vec<Vec<f32>> = (0..batch)
        .map(|seed| {
            (0..dim)
                .map(|i| ((seed * dim + i) as f32 * 0.01) - 0.5)
                .collect()
        })
        .collect();

    let t0 = std::time::Instant::now();
    for v in &vectors {
        let _ = tq.encode_vector(v);
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_op = elapsed_us / batch as f64;

    ShaderBenchResult {
        operation: format!("TQ encode {bits}bit d={dim}"),
        backend: "CPU f32".to_string(),
        dimension: dim,
        batch_size: batch,
        time_us: per_op,
        throughput_ops_per_sec: batch as f64 * 1e6 / elapsed_us,
    }
}

/// Run CPU TurboQuant decode benchmark.
pub fn bench_tq_decode_cpu(dim: usize, bits: u8, batch: usize) -> ShaderBenchResult {
    let tq = TurboQuant::new(bits);
    let vectors: Vec<Vec<f32>> = (0..batch)
        .map(|seed| {
            (0..dim)
                .map(|i| ((seed * dim + i) as f32 * 0.01) - 0.5)
                .collect()
        })
        .collect();

    // Pre-encode
    let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| tq.encode_vector(v)).collect();

    let t0 = std::time::Instant::now();
    for enc in &encoded {
        let _ = tq.decode_vector(enc, dim);
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_op = elapsed_us / batch as f64;

    ShaderBenchResult {
        operation: format!("TQ decode {bits}bit d={dim}"),
        backend: "CPU f32".to_string(),
        dimension: dim,
        batch_size: batch,
        time_us: per_op,
        throughput_ops_per_sec: batch as f64 * 1e6 / elapsed_us,
    }
}

/// Run CPU TurboQuant encode+decode roundtrip with accuracy.
pub fn bench_tq_roundtrip_cpu(dim: usize, bits: u8, batch: usize) -> (ShaderBenchResult, f64, f64) {
    let tq = TurboQuant::new(bits);
    // Use unit-scale vectors (matching real model KV vector magnitudes)
    let vectors: Vec<Vec<f32>> = (0..batch)
        .map(|seed| {
            (0..dim)
                .map(|i| {
                    let x = ((seed * dim + i) as f32 * 2_654_435.8).sin();
                    x * 0.5 // [-0.5, 0.5] range, similar to real KV vectors
                })
                .collect()
        })
        .collect();

    let t0 = std::time::Instant::now();
    let mut total_mse = 0.0;
    let mut total_cosine = 0.0;
    for v in &vectors {
        let encoded = tq.encode_vector(v);
        let decoded = tq.decode_vector(&encoded, dim);
        total_mse += Metrics::compute_mse(v, &decoded);
        total_cosine += Metrics::compute_cosine(v, &decoded);
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;

    let result = ShaderBenchResult {
        operation: format!("TQ roundtrip {bits}bit d={dim}"),
        backend: "CPU f32".to_string(),
        dimension: dim,
        batch_size: batch,
        time_us: elapsed_us / batch as f64,
        throughput_ops_per_sec: batch as f64 * 1e6 / elapsed_us,
    };

    let avg_mse = total_mse / batch as f64;
    let avg_cosine = total_cosine / batch as f64;
    (result, avg_mse, avg_cosine)
}

/// Run the full CPU shader benchmark suite.
pub fn run_cpu_benchmark_suite() -> Vec<ShaderBenchResult> {
    let mut results = Vec::new();

    // WHT benchmarks
    for dim in [128, 256] {
        results.push(bench_wht_cpu(dim, 10000));
    }

    // TurboQuant encode/decode
    for bits in [3, 4] {
        for dim in [128, 256] {
            results.push(bench_tq_encode_cpu(dim, bits, 1000));
            results.push(bench_tq_decode_cpu(dim, bits, 1000));
        }
    }

    results
}

/// Format benchmark results as a table.
pub fn format_shader_results(results: &[ShaderBenchResult]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Shader Benchmark Results ===\n\n");
    out.push_str(&format!(
        "{:<30} {:>10} {:>10} {:>10} {:>15}\n",
        "Operation", "Backend", "Dim", "Time (us)", "Ops/sec"
    ));
    out.push_str(&"-".repeat(80));
    out.push('\n');

    for r in results {
        out.push_str(&format!(
            "{:<30} {:>10} {:>10} {:>10.2} {:>15.0}\n",
            r.operation, r.backend, r.dimension, r.time_us, r.throughput_ops_per_sec,
        ));
    }
    out
}
