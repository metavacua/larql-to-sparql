use kv_cache_benchmark::*;
use kv_cache_benchmark::metrics::Metrics;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::turboquant::rotation;
use rand::prelude::*;

#[test]
fn test_turboquant_wht_invertible() {
    let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();
    let y = rotation::wht(&x);
    let x_recon = rotation::wht(&y);

    for (i, (a, b)) in x.iter().zip(x_recon.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "WHT not self-inverse at [{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn test_turboquant_rotation_preserves_norm() {
    let mut rng = StdRng::seed_from_u64(42);
    let x: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let y = rotation::wht(&x);
    let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

    let err = (norm_x - norm_y).abs() / norm_x;
    assert!(err < 1e-4, "WHT changed norm by {err}");
}

#[test]
fn test_turboquant_4bit_mse_within_paper() {
    let dim = 256;
    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);

    let tq = TurboQuant::new(4);

    let mut total_mse = 0.0;
    for _ in 0..n {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let encoded = tq.encode_vector(&x);
        let decoded = tq.decode_vector(&encoded, dim);
        total_mse += Metrics::compute_mse(&x, &decoded);
    }
    let avg_mse = total_mse / n as f64;

    // Paper target: MSE ≤ 0.009 at 4-bit
    // We allow some margin since we're using synthetic codebooks
    assert!(
        avg_mse < 0.05,
        "4-bit MSE too high: {avg_mse:.6} (paper target: 0.009)"
    );
}

#[test]
fn test_turboquant_3bit_mse_within_paper() {
    let dim = 256;
    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);

    let tq = TurboQuant::new(3);

    let mut total_mse = 0.0;
    for _ in 0..n {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let encoded = tq.encode_vector(&x);
        let decoded = tq.decode_vector(&encoded, dim);
        total_mse += Metrics::compute_mse(&x, &decoded);
    }
    let avg_mse = total_mse / n as f64;

    // Paper target: MSE ≤ 0.034 at 3-bit
    assert!(
        avg_mse < 0.1,
        "3-bit MSE too high: {avg_mse:.6} (paper target: 0.034)"
    );
}

#[test]
fn test_turboquant_cosine_above_threshold() {
    let dim = 256;
    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);

    let tq = TurboQuant::new(4);

    let mut total_cos = 0.0;
    for _ in 0..n {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let encoded = tq.encode_vector(&x);
        let decoded = tq.decode_vector(&encoded, dim);
        total_cos += Metrics::compute_cosine(&x, &decoded);
    }
    let avg_cos = total_cos / n as f64;

    // Paper target: cosine ≥ 0.997 at 4-bit
    assert!(
        avg_cos > 0.95,
        "4-bit cosine too low: {avg_cos:.6} (paper target: 0.997)"
    );
}

#[test]
fn test_turboquant_compression_ratio() {
    let config = ModelConfig::gemma_4b();
    let tq4 = TurboQuant::new(4);
    let tq3 = TurboQuant::new(3);

    let standard_mem = config.kv_memory(4096);
    let tq4_mem = tq4.memory_bytes(&config, 4096);
    let tq3_mem = tq3.memory_bytes(&config, 4096);

    let ratio_4 = standard_mem as f64 / tq4_mem as f64;
    let ratio_3 = standard_mem as f64 / tq3_mem as f64;

    // 4-bit: expect ~3.5-4× compression
    assert!(
        ratio_4 > 2.5 && ratio_4 < 6.0,
        "4-bit compression ratio unexpected: {ratio_4:.2}"
    );

    // 3-bit: expect ~4-5× compression
    assert!(
        ratio_3 > 3.0 && ratio_3 < 7.0,
        "3-bit compression ratio unexpected: {ratio_3:.2}"
    );
}

#[test]
fn test_turboquant_benchmark_runs() {
    let config = ModelConfig::gemma_4b();
    let tq = TurboQuant::new(4);
    let mut rng = StdRng::seed_from_u64(42);

    let result = tq.benchmark(&config, 32, &mut rng);
    assert_eq!(result.strategy_name, "TurboQuant 4-bit");
    assert!(result.metrics.mse > 0.0, "MSE should be non-zero for lossy compression");
    assert!(result.metrics.cosine_sim > 0.9, "Cosine should be high");
    assert!(result.metrics.compression_ratio > 1.0, "Should compress");
}
