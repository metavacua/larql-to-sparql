use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::*;
use rand::prelude::*;

#[test]
fn test_standard_kv_exact_roundtrip() {
    let dim = 128;
    let n = 50;
    let mut rng = StdRng::seed_from_u64(42);

    let keys: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let values: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    let strategy = StandardKv;
    let encoded = strategy.encode(&keys, &values);
    let (dec_keys, dec_values) = strategy.decode(&encoded, n, dim);

    // FP16 roundtrip: expect small error but not exact
    for i in 0..n {
        for j in 0..dim {
            let err = (keys[i][j] - dec_keys[i][j]).abs();
            assert!(
                err < 0.01,
                "Key [{i}][{j}]: {:.6} vs {:.6}, err {:.6}",
                keys[i][j],
                dec_keys[i][j],
                err
            );
        }
    }

    for i in 0..n {
        for j in 0..dim {
            let err = (values[i][j] - dec_values[i][j]).abs();
            assert!(
                err < 0.01,
                "Value [{i}][{j}]: {:.6} vs {:.6}, err {:.6}",
                values[i][j],
                dec_values[i][j],
                err
            );
        }
    }
}

#[test]
fn test_standard_kv_memory_formula() {
    let config = ModelConfig::gemma_4b();

    // 4K tokens
    let mem = StandardKv.memory_bytes(&config, 4096);
    let expected = 4096 * 34 * 2 * 2 * 256 * 2;
    assert_eq!(mem, expected);

    // 370K tokens: ~25.8 GB — only valid on 64-bit (x86_64, aarch64).
    // On 32-bit (ARMv7, WASM), usize::MAX ≈ 4.3 GB so 370K tokens overflows.
    #[cfg(target_pointer_width = "64")]
    {
        let mem_370k = StandardKv.memory_bytes(&config, 370_000);
        let expected_370k = 370_000 * config.kv_bytes_per_token();
        assert_eq!(mem_370k, expected_370k);
        // 370K × 34L × 2(KV) × 2 heads × 256 dim × 2 bytes = ~25.8 GB
        assert!(mem_370k > 20_000_000_000);
        assert!(mem_370k < 30_000_000_000);
    }

    // On 32-bit targets (ARMv7, WASM in-browser), the maximum representable
    // KV cache without usize overflow is 61_681 tokens (~4.29 GB).
    // Verify the boundary: max_tokens fits, max_tokens+1 wraps.
    #[cfg(target_pointer_width = "32")]
    {
        let bytes_per_token = config.kv_bytes_per_token(); // 69_632
        let max_tokens = usize::MAX / bytes_per_token; // 61_681
        assert_eq!(max_tokens, 61_681);
        assert!((max_tokens + 1).checked_mul(bytes_per_token).is_none());
        let mem_max = StandardKv.memory_bytes(&config, max_tokens);
        assert_eq!(mem_max, max_tokens * bytes_per_token);
    }
}

#[test]
fn test_standard_kv_benchmark_runs() {
    let config = ModelConfig::gemma_4b();
    let strategy = StandardKv;
    let mut rng = StdRng::seed_from_u64(42);

    let result = kv_cache_benchmark::run_strategy_benchmark(&strategy, &config, 64, &mut rng);
    assert_eq!(result.strategy_name, "Standard KV (FP16)");
    assert_eq!(result.seq_len, 64);
    // MSE should be very small (FP16 quantization noise only)
    assert!(
        result.metrics.mse < 0.001,
        "MSE too high: {}",
        result.metrics.mse
    );
    // Cosine sim should be very high
    assert!(
        result.metrics.cosine_sim > 0.999,
        "Cosine too low: {}",
        result.metrics.cosine_sim
    );
}
