use kv_cache_benchmark::*;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::markov_residual::MarkovResidual;

#[test]
fn test_markov_cold_tier_size() {
    let config = ModelConfig::gemma_4b();
    let strategy = MarkovResidual::new(512);

    // Cold tier: 4 bytes per token regardless of model size
    let mem_4k = strategy.memory_bytes(&config, 4096);
    let mem_370k = strategy.memory_bytes(&config, 370_000);

    // At 370K, cold tier dominates: 370K × 4 = 1.48 MB
    // Standard KV at 370K: ~56 GB
    let standard_370k = config.kv_memory(370_000);
    let ratio = standard_370k as f64 / mem_370k as f64;

    assert!(
        ratio > 100.0,
        "Markov RS should be >100× smaller than standard KV at 370K, got {ratio:.1}×"
    );
}

#[test]
fn test_markov_window_bounded() {
    let config = ModelConfig::gemma_4b();
    let strategy = MarkovResidual::new(512);

    // Memory at different context lengths should plateau
    let mem_4k = strategy.memory_bytes(&config, 4_096);
    let mem_32k = strategy.memory_bytes(&config, 32_768);
    let mem_370k = strategy.memory_bytes(&config, 370_000);

    // Window + checkpoint bytes are the same for all (bounded by window_size)
    // Only cold tier grows: (370K - 32K) × 4 bytes
    let growth = mem_370k - mem_32k;
    let expected_cold_growth = (370_000 - 32_768) * 4;
    assert_eq!(growth, expected_cold_growth);
}

#[test]
fn test_markov_much_smaller_than_standard() {
    let config = ModelConfig::gemma_4b();
    let standard = kv_cache_benchmark::standard_kv::StandardKv;
    let markov = MarkovResidual::new(512);

    for &seq_len in &[4096, 32768, 131072, 370_000] {
        let std_mem = standard.memory_bytes(&config, seq_len);
        let mrk_mem = markov.memory_bytes(&config, seq_len);
        assert!(
            mrk_mem < std_mem / 10,
            "At {seq_len} tokens: Markov RS ({mrk_mem}) should be <10% of Standard KV ({std_mem})"
        );
    }
}

#[test]
fn test_markov_encode_decode() {
    let strategy = MarkovResidual::new(4);
    let dim = 8;

    let keys: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32; dim])
        .collect();
    let values: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32 + 100.0; dim])
        .collect();

    let encoded = strategy.encode(&keys, &values);
    let (dec_keys, dec_values) = strategy.decode(&encoded, 10, dim);

    assert_eq!(dec_keys.len(), 10);

    // Cold tier vectors (first 6) should be zeros (simulating replay)
    for i in 0..6 {
        assert_eq!(dec_keys[i], vec![0.0f32; dim]);
    }

    // Window vectors (last 4) should match original keys
    for i in 6..10 {
        for j in 0..dim {
            assert!(
                (dec_keys[i][j] - keys[i][j]).abs() < 1e-6,
                "Window key [{i}][{j}] mismatch"
            );
        }
    }
}
