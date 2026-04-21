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

    // MarkovRS W=512 hot window costs ~192 MB (fixed).
    // At short contexts that's not much smaller than standard KV.
    // The benefit is that it stays FLAT while standard KV grows O(n).
    // At 32K+ the window is a fraction of standard KV.
    for &seq_len in &[32768, 131072, 370_000] {
        let std_mem = standard.memory_bytes(&config, seq_len);
        let mrk_mem = markov.memory_bytes(&config, seq_len);
        assert!(
            mrk_mem < std_mem / 10,
            "At {seq_len} tokens: Markov RS ({mrk_mem}) should be <10% of Standard KV ({std_mem})"
        );
    }

    // At 4K the window still dominates, but MarkovRS is still smaller than standard.
    let std_4k = standard.memory_bytes(&config, 4096);
    let mrk_4k = markov.memory_bytes(&config, 4096);
    assert!(mrk_4k < std_4k, "Markov RS should be smaller than standard KV at 4K");
}

#[test]
fn test_boundary_residual_always_flat() {
    let config = ModelConfig::gemma_4b();
    let standard = kv_cache_benchmark::standard_kv::StandardKv;
    let boundary = kv_cache_benchmark::boundary_residual::BoundaryResidual::gemma_4b();

    // BoundaryRS W=32 is always much smaller: ~11 MB hot + tiny cold IDs.
    // At 4K it's ~25× smaller; at 370K it's ~2000× smaller.
    for &seq_len in &[4096, 32768, 131072, 370_000] {
        let std_mem = standard.memory_bytes(&config, seq_len);
        let brs_mem = boundary.memory_bytes(&config, seq_len);
        assert!(
            brs_mem * 20 < std_mem,
            "At {seq_len}: Boundary RS ({brs_mem}) should be >20× smaller than Standard KV ({std_mem})"
        );
    }
    // At 370K it's genuinely ~2000× compression.
    let std_370k = standard.memory_bytes(&config, 370_000) as f64;
    let brs_370k = boundary.memory_bytes(&config, 370_000) as f64;
    assert!(std_370k / brs_370k > 1000.0,
        "At 370K: compression ratio should exceed 1000× (got {:.0}×)", std_370k / brs_370k);
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

#[test]
fn test_markov_reconstruction_exact() {
    // Within the active window, residuals are stored at full precision.
    // Reconstruction from the window should be bit-perfect (KL = 0.0).
    let strategy = MarkovResidual::new(512);
    let dim = 64;
    let n = 100; // All within window

    let keys: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect())
        .collect();
    let values: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.02).collect())
        .collect();

    let encoded = strategy.encode(&keys, &values);
    let (dec_keys, _dec_values) = strategy.decode(&encoded, n, dim);

    // All within window — should be exact (bit-perfect)
    for i in 0..n {
        for j in 0..dim {
            assert!(
                (dec_keys[i][j] - keys[i][j]).abs() < 1e-6,
                "Not bit-perfect at [{i}][{j}]: {} vs {}",
                dec_keys[i][j], keys[i][j],
            );
        }
    }
}

#[test]
fn test_markov_checkpoint_spacing() {
    use kv_cache_benchmark::markov_residual::checkpoint::CheckpointConfig;

    let config = CheckpointConfig::gemma_4b();
    // Max recompute should be bounded (max 8-9 layers)
    let max = config.max_recompute();
    assert!(
        max <= 10,
        "Max recompute from checkpoint should be ≤10 layers, got {max}"
    );

    // Should have 5 checkpoints
    assert_eq!(config.layers.len(), 5);
}
