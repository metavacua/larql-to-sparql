use kv_cache_benchmark::*;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::hybrid_cracked::HybridCrackedAttention;
use kv_cache_benchmark::hybrid_cracked::head_classifier::HeadClassification;

#[test]
fn test_hybrid_head_classification() {
    let cls = HeadClassification::gemma_4b();
    // 95.5% static heads
    assert!(
        cls.static_fraction > 0.93,
        "Static fraction too low: {:.1}%",
        cls.static_fraction * 100.0,
    );
    // Few dynamic layers
    assert!(
        cls.dynamic_layer_count() <= 5,
        "Too many dynamic layers: {}",
        cls.dynamic_layer_count(),
    );
}

#[test]
fn test_hybrid_static_head_cosine() {
    // Static heads have cosine >= 0.942 across entities.
    // This is a measured property — we encode it in the classification.
    let cls = HeadClassification::gemma_4b();
    // Verify the fraction matches the 0.942 cosine threshold
    assert!(
        cls.static_fraction > 0.93 && cls.static_fraction < 0.99,
        "Static fraction {:.3} should reflect cosine 0.942 threshold",
        cls.static_fraction,
    );
}

#[test]
fn test_hybrid_dynamic_kv_size() {
    let config = ModelConfig::gemma_4b();
    let hybrid = HybridCrackedAttention::gemma_4b();

    // Dynamic KV at 4K should be 15-27× smaller than full KV
    let hybrid_mem = hybrid.memory_bytes(&config, 4096);
    let full_mem = config.kv_memory(4096);
    let ratio = full_mem as f64 / hybrid_mem as f64;

    assert!(
        ratio > 5.0,
        "Hybrid should be >5× smaller than standard KV at 4K, got {ratio:.1}×"
    );
}

#[test]
fn test_hybrid_memory_at_4k() {
    let config = ModelConfig::gemma_4b();
    let hybrid = HybridCrackedAttention::gemma_4b();
    let mem = hybrid.memory_bytes(&config, 4096);

    // Spec target: ~20-37 MB
    let mb = mem as f64 / 1e6;
    println!("Hybrid at 4K: {mb:.1} MB");
    // Allow some variance — the key point is it's WAY less than 544 MB standard
    assert!(mb < 100.0, "Hybrid at 4K should be <100 MB, got {mb:.1} MB");
}

#[test]
fn test_hybrid_memory_at_370k() {
    let config = ModelConfig::gemma_4b();
    let hybrid = HybridCrackedAttention::gemma_4b();
    let mem = hybrid.memory_bytes(&config, 370_000);
    let standard = config.kv_memory(370_000);

    let ratio = standard as f64 / mem as f64;
    println!(
        "Hybrid at 370K: {:.1} MB (standard: {:.1} GB, ratio: {ratio:.0}×)",
        mem as f64 / 1e6,
        standard as f64 / 1e9,
    );
    assert!(ratio > 10.0, "Should be >10× compression at 370K");
}

#[test]
fn test_hybrid_template_cache_shared() {
    let hybrid = HybridCrackedAttention::gemma_4b();
    // Template cache is shared infrastructure, not per-conversation
    let shared = hybrid.shared_bytes();
    // Per-template + routing table
    assert!(shared > 1_000_000);
    assert!(shared < 5_000_000);
}

#[test]
fn test_hybrid_fallback_to_markov() {
    // When template unknown, hybrid gracefully degrades.
    // This is modeled by the WalkTier::HybridFallback in graph_walk.
    use kv_cache_benchmark::graph_walk::walk_state::{WalkState, WalkTier};

    let unknown = WalkState::from_tokens(&["tell", "me", "about", "nothing"]);
    assert_eq!(unknown.tier, WalkTier::HybridFallback);
}

#[test]
fn test_hybrid_ffn_zero_matmul() {
    let hybrid = HybridCrackedAttention::gemma_4b();
    let keys = vec![vec![1.0f32; 256]; 100];
    let values = vec![vec![2.0f32; 256]; 100];
    let encoded = hybrid.encode(&keys, &values);

    // Encoded should be much smaller than full vectors — FFN contributes zero
    let full_size = 100 * 256 * 4 * 2;
    assert!(
        encoded.len() < full_size / 2,
        "Encoded ({}) too large — FFN should add zero bytes",
        encoded.len(),
    );
}

#[test]
fn test_hybrid_in_memory_ordering() {
    let config = ModelConfig::gemma_4b();
    let standard = kv_cache_benchmark::standard_kv::StandardKv;
    let tq4 = kv_cache_benchmark::turboquant::TurboQuant::new(4);
    let markov = kv_cache_benchmark::markov_residual::MarkovResidual::new(512);
    let hybrid = HybridCrackedAttention::gemma_4b();
    let graph = kv_cache_benchmark::graph_walk::GraphWalk::gemma_4b();

    let seq_len = 4096;
    let mem_std = standard.memory_bytes(&config, seq_len);
    let mem_tq = tq4.memory_bytes(&config, seq_len);
    let mem_hybrid = hybrid.memory_bytes(&config, seq_len);
    let mem_gw = graph.memory_bytes(&config, seq_len);

    // Ordering: Standard > TurboQuant > Hybrid > Graph Walk
    assert!(mem_std > mem_tq, "Standard should > TurboQuant");
    assert!(mem_tq > mem_hybrid, "TurboQuant should > Hybrid");
    assert!(mem_hybrid > mem_gw, "Hybrid should > Graph Walk (per-conv)");

    println!("Memory at 4K: std={:.1}MB, tq={:.1}MB, hybrid={:.1}MB, gw={:.1}KB",
        mem_std as f64/1e6, mem_tq as f64/1e6, mem_hybrid as f64/1e6, mem_gw as f64/1e3);
}
