use kv_cache_benchmark::*;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::graph_walk::GraphWalk;
use kv_cache_benchmark::graph_walk::walk_state::{WalkState, WalkMode, WalkTier};
use kv_cache_benchmark::graph_walk::fallback::TierDistribution;

#[test]
fn test_graph_walk_memory_tiny() {
    let config = ModelConfig::gemma_4b();
    let gw = GraphWalk::gemma_4b();

    // Per-conversation: just token IDs
    let mem_4k = gw.memory_bytes(&config, 4096);
    assert_eq!(mem_4k, 4096 * 4);

    let mem_370k = gw.memory_bytes(&config, 370_000);
    assert_eq!(mem_370k, 370_000 * 4);
    assert!(mem_370k < 2_000_000, "Graph walk per-conversation should be < 2MB");
}

#[test]
fn test_graph_walk_no_kv_stored() {
    let gw = GraphWalk::gemma_4b();
    let keys = vec![vec![1.0f32; 256]; 100];
    let values = vec![vec![2.0f32; 256]; 100];

    let encoded = gw.encode(&keys, &values);
    // Header (4 bytes) + 100 token IDs (400 bytes)
    assert_eq!(encoded.len(), 4 + 100 * 4);
}

#[test]
fn test_graph_walk_france_paris_detection() {
    let state = WalkState::from_tokens(&["What", "is", "the", "capital", "of", "France"]);
    assert_eq!(state.mode, WalkMode::Factual);
    assert_eq!(state.current_relation.as_deref(), Some("capital-of"));
    assert_eq!(state.last_entity.as_deref(), Some("France"));
    assert_eq!(state.tier, WalkTier::CachedTemplate);
}

#[test]
fn test_graph_walk_matches_forward_pass_detection() {
    // Test multiple factual queries are detected correctly
    let queries = vec![
        (vec!["capital", "of", "Germany"], "capital-of", "Germany"),
        (vec!["Mozart", "was", "born", "in"], "birthplace", "Mozart"),
        (vec!["currency", "of", "Japan"], "currency-of", "Japan"),
    ];

    for (tokens, expected_relation, expected_entity) in queries {
        let state = WalkState::from_tokens(&tokens);
        assert_eq!(state.mode, WalkMode::Factual, "Query: {:?}", tokens);
        assert_eq!(
            state.current_relation.as_deref(),
            Some(expected_relation),
            "Query: {:?}",
            tokens
        );
        assert_eq!(
            state.last_entity.as_deref(),
            Some(expected_entity),
            "Query: {:?}",
            tokens
        );
    }
}

#[test]
fn test_graph_walk_routing_table_coverage() {
    let queries = vec![
        vec!["capital", "of", "France"],
        vec!["capital", "of", "Germany"],
        vec!["capital", "of", "Japan"],
        vec!["born", "in", "Mozart"],
        vec!["currency", "of", "USA"],
        vec!["tell", "me", "a", "story"],
        vec!["what", "is", "the", "meaning"],
        vec!["how", "does", "this", "work"],
        vec!["write", "a", "function", "that"],
        vec!["the", "weather", "today"],
    ];

    let states: Vec<WalkState> = queries
        .iter()
        .map(|q| WalkState::from_tokens(&q.iter().map(|s| *s).collect::<Vec<_>>()))
        .collect();

    let dist = TierDistribution::from_states(&states);

    // At least some queries should resolve at Tier A
    assert!(dist.tier_a_count > 0, "No Tier A resolutions");
    // Some should fall back
    assert!(dist.tier_c_count > 0, "No fallback queries");
    // Coverage should be realistic (not 100%, not 0%)
    let coverage = (dist.tier_a_count + dist.tier_b_count) as f64 / dist.total as f64;
    assert!(
        coverage > 0.2 && coverage < 0.9,
        "Coverage {coverage:.2} seems unrealistic"
    );
}

#[test]
fn test_graph_walk_fallback_triggers() {
    // Free-form queries should trigger fallback
    let fallback_queries = vec![
        vec!["tell", "me", "about", "your", "day"],
        vec!["once", "upon", "a", "time"],
        vec!["I", "think", "therefore"],
    ];

    for tokens in &fallback_queries {
        let state = WalkState::from_tokens(&tokens.iter().map(|s| *s).collect::<Vec<_>>());
        assert_eq!(
            state.tier,
            WalkTier::HybridFallback,
            "Expected fallback for: {:?}",
            tokens
        );
    }
}

#[test]
fn test_graph_walk_no_matmul() {
    // Graph Walk should have zero matrix multiplications in the encode path.
    // Verify: encoded data is just token IDs, no vectors.
    let gw = GraphWalk::gemma_4b();
    let keys = vec![vec![99.0f32; 256]; 50];
    let values = vec![vec![99.0f32; 256]; 50];

    let encoded = gw.encode(&keys, &values);

    // Should be header + token IDs only (no float data)
    let expected_size = 4 + 50 * 4; // header + 50 × u32
    assert_eq!(
        encoded.len(),
        expected_size,
        "Encoded size suggests vector data was stored (matmul proxy)"
    );
}

#[test]
fn test_graph_walk_shared_infrastructure_size() {
    let gw = GraphWalk::gemma_4b();
    // Shared: ~1.5 GB vindex + 352 KB routing table
    let shared = gw.shared_bytes();
    assert!(shared > 1_000_000_000, "Shared infra too small: {shared}");
    assert!(shared < 2_000_000_000, "Shared infra too large: {shared}");
}
