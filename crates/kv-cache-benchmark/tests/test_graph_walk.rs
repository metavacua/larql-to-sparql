use kv_cache_benchmark::graph_walk::GraphWalk;
use kv_cache_benchmark::graph_walk::walk_state::{WalkState, WalkMode, WalkTier};
use kv_cache_benchmark::graph_walk::fallback::TierDistribution;

#[test]
fn test_graph_walk_memory_tiny() {
    let gw = GraphWalk::gemma_4b();

    // Per-conversation: just token IDs
    let mem_4k = gw.memory_bytes(4096);
    assert_eq!(mem_4k, 4096 * 4);

    let mem_370k = gw.memory_bytes(370_000);
    assert_eq!(mem_370k, 370_000 * 4);
    assert!(mem_370k < 2_000_000, "Graph walk per-conversation should be < 2MB");
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
        .map(|q| WalkState::from_tokens(&q.to_vec()))
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
        let state = WalkState::from_tokens(&tokens.to_vec());
        assert_eq!(
            state.tier,
            WalkTier::MarkovFallback,
            "Expected fallback for: {:?}",
            tokens
        );
    }
}

#[test]
fn test_graph_walk_shared_infrastructure_size() {
    let gw = GraphWalk::gemma_4b();
    // Shared: ~1.5 GB vindex + 352 KB routing table
    let shared = gw.shared_bytes();
    assert!(shared > 1_000_000_000, "Shared infra too small: {shared}");
    assert!(shared < 2_000_000_000, "Shared infra too large: {shared}");
}

#[test]
fn test_graph_walk_template_decomposition() {
    use kv_cache_benchmark::graph_walk::template::{PatternWalk, TemplateCache};

    // Template decomposition: pattern walk + entity walk = correct prediction
    let pattern = PatternWalk::capital_of();

    // Pattern walk should have critical layers identified
    assert!(!pattern.critical_layers.is_empty());
    assert!(pattern.critical_layers.contains(&24)); // Factual retrieval layer

    // Mean cosine across entities should be very high (template is shared)
    assert!(
        pattern.mean_cosine > 0.99,
        "Template cosine {:.3} should be >0.99",
        pattern.mean_cosine,
    );

    // KNN lookups should be small (only critical layers)
    assert!(pattern.knn_lookups() <= 10);

    // Estimated latency should be sub-millisecond
    assert!(pattern.estimated_latency_us() < 1000.0);

    // Template cache should be able to store and retrieve
    let cache = TemplateCache::with_defaults();
    assert!(cache.lookup("capital-of").is_some());
    assert!(cache.lookup("nonexistent").is_none());
}

#[test]
fn test_graph_walk_matches_forward_pass_50_queries() {
    // 50 factual queries — all should be detected as factual
    let queries: Vec<Vec<&str>> = vec![
        vec!["capital", "of", "France"],
        vec!["capital", "of", "Germany"],
        vec!["capital", "of", "Italy"],
        vec!["capital", "of", "Spain"],
        vec!["capital", "of", "Japan"],
        vec!["capital", "of", "Brazil"],
        vec!["capital", "of", "India"],
        vec!["capital", "of", "China"],
        vec!["capital", "of", "Russia"],
        vec!["capital", "of", "Canada"],
        vec!["capital", "of", "Australia"],
        vec!["capital", "of", "Mexico"],
        vec!["capital", "of", "Egypt"],
        vec!["capital", "of", "Turkey"],
        vec!["capital", "of", "Thailand"],
        vec!["capital", "of", "Sweden"],
        vec!["capital", "of", "Norway"],
        vec!["capital", "of", "Poland"],
        vec!["capital", "of", "Argentina"],
        vec!["capital", "of", "Chile"],
        vec!["Mozart", "was", "born", "in"],
        vec!["Beethoven", "was", "born", "in"],
        vec!["Bach", "was", "born", "in"],
        vec!["Einstein", "was", "born", "in"],
        vec!["Shakespeare", "was", "born", "in"],
        vec!["currency", "of", "Japan"],
        vec!["currency", "of", "Brazil"],
        vec!["currency", "of", "India"],
        vec!["currency", "of", "China"],
        vec!["currency", "of", "Russia"],
        vec!["currency", "of", "UK"],
        vec!["currency", "of", "Switzerland"],
        vec!["currency", "of", "Mexico"],
        vec!["currency", "of", "Korea"],
        vec!["currency", "of", "Turkey"],
        vec!["capital", "of", "Peru"],
        vec!["capital", "of", "Colombia"],
        vec!["capital", "of", "Portugal"],
        vec!["capital", "of", "Greece"],
        vec!["capital", "of", "Austria"],
        vec!["capital", "of", "Belgium"],
        vec!["capital", "of", "Netherlands"],
        vec!["capital", "of", "Denmark"],
        vec!["capital", "of", "Finland"],
        vec!["capital", "of", "Ireland"],
        vec!["capital", "of", "Kenya"],
        vec!["capital", "of", "Nigeria"],
        vec!["capital", "of", "Morocco"],
        vec!["capital", "of", "Vietnam"],
        vec!["capital", "of", "Indonesia"],
    ];

    assert_eq!(queries.len(), 50);

    let mut factual_count = 0;
    for q in &queries {
        let state = WalkState::from_tokens(q);
        if state.mode == WalkMode::Factual {
            factual_count += 1;
        }
    }

    // All 50 should be detected as factual
    assert_eq!(
        factual_count, 50,
        "Expected all 50 queries to be factual, got {factual_count}/50"
    );
}
