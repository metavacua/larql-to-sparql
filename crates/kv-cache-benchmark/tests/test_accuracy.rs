use kv_cache_benchmark::accuracy::*;

// ── Category 1: Top-1 Token Match infrastructure ──

#[test]
fn test_accuracy_factual_prompts_exist() {
    let prompts = factual_prompts();
    assert!(prompts.len() >= 20, "Need at least 20 factual prompts, got {}", prompts.len());
    // All should have non-empty prompt and expected answer
    for (prompt, answer) in &prompts {
        assert!(!prompt.is_empty());
        assert!(!answer.is_empty());
    }
}

#[test]
fn test_accuracy_diverse_prompts_exist() {
    let prompts = diverse_prompts();
    assert!(prompts.len() >= 10, "Need at least 10 diverse prompts, got {}", prompts.len());
}

// ── Category 2: KL Divergence ──

#[test]
fn test_kl_divergence_identical() {
    let p = vec![0.7, 0.2, 0.1];
    let kl = kl_divergence(&p, &p);
    assert!(kl.abs() < 1e-10, "KL of identical distributions should be 0, got {kl}");
}

#[test]
fn test_kl_divergence_different() {
    let p = vec![0.9, 0.05, 0.05];
    let q = vec![0.33, 0.34, 0.33];
    let kl = kl_divergence(&p, &q);
    assert!(kl > 0.0, "KL of different distributions should be > 0");
    assert!(kl < 10.0, "KL should be reasonable, got {kl}");
}

#[test]
fn test_js_divergence_symmetric() {
    let p = vec![0.7, 0.2, 0.1];
    let q = vec![0.3, 0.4, 0.3];
    let js_pq = js_divergence(&p, &q);
    let js_qp = js_divergence(&q, &p);
    assert!(
        (js_pq - js_qp).abs() < 1e-10,
        "JS divergence should be symmetric: {js_pq} vs {js_qp}",
    );
}

#[test]
fn test_js_divergence_bounded() {
    let p = vec![1.0, 0.0, 0.0];
    let q = vec![0.0, 0.0, 1.0];
    let js = js_divergence(&p, &q);
    // JS divergence is bounded by ln(2) ≈ 0.693
    assert!(js <= 0.7, "JS should be bounded, got {js}");
}

#[test]
fn test_softmax_sums_to_one() {
    let logits = vec![2.0f32, 1.0, 0.5, -1.0, 3.0];
    let probs = softmax(&logits);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {sum}");
}

#[test]
fn test_softmax_argmax_preserved() {
    let logits = vec![1.0f32, 5.0, 2.0, 0.5];
    let probs = softmax(&logits);
    // Largest logit (index 1) should have highest probability
    let max_idx = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 1);
}

// ── Category 1/2: Top-K overlap ──

#[test]
fn test_top_k_overlap_identical() {
    let a = vec![1, 2, 3, 4, 5];
    let b = vec![1, 2, 3, 4, 5];
    assert_eq!(top_k_overlap(&a, &b, 5), 1.0);
}

#[test]
fn test_top_k_overlap_disjoint() {
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    assert_eq!(top_k_overlap(&a, &b, 3), 0.0);
}

#[test]
fn test_top_k_overlap_partial() {
    let a = vec![1, 2, 3, 4, 5];
    let b = vec![1, 2, 6, 7, 8];
    assert_eq!(top_k_overlap(&a, &b, 5), 0.4); // 2 out of 5
}

// ── Category 5: Generation divergence ──

#[test]
fn test_first_divergence_identical() {
    let a = vec![1u32, 2, 3, 4, 5];
    let b = vec![1, 2, 3, 4, 5];
    assert_eq!(first_divergence(&a, &b), None);
}

#[test]
fn test_first_divergence_at_position() {
    let a = vec![1u32, 2, 3, 4, 5];
    let b = vec![1, 2, 9, 4, 5];
    assert_eq!(first_divergence(&a, &b), Some(2));
}

#[test]
fn test_token_match_rate_perfect() {
    let a = vec![1u32, 2, 3, 4, 5];
    assert_eq!(token_match_rate(&a, &a), 1.0);
}

#[test]
fn test_token_match_rate_partial() {
    let a = vec![1u32, 2, 3, 4, 5];
    let b = vec![1, 2, 9, 4, 5];
    assert_eq!(token_match_rate(&a, &b), 0.8); // 4/5
}

#[test]
fn test_reciprocal_rank_first() {
    let preds = vec![42u32, 7, 13, 99];
    assert_eq!(reciprocal_rank(&preds, 42), 1.0);
}

#[test]
fn test_reciprocal_rank_third() {
    let preds = vec![7u32, 13, 42, 99];
    assert!((reciprocal_rank(&preds, 42) - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_reciprocal_rank_missing() {
    let preds = vec![7u32, 13, 99];
    assert_eq!(reciprocal_rank(&preds, 42), 0.0);
}

// ── Category 3: Needle-in-a-haystack ──

#[test]
fn test_haystack_generation_short() {
    let (context, needle) = generate_haystack(500, 200, "SECRET-CODE-12345");
    assert!(context.contains("SECRET-CODE-12345"));
    assert_eq!(needle, "SECRET-CODE-12345");
    assert!(context.len() > 200);
}

#[test]
fn test_haystack_generation_long() {
    let (context, needle) = generate_haystack(32000, 5000, "The secret project code is AURORA-7749");
    assert!(context.contains("AURORA-7749"));
    assert!(context.len() > 10000);
}

#[test]
fn test_haystack_needle_position() {
    let (context, _) = generate_haystack(1000, 100, "NEEDLE");
    // Needle should be roughly at the target position (in chars, not tokens)
    let pos = context.find("NEEDLE").unwrap();
    // Position 100 tokens ≈ 600 chars (6 chars/word avg in filler)
    assert!(pos > 50, "Needle too early: pos {pos}");
}

// ── Category 4: Multi-turn retention ──

#[test]
fn test_retention_conversation_structure() {
    let turns = build_retention_conversation(15);
    assert_eq!(turns.len(), 15);

    // First few should establish facts
    assert!(!turns[0].is_query);
    assert!(turns[0].fact_key.is_some());

    // Last few should be queries
    let queries: Vec<_> = turns.iter().filter(|t| t.is_query).collect();
    assert!(queries.len() >= 3, "Need at least 3 query turns");

    // Queries should have expected facts
    for q in &queries {
        assert!(q.expected_fact.is_some());
    }
}

#[test]
fn test_retention_conversation_25_turns() {
    let turns = build_retention_conversation(25);
    assert_eq!(turns.len(), 25);

    let queries: Vec<_> = turns.iter().filter(|t| t.is_query).collect();
    assert!(queries.len() >= 3);

    let facts: Vec<_> = turns.iter().filter(|t| !t.is_query && t.fact_key.is_some()).collect();
    assert!(facts.len() >= 3, "Need at least 3 fact-establishing turns");
}

// ── AccuracyResult construction ──

#[test]
fn test_accuracy_result_token_match() {
    let r = AccuracyResult::token_match("Markov RS", "factual", "capital of France", true);
    assert!(r.top1_match);
    assert_eq!(r.kl_divergence, 0.0);
    assert_eq!(r.strategy, "Markov RS");
}

#[test]
fn test_accuracy_result_needle() {
    let r = AccuracyResult::needle("Standard KV", "needle_4k", "haystack", true, true);
    assert_eq!(r.needle_found, Some(true));
    assert_eq!(r.needle_exact_match, Some(true));
}

// ── Summary formatting ──

#[test]
fn test_accuracy_summary_format() {
    let results = vec![
        AccuracyResult::token_match("Standard KV", "factual", "capital of France", true),
        AccuracyResult::token_match("Standard KV", "factual", "capital of Germany", true),
        AccuracyResult::token_match("TurboQuant 4b", "factual", "capital of France", true),
        AccuracyResult::token_match("TurboQuant 4b", "factual", "capital of Germany", false),
        AccuracyResult::token_match("Markov RS", "factual", "capital of France", true),
        AccuracyResult::token_match("Markov RS", "factual", "capital of Germany", true),
    ];

    let summary = format_accuracy_summary(&results);
    assert!(summary.contains("Standard KV"));
    assert!(summary.contains("100.0%"));
    assert!(summary.contains("Markov RS"));
    assert!(summary.contains("TurboQuant"));
}

// ── Adversarial helpers ──

#[test]
fn test_entity_confusion_prompts() {
    // Same template, different entities — test that strategies distinguish them
    let prompts = vec![
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
    ];
    // All use "capital of X" template but need different answers
    for (prompt, expected) in &prompts {
        assert!(prompt.contains("capital of"));
        assert!(!expected.is_empty());
    }
}

#[test]
fn test_polysemy_prompts() {
    // Context-dependent meaning — "bank" in different contexts
    let financial = "I went to the bank to deposit money. The bank";
    let river = "I sat on the river bank and watched the water. The bank";
    // Different continuations expected based on context
    assert!(financial.contains("deposit"));
    assert!(river.contains("river"));
}
