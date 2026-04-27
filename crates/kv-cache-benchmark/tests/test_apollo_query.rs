//! End-to-end Apollo query pipeline: real model + real apollo11_store.
//!
//! Exercises routing + entry retrieval + forward-with-injection + greedy
//! top-1 decode. Gated on `real-model` because it needs Gemma 3 4B weights.
//!
//! Run with:
//!   APOLLO_STORE_PATH=/Users/christopherhay/chris-source/apollo-demo/apollo11_store \
//!   cargo test --features real-model -p kv-cache-benchmark \
//!       --test test_apollo_query -- --ignored --nocapture

#![cfg(feature = "real-model")]

use std::path::Path;

use kv_cache_benchmark::apollo::{ApolloEngine, ApolloStore, InjectionConfig};

fn store_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("APOLLO_STORE_PATH") {
        return std::path::PathBuf::from(p);
    }
    for c in [
        "../../../apollo-demo/apollo11_store",
        "../../../../apollo-demo/apollo11_store",
        "/Users/christopherhay/chris-source/apollo-demo/apollo11_store",
    ] {
        let p = Path::new(c);
        if p.join("manifest.json").exists() {
            return p.to_path_buf();
        }
    }
    panic!("apollo11_store not found — set APOLLO_STORE_PATH");
}

fn load_model() -> larql_inference::InferenceModel {
    let model_path = std::env::var("LARQL_MODEL_PATH")
        .unwrap_or_else(|_| "google/gemma-3-4b-it".to_string());
    larql_inference::InferenceModel::load(&model_path).expect("load gemma")
}

#[test]
#[ignore]
fn test_routing_resolves_porridge_to_w170_region() {
    // No model needed — tests routing alone on the real store.
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    // Tokenize with the model's tokenizer so IDs line up with the store.
    let model = load_model();
    let tok = model.tokenizer();

    for query in [
        "porridge eating contest",
        "Corby England",
        "John Coyle",
    ] {
        let enc = tok.encode(query, false).expect("tokenize");
        let qids: Vec<u32> = enc.get_ids().to_vec();
        let q = kv_cache_benchmark::apollo::RoutingQuery { token_ids: qids };
        let res = engine.routing().resolve(&q, 5);
        println!("  query {query:?} → top-5 windows: {:?}", res);
        assert!(!res.is_empty(), "routing returned no windows for {query}");
    }
}

#[test]
#[ignore]
fn test_retrieve_entries_for_query() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model = load_model();
    let tok = model.tokenizer();

    let query = "porridge eating contest Corby";
    let enc = tok.encode(query, false).expect("tokenize");
    let qids: Vec<u32> = enc.get_ids().to_vec();

    // Route first
    let q = kv_cache_benchmark::apollo::RoutingQuery {
        token_ids: qids.clone(),
    };
    let windows = engine.routing().resolve(&q, 5);
    assert!(!windows.is_empty());

    // Retrieve entries scoped to routed windows
    let entries = engine
        .retrieve_entries(&qids, &windows)
        .expect("retrieve");
    println!("  retrieved {} entries", entries.len());
    for e in entries.iter().take(10) {
        let txt = tok.decode(&[e.token_id], false).unwrap_or_default();
        println!(
            "    token {} ({txt:?}) coef={:.2} window={} fact_id={}",
            e.token_id, e.coefficient, e.window_id, e.fact_id,
        );
    }

    // At minimum the retrieve step shouldn't crash; any non-empty set of
    // entries validates the pipeline.
    // (For this specific query we expect entries since Apollo's store
    // should have porridge/Coyle/Corby fact tokens.)
    assert!(
        entries.len() <= engine.config().top_k,
        "retrieve returned more than top_k",
    );
}

#[test]
#[ignore]
fn test_end_to_end_query_produces_nonempty_answer() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model = load_model();
    let weights = model.weights();
    let tok = model.tokenizer();

    let query = "Who won the porridge eating contest?";
    let trace = engine
        .query_greedy(weights, tok, query, 3)
        .expect("query_greedy");

    println!("\n=== Apollo query trace ===");
    println!("  query: {query:?}");
    println!("  routed windows: {:?}", trace.routed_windows);
    println!("  injected entries ({}):", trace.injected_entries.len());
    for e in &trace.injected_entries {
        let txt = tok.decode(&[e.token_id], false).unwrap_or_default();
        println!(
            "    token {} ({txt:?}) coef={:.2} window={}",
            e.token_id, e.coefficient, e.window_id,
        );
    }
    println!("  context tokens: {}", trace.context_tokens);
    let top1_txt = tok.decode(&[trace.top1_token_id], false).unwrap_or_default();
    println!(
        "  top-1 prediction: token {} ({top1_txt:?}) logit={:.3}",
        trace.top1_token_id, trace.top1_logit,
    );

    // Sanity: the pipeline produced a valid token ID and non-empty state.
    assert!(!trace.routed_windows.is_empty());
    assert!(trace.context_tokens > 0);
    assert!(trace.top1_logit.is_finite());
}

/// Compressed-path variant: forwards the 10 KB boundary + query tokens
/// only, NOT the full window tokens. This is the path that actually
/// exercises Apollo's ~20,000× compression claim at inference time.
///
/// Expect weaker answer quality than the uncompressed path (single-vector
/// boundary is lossy vs joint forward) — this test asserts the pipeline
/// runs end-to-end and produces finite logits, not specific correctness.
#[test]
#[ignore]
fn test_end_to_end_query_compressed_path() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model = load_model();
    let weights = model.weights();
    let tok = model.tokenizer();

    let query = "Who won the porridge eating contest?";
    let trace = engine
        .query_greedy_compressed(weights, tok, query, 3)
        .expect("query_greedy_compressed");

    println!("\n=== Apollo query trace (COMPRESSED path) ===");
    println!("  query: {query:?}");
    println!(
        "  context tokens: {} (= query_tokens + 1 virtual boundary)",
        trace.context_tokens,
    );
    println!(
        "  boundary size: {} B (one f32 vector per window)",
        weights.hidden_size * 4,
    );
    println!("  routed windows: {:?}", trace.routed_windows);
    println!("  injected entries ({}):", trace.injected_entries.len());
    for e in &trace.injected_entries {
        let txt = tok.decode(&[e.token_id], false).unwrap_or_default();
        println!(
            "    token {} ({txt:?}) coef={:.2} window={}",
            e.token_id, e.coefficient, e.window_id,
        );
    }
    let top1_txt = tok.decode(&[trace.top1_token_id], false).unwrap_or_default();
    println!(
        "  top-1 prediction: token {} ({top1_txt:?}) logit={:.3}",
        trace.top1_token_id, trace.top1_logit,
    );

    // In the compressed path the "context" is tiny — query length + 1.
    // Verify we're actually using the compressed path, not accidentally
    // falling back to window tokens.
    assert!(
        trace.context_tokens <= 32,
        "compressed path should use ≤ 32 tokens, got {}",
        trace.context_tokens,
    );
    assert!(trace.top1_logit.is_finite());
}

/// Iterative decode under the compressed path — generates sentence-length
/// output by looping the forward pass with the same boundary + injection
/// applied at every step. This is the output format Video 2 demonstrates.
#[test]
#[ignore]
fn test_apollo_generate_compressed() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model = load_model();
    let weights = model.weights();
    let tok = model.tokenizer();

    let query = "Who won the porridge eating contest?";
    let trace = engine
        .query_generate_compressed(weights, tok, query, 25, 3)
        .expect("generate");

    let generated_text = tok
        .decode(&trace.generated_token_ids, false)
        .unwrap_or_default();

    println!("\n=== Apollo iterative decode (COMPRESSED path) ===");
    println!("  query:  {query:?}");
    println!(
        "  routed windows: {:?}",
        trace.routed_windows
    );
    println!(
        "  initial context: {} tokens (boundary + query)",
        trace.initial_context_tokens,
    );
    println!(
        "  injected entries ({}):",
        trace.injected_entries.len()
    );
    for e in &trace.injected_entries {
        let txt = tok.decode(&[e.token_id], false).unwrap_or_default();
        println!(
            "    token {} ({txt:?}) coef={:.2}",
            e.token_id, e.coefficient,
        );
    }
    println!("  generated ({} tokens, stopped_on_eos={}):", trace.generated_token_ids.len(), trace.stopped_on_eos);
    println!("    {generated_text:?}");
    print!("  per-step logits:");
    for v in &trace.per_step_logits {
        print!(" {:.1}", v);
    }
    println!();

    assert!(!trace.generated_token_ids.is_empty(), "no tokens generated");
    assert!(trace.per_step_logits.iter().all(|v| v.is_finite()));
}

/// Side-by-side iterative decode on both paths for the porridge query.
/// Uncompressed forwards the full window tokens (~520), compressed forwards
/// just the 10 KB boundary + query (~9). Same injection + decode loop.
#[test]
#[ignore]
fn test_apollo_generate_side_by_side() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model = load_model();
    let weights = model.weights();
    let tok = model.tokenizer();

    let query = "Who won the porridge eating contest?";
    // Set via env or default; uncompressed is O(n^2 per step) so we keep
    // this short unless SIDE_BY_SIDE_TOKENS overrides it.
    let max_tokens: usize = std::env::var("SIDE_BY_SIDE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    println!("\n=== Apollo generation — both paths, same query ===");
    println!("  query: {query:?}");

    for (label, trace) in [
        (
            "compressed",
            engine
                .query_generate_compressed(weights, tok, query, max_tokens, 3)
                .expect("gen compressed"),
        ),
        (
            "uncompressed",
            engine
                .query_generate_uncompressed(weights, tok, query, max_tokens, 3)
                .expect("gen uncompressed"),
        ),
    ] {
        let text = tok
            .decode(&trace.generated_token_ids, false)
            .unwrap_or_default();
        println!(
            "\n  [{label}] initial_ctx={} tokens, generated={}, stopped_on_eos={}",
            trace.initial_context_tokens,
            trace.generated_token_ids.len(),
            trace.stopped_on_eos,
        );
        println!("    {text:?}");
        print!("    logits:");
        for v in &trace.per_step_logits {
            print!(" {:.1}", v);
        }
        println!();
    }
}
