//! Apollo accuracy sweep across multiple queries + both forward paths.
//!
//! Loads model + store ONCE, runs N queries through both `query_greedy`
//! (uncompressed: window tokens + query) and `query_greedy_compressed`
//! (compressed: 10 KB boundary + query). Prints a comparison table.
//!
//! Run with:
//!   cargo test --features real-model -p kv-cache-benchmark \
//!       --test test_apollo_accuracy -- --ignored --nocapture

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

/// Queries from the Video 2 script (+ a couple extras) and the token
/// substring we'd expect the top-1 token to start with or contain as a
/// rough correctness signal. The goal here is qualitative — record what
/// each path predicts, not enforce a specific answer.
const QUERIES: &[&str] = &[
    "Who won the porridge eating contest?",
    "How many bowls of oatmeal?",
    "What city was Corby in?",
    "What did Neil Armstrong say about the TV camera?",
    "What was the weather in Minneapolis?",
    "What did Le Figaro say?",
];

#[test]
#[ignore]
fn test_apollo_accuracy_sweep() {
    let store = ApolloStore::load(&store_path()).expect("load store");
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("build routing");

    let model_path = std::env::var("LARQL_MODEL_PATH")
        .unwrap_or_else(|_| "google/gemma-3-4b-it".to_string());
    let model = larql_inference::InferenceModel::load(&model_path).expect("load model");
    let weights = model.weights();
    let tok = model.tokenizer();

    println!("\n{}", "=".repeat(100));
    println!("Apollo accuracy sweep — {} queries × 2 paths", QUERIES.len());
    println!("{}", "=".repeat(100));

    println!(
        "\n{:<48}  {:<20}  {:<20}  {}",
        "query", "uncompressed top-1", "compressed top-1", "ctx ratio"
    );
    println!("{}", "-".repeat(110));

    for q in QUERIES {
        let uncomp = engine.query_greedy(weights, tok, q, 3);
        let comp = engine.query_greedy_compressed(weights, tok, q, 3);

        let fmt_top = |r: &Result<_, _>| -> (String, usize, f32) {
            match r {
                Ok(t) => {
                    let t: &kv_cache_benchmark::apollo::QueryTrace = t;
                    let txt = tok
                        .decode(&[t.top1_token_id], false)
                        .unwrap_or_default();
                    (
                        format!("{:?} @ {:.1}", txt, t.top1_logit),
                        t.context_tokens,
                        t.top1_logit,
                    )
                }
                Err(e) => (format!("ERR: {e}"), 0, 0.0),
            }
        };

        let (u_fmt, u_ctx, _) = fmt_top(&uncomp);
        let (c_fmt, c_ctx, _) = fmt_top(&comp);
        let ratio = if c_ctx > 0 {
            format!("{:.1}×", u_ctx as f64 / c_ctx as f64)
        } else {
            "—".into()
        };

        let truncq: String = q.chars().take(46).collect();
        println!(
            "{:<48}  {:<20}  {:<20}  {}",
            truncq, u_fmt, c_fmt, ratio
        );
    }
    println!();
}
