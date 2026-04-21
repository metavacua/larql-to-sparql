//! Embed server demo — shows what the embed endpoints return with synthetic data.
//!
//! Simulates the three embed-service operations:
//!   1. `POST /v1/embed`   — token_ids → scaled residual vectors
//!   2. `POST /v1/logits`  — final residual → top-k token probabilities
//!   3. `GET /v1/token/*`  — tokenizer encode / decode
//!
//! No real model needed. Run:
//!   cargo run -p larql-server --example embed_demo

use larql_vindex::ndarray::Array2;

fn section(title: &str) {
    println!("\n══ {} ══", title);
}

// ── Synthetic data ────────────────────────────────────────────────────────────

/// Tiny vocab / embedding table for demo purposes.
/// 8 tokens, hidden_size = 4.  Each token activates a different direction.
fn demo_embeddings() -> (Array2<f32>, f32) {
    let vocab = 8;
    let hidden = 4;
    let scale = 1.0f32; // Gemma uses sqrt(hidden_size); kept simple here

    let mut embed = Array2::<f32>::zeros((vocab, hidden));
    // token 0 → [1,0,0,0],  1 → [0,1,0,0], …
    embed[[0, 0]] = 1.0;
    embed[[1, 1]] = 1.0;
    embed[[2, 2]] = 1.0;
    embed[[3, 3]] = 1.0;
    // blended tokens (simulate subword pieces)
    embed[[4, 0]] = 0.7; embed[[4, 1]] = 0.7;
    embed[[5, 1]] = 0.6; embed[[5, 2]] = 0.8;
    embed[[6, 2]] = 0.5; embed[[6, 3]] = 0.5; embed[[6, 0]] = 0.5;
    embed[[7, 3]] = 1.0;

    (embed, scale)
}

/// Pretend "token vocabulary" for decode output.
fn token_name(id: u32) -> &'static str {
    match id {
        0 => "▁The",
        1 => "▁capital",
        2 => "▁of",
        3 => "▁France",
        4 => "▁is",
        5 => "▁Paris",
        6 => "▁Berlin",
        7 => "▁London",
        _ => "<unk>",
    }
}

// ── Embed endpoint simulation ─────────────────────────────────────────────────

fn demo_embed(embed: &Array2<f32>, scale: f32, token_ids: &[u32]) {
    let hidden = embed.shape()[1];
    println!("Request:  {{ \"token_ids\": {:?} }}", token_ids);
    let start = std::time::Instant::now();

    let residual: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| {
            let row = embed.row(id as usize);
            row.iter().map(|&v| v * scale).collect()
        })
        .collect();

    let ms = start.elapsed().as_secs_f32() * 1000.0;

    println!("Response: {{");
    println!("  \"seq_len\": {},", token_ids.len());
    println!("  \"hidden_size\": {},", hidden);
    for (i, row) in residual.iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|v| format!("{:.2}", v)).collect();
        println!("  \"residual[{}]\": [{}],", i, formatted.join(", "));
    }
    println!("  \"latency_ms\": {:.4}", ms);
    println!("}}");
}

// ── Logits endpoint simulation ────────────────────────────────────────────────

/// Simulate lm_head by projecting the residual against the embedding table
/// (tied weights — exact pattern used by Gemma 3/4).
fn demo_logits(embed: &Array2<f32>, residual: &[f32], top_k: usize) {
    let vocab = embed.shape()[0];
    println!("Request:  {{ \"residual\": [{}...], \"top_k\": {} }}",
        residual.iter().take(4).map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join(", "),
        top_k);
    let start = std::time::Instant::now();

    // Compute scores = embed @ residual (one dot product per token)
    let mut scores: Vec<(u32, f32)> = (0..vocab)
        .map(|id| {
            let row = embed.row(id);
            let score: f32 = row.iter().zip(residual).map(|(&e, &r)| e * r).sum();
            (id as u32, score)
        })
        .collect();

    // Softmax
    let max_score = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();

    // Update with probs, sort descending
    for (i, (_, score)) in scores.iter_mut().enumerate() {
        *score = probs[i];
    }
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);

    let ms = start.elapsed().as_secs_f32() * 1000.0;

    println!("Response: {{");
    println!("  \"top_k\": [");
    for (token_id, prob) in &scores {
        println!("    {{ \"token_id\": {}, \"token\": {:?}, \"prob\": {:.4} }},",
            token_id, token_name(*token_id), prob);
    }
    println!("  ],");
    println!("  \"latency_ms\": {:.4}", ms);
    println!("}}");
}

// ── Token encode / decode simulation ─────────────────────────────────────────

fn demo_token_encode(text: &str) {
    // Simple lookup: split on spaces, match against our tiny vocab.
    let mapping = [
        ("The", 0u32), ("capital", 1), ("of", 2), ("France", 3),
        ("is", 4), ("Paris", 5), ("Berlin", 6), ("London", 7),
    ];
    let ids: Vec<u32> = text.split_whitespace()
        .filter_map(|w| mapping.iter().find(|(k, _)| *k == w).map(|(_, id)| *id))
        .collect();

    println!("GET /v1/token/encode?text={:?}", text);
    println!("Response: {{ \"token_ids\": {:?}, \"text\": {:?} }}", ids, text);
}

fn demo_token_decode(ids: &[u32]) {
    let text: Vec<&str> = ids.iter().map(|&id| token_name(id)).collect();
    let decoded = text.join(" ");
    println!("GET /v1/token/decode?ids={}", ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","));
    println!("Response: {{ \"text\": {:?}, \"token_ids\": {:?} }}", decoded, ids);
}

// ── Binary wire format demonstration ─────────────────────────────────────────

fn demo_binary_wire() {
    section("Binary Wire Format (application/x-larql-ffn)");

    // Embed request: [num_tokens u32][token_id u32 × N]
    let token_ids = [3u32, 4, 5]; // France, is, Paris
    let mut embed_req = Vec::new();
    embed_req.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &id in &token_ids {
        embed_req.extend_from_slice(&id.to_le_bytes());
    }
    println!("Embed request  ({} bytes): {:?}", embed_req.len(), &embed_req[..embed_req.len().min(16)]);

    // Embed response: [seq_len u32][hidden_size u32][floats]
    let seq_len = 3u32;
    let hidden = 4u32;
    let mut embed_resp = Vec::new();
    embed_resp.extend_from_slice(&seq_len.to_le_bytes());
    embed_resp.extend_from_slice(&hidden.to_le_bytes());
    for _ in 0..seq_len * hidden {
        embed_resp.extend_from_slice(&0.5f32.to_le_bytes());
    }
    println!("Embed response ({} bytes): seq_len={seq_len}, hidden={hidden}, payload={} bytes",
        embed_resp.len(), seq_len * hidden * 4);

    // Logits request: raw [f32 × hidden_size]
    let residual = [0.1f32, 0.2, 0.3, 0.4];
    let logits_req: Vec<u8> = residual.iter().flat_map(|v| v.to_le_bytes()).collect();
    println!("Logits request  ({} bytes): {:?}", logits_req.len(), &residual);
}

// ── Stats response ────────────────────────────────────────────────────────────

fn demo_stats() {
    section("GET /v1/stats (embed-service mode)");
    let stats = serde_json::json!({
        "model": "google/gemma-3-4b-it",
        "family": "gemma3",
        "mode": "embed-service",
        "layers": 34,
        "hidden_size": 2560,
        "vocab_size": 262208,
        "loaded": {
            "browse": false,
            "inference": false,
            "ffn_service": false,
            "embed_service": true,
        }
    });
    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("LARQL Embed Server Demo");
    println!("═══════════════════════");
    println!("Simulates the embed-service endpoints with synthetic data.");
    println!("In production: larql-server <vindex> --embed-only --port 8082");

    let (embed, scale) = demo_embeddings();
    println!("\nEmbeddings: {}×{} matrix, scale={}", embed.shape()[0], embed.shape()[1], scale);

    // ── POST /v1/embed ────────────────────────────────────────────────────
    section("POST /v1/embed — single token (decode step)");
    demo_embed(&embed, scale, &[5]); // "Paris"

    section("POST /v1/embed — full prompt (prefill)");
    demo_embed(&embed, scale, &[0, 1, 2, 3, 4]); // "The capital of France is"

    // ── POST /v1/logits ───────────────────────────────────────────────────
    section("POST /v1/logits — residual → top-5 tokens");
    // Residual that points toward token 5 ("Paris") — dim 1 high, dim 2 moderate
    let residual = [0.1f32, 0.9, 0.6, 0.1];
    demo_logits(&embed, &residual, 5);

    section("POST /v1/logits — residual pointing at Berlin");
    let residual = [0.1f32, 0.5, 0.9, 0.0];
    demo_logits(&embed, &residual, 3);

    // ── GET /v1/token/encode ──────────────────────────────────────────────
    section("GET /v1/token/encode");
    demo_token_encode("The capital of France is");

    // ── GET /v1/token/decode ──────────────────────────────────────────────
    section("GET /v1/token/decode");
    demo_token_decode(&[0, 1, 2, 3, 4, 5]);

    // ── Binary wire format ────────────────────────────────────────────────
    demo_binary_wire();

    // ── Stats ─────────────────────────────────────────────────────────────
    demo_stats();

    println!("\n══ Summary ══");
    println!("  Embed lookup:    O(1) table access — one row per token_id");
    println!("  Logits:          O(vocab × hidden) matmul — ~2ms CPU / ~0.1ms Metal");
    println!("  Token encode:    tokenizer lookup — microseconds");
    println!("  Token decode:    tokenizer lookup — microseconds");
    println!("\n  Deploy:  larql-server <vindex> --embed-only --port 8082");
    println!("  Client:  POST http://embed-server:8082/v1/embed");
    println!("           POST http://embed-server:8082/v1/logits");
}
