//! Bounded-state decode comparison: Full-KV vs RS-decode.
//!
//! This is the actual experiment the email asks for: run `rs_decode_step`
//! (which reconstructs K/V from stored residuals) against a full-KV decode
//! step on the same token, and measure whether predictions match as context
//! grows and the window boundary becomes load-bearing.
//!
//! Two query types:
//!   Parametric  — answer lives in model weights (factual recall).
//!                 Window barely matters: the entity info is encoded in the
//!                 residual stream from training, not from the context.
//!   InContext   — answer is planted in the prompt context (in-context lookup).
//!                 When the window excludes the planted fact, RS decode must
//!                 fail — there is no route to the answer.
//!
//! The distinction maps directly to the spec's dual retrieval circuits:
//!   L1/L32  → parametric routing (static for in-context queries)
//!   L29/L30 → in-context comprehension (dynamic for in-context, static for parametric)

use ndarray::Array2;
use larql_inference::model::ModelWeights;
use larql_inference::attention::run_attention_block_decode_step;
use larql_inference::forward::{embed_tokens_pub, run_ffn, logits_to_predictions_pub};
use larql_inference::ffn::WeightFfn;

use super::kv_capture::capture_kv;
use super::markov_layer::{rs_prefill, rs_decode_step};

/// Whether the answer is in the model's weights or planted in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum QueryType {
    /// Answer from training (factual recall). Window should not matter.
    Parametric,
    /// Answer planted in context. Fails when window excludes the fact.
    InContext,
}

/// Result of one decode step comparing full-KV vs RS.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecodeStep {
    pub step: usize,
    pub full_kv_token: String,
    pub rs_token: String,
    pub top1_match: bool,
    pub hidden_cosine: f64,
    pub full_kv_prob: f64,
    pub rs_prob: f64,
}

/// Full result of the decode comparison for one prompt + window size.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecodeComparisonResult {
    pub prompt: String,
    pub query_type: QueryType,
    pub window_size: usize,
    pub prompt_tokens: usize,
    pub steps: Vec<DecodeStep>,
    /// Step index of first divergence, if any.
    pub first_divergence: Option<usize>,
    pub match_rate: f64,
}

impl DecodeComparisonResult {
    pub fn verdict(&self) -> &'static str {
        match self.first_divergence {
            None => "MATCH",
            Some(_) => "DIVERGE",
        }
    }
}

/// Run the decode comparison: full-KV decode vs RS-decode, N steps.
///
/// Both decoders start from the same prefill (identical hidden state at
/// every position). Divergence only starts when `rs_decode_step` operates
/// under a bounded window and the full-KV path has access to tokens that
/// the RS path has evicted.
pub fn run_decode_comparison(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    query_type: QueryType,
    window_size: usize,
    decode_steps: usize,
) -> DecodeComparisonResult {
    let prompt = tokenizer
        .decode(token_ids, false)
        .unwrap_or_default();

    // --- Prefill -----------------------------------------------------------
    // Both strategies share the same prefill. Divergence is decode-only.
    let kv = capture_kv(weights, token_ids);
    let rs_result = rs_prefill(weights, token_ids, Some(window_size));

    // Build per-layer mutable KV cache from captured tensors.
    let num_layers = weights.num_layers;
    let mut kv_cache: Vec<(Array2<f32>, Array2<f32>)> = kv.keys
        .into_iter()
        .zip(kv.values)
        .collect();

    // RS store starts with the bounded window from prefill.
    let mut rs_store = rs_result.store;

    // Seed both decoders with the first predicted token (from the identical
    // prefill — this token is the same for both).
    let preds = logits_to_predictions_pub(weights, &kv.hidden, tokenizer, 1, 1.0);
    let seed_token = preds.predictions
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();
    let mut full_kv_token = seed_token.clone();
    let mut rs_token = seed_token;

    let ffn = WeightFfn { weights };
    let mut next_pos = token_ids.len();
    let mut steps = Vec::with_capacity(decode_steps);

    for step in 0..decode_steps {
        // Encode the current token to get its ID.
        let full_id = token_to_id(tokenizer, &full_kv_token);
        let rs_id = token_to_id(tokenizer, &rs_token);

        // --- Full-KV decode step ---
        let h_full = full_kv_step(weights, full_id, &mut kv_cache, next_pos, &ffn);
        let full_preds = logits_to_predictions_pub(weights, &h_full, tokenizer, 3, 1.0);
        let next_full = full_preds.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
        let next_full_prob = full_preds.predictions.first().map(|(_, p)| *p).unwrap_or(0.0);

        // --- RS decode step ---
        let (h_rs, new_store) = match rs_decode_step(weights, rs_id, rs_store) {
            Some(r) => r,
            None => break,
        };
        rs_store = new_store;
        let rs_preds = logits_to_predictions_pub(weights, &h_rs, tokenizer, 3, 1.0);
        let next_rs = rs_preds.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
        let next_rs_prob = rs_preds.predictions.first().map(|(_, p)| *p).unwrap_or(0.0);

        let cosine = hidden_cosine(&h_full, &h_rs);
        let top1_match = next_full == next_rs;

        steps.push(DecodeStep {
            step,
            full_kv_token: full_kv_token.clone(),
            rs_token: rs_token.clone(),
            top1_match,
            hidden_cosine: cosine,
            full_kv_prob: next_full_prob,
            rs_prob: next_rs_prob,
        });

        full_kv_token = next_full;
        rs_token = next_rs;
        next_pos += 1;
    }

    let first_divergence = steps.iter().find(|s| !s.top1_match).map(|s| s.step);
    let match_rate = if steps.is_empty() {
        1.0
    } else {
        steps.iter().filter(|s| s.top1_match).count() as f64 / steps.len() as f64
    };

    DecodeComparisonResult {
        prompt,
        query_type,
        window_size,
        prompt_tokens: token_ids.len(),
        steps,
        first_divergence,
        match_rate,
    }
}

/// Run one full-KV decode step: embed token, run all layers, return hidden.
fn full_kv_step(
    weights: &ModelWeights,
    token_id: u32,
    kv_cache: &mut Vec<(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
    ffn: &WeightFfn,
) -> Array2<f32> {
    let mut h = embed_tokens_pub(weights, &[token_id]);
    for layer in 0..weights.num_layers {
        let old_kv = &kv_cache[layer];
        let (h_post, new_kv) = run_attention_block_decode_step(
            weights, &h, layer, Some(old_kv), abs_position,
        ).expect("full-KV decode step failed");
        kv_cache[layer] = new_kv;
        let (h_out, _) = run_ffn(weights, &h_post, layer, ffn, false);
        h = h_out;
    }
    h
}

/// Cosine similarity of the last row of two hidden-state arrays.
fn hidden_cosine(h1: &Array2<f32>, h2: &Array2<f32>) -> f64 {
    let v1 = h1.row(h1.shape()[0] - 1);
    let v2 = h2.row(h2.shape()[0] - 1);
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
    let n1: f64 = v1.iter().map(|&a| a as f64 * a as f64).sum::<f64>().sqrt();
    let n2: f64 = v2.iter().map(|&a| a as f64 * a as f64).sum::<f64>().sqrt();
    if n1 * n2 < 1e-12 { 0.0 } else { dot / (n1 * n2) }
}

/// Get the first token ID for a token string.
/// Falls back to 0 (BOS/PAD) if the string encodes to multiple or zero tokens.
fn token_to_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> u32 {
    tokenizer
        .encode(token, false)
        .ok()
        .and_then(|e| e.get_ids().first().copied())
        .unwrap_or(0)
}

/// Format a comparison result as a per-step table.
pub fn format_comparison(result: &DecodeComparisonResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n=== Decode Comparison: {:?} | window={} | {} tokens ===\n",
        result.query_type, result.window_size, result.prompt_tokens,
    ));
    out.push_str(&format!("Prompt: \"{}\"\n\n", result.prompt));
    out.push_str(&format!(
        "{:<6} {:<18} {:<18} {:>7} {:>8}\n",
        "Step", "Full-KV", "RS-decode", "Match?", "cos(h)"
    ));
    out.push_str(&"-".repeat(62));
    out.push('\n');

    for s in &result.steps {
        out.push_str(&format!(
            "{:<6} {:<18} {:<18} {:>7} {:>8.6}\n",
            s.step,
            truncate(&s.full_kv_token, 16),
            truncate(&s.rs_token, 16),
            if s.top1_match { "YES" } else { "NO" },
            s.hidden_cosine,
        ));
    }

    out.push_str(&format!(
        "\nMatch rate: {:.1}% ({}/{})",
        result.match_rate * 100.0,
        result.steps.iter().filter(|s| s.top1_match).count(),
        result.steps.len(),
    ));
    if let Some(d) = result.first_divergence {
        out.push_str(&format!("  |  First divergence: step {d}"));
    } else {
        out.push_str("  |  No divergence");
    }
    out.push('\n');
    out
}

/// Format a summary table across window sizes.
pub fn format_window_sweep(results: &[DecodeComparisonResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n{:<12} {:<12} {:>12} {:>12} {}\n",
        "Window", "QueryType", "MatchRate", "FirstDiv", "Verdict"
    ));
    out.push_str(&"-".repeat(60));
    out.push('\n');
    for r in results {
        out.push_str(&format!(
            "{:<12} {:<12} {:>11.1}% {:>12} {}\n",
            r.window_size,
            format!("{:?}", r.query_type),
            r.match_rate * 100.0,
            r.first_divergence.map(|d| d.to_string()).unwrap_or("-".to_string()),
            r.verdict(),
        ));
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..s.char_indices().nth(max - 1).map(|(i, _)| i).unwrap_or(s.len())])
    }
}

/// Default parametric prompts (answers from model weights).
pub fn parametric_prompts() -> Vec<&'static str> {
    vec![
        "The capital of France is",
        "The chemical symbol for gold is",
        "The year the Berlin Wall fell is",
    ]
}

/// In-context prompts (answer planted at beginning, question at end).
/// The gap between planted fact and query is the stress test.
/// With a small window the RS decoder cannot see the planted token.
pub fn in_context_prompts() -> Vec<String> {
    vec![
        // Short gap — fact and query close together
        "The secret code is ZEBRA. The secret code is".to_string(),
        // Medium gap — fact buried under filler
        "Remember: the answer is forty-two. \
         The weather today is pleasant and calm. \
         The answer is".to_string(),
        // Long gap — fact far from query
        "Note: the password is CRIMSON. \
         It is a beautiful day outside. The sun is shining brightly. \
         The birds are singing in the trees. \
         The password is".to_string(),
    ]
}
