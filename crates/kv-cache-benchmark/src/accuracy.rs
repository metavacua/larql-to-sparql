//! Accuracy test infrastructure.
//!
//! Proves memory savings don't come at the cost of correctness.
//! Six categories: top-1 match, KL divergence, needle-in-haystack,
//! multi-turn retention, generation coherence, adversarial.

use serde::Serialize;

/// Result of an accuracy test for one strategy on one prompt.
#[derive(Debug, Clone, Serialize)]
pub struct AccuracyResult {
    pub strategy: String,
    pub test_name: String,
    pub prompt: String,

    // Token-level
    pub top1_match: bool,
    pub top5_overlap: f32,
    pub baseline_token_rank: u32,

    // Distribution-level
    pub kl_divergence: f64,
    pub js_divergence: f64,
    pub correct_token_prob: f64,

    // Generation-level
    pub tokens_before_diverge: Option<u32>,
    pub token_match_rate: Option<f32>,

    // Needle-level
    pub needle_found: Option<bool>,
    pub needle_exact_match: Option<bool>,
}

impl AccuracyResult {
    pub fn token_match(strategy: &str, test_name: &str, prompt: &str, matched: bool) -> Self {
        Self {
            strategy: strategy.to_string(),
            test_name: test_name.to_string(),
            prompt: prompt.to_string(),
            top1_match: matched,
            top5_overlap: if matched { 1.0 } else { 0.0 },
            baseline_token_rank: if matched { 1 } else { 0 },
            kl_divergence: if matched { 0.0 } else { f64::NAN },
            js_divergence: if matched { 0.0 } else { f64::NAN },
            correct_token_prob: if matched { 1.0 } else { 0.0 },
            tokens_before_diverge: None,
            token_match_rate: None,
            needle_found: None,
            needle_exact_match: None,
        }
    }

    pub fn needle(strategy: &str, test_name: &str, prompt: &str, found: bool, exact: bool) -> Self {
        Self {
            strategy: strategy.to_string(),
            test_name: test_name.to_string(),
            prompt: prompt.to_string(),
            top1_match: found,
            top5_overlap: 0.0,
            baseline_token_rank: 0,
            kl_divergence: 0.0,
            js_divergence: 0.0,
            correct_token_prob: 0.0,
            tokens_before_diverge: None,
            token_match_rate: None,
            needle_found: Some(found),
            needle_exact_match: Some(exact),
        }
    }
}

/// Compute KL divergence: sum(p * log(p / q)) where p = baseline, q = strategy.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1e-12 && qi > 1e-12 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl
}

/// Compute Jensen-Shannon divergence (symmetric, bounded 0-1).
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&a, &b)| (a + b) / 2.0).collect();
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// Compute softmax of logits.
pub fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Top-K overlap: fraction of top-K tokens shared between two ranked lists.
pub fn top_k_overlap(a: &[u32], b: &[u32], k: usize) -> f32 {
    let a_set: std::collections::HashSet<u32> = a.iter().take(k).copied().collect();
    let b_set: std::collections::HashSet<u32> = b.iter().take(k).copied().collect();
    let intersection = a_set.intersection(&b_set).count();
    intersection as f32 / k as f32
}

/// First divergence point between two token sequences.
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<u32> {
    for (i, (&ta, &tb)) in a.iter().zip(b.iter()).enumerate() {
        if ta != tb {
            return Some(i as u32);
        }
    }
    None
}

/// Token-level match rate between two sequences.
pub fn token_match_rate(a: &[u32], b: &[u32]) -> f32 {
    if a.is_empty() { return 0.0; }
    let matches = a.iter().zip(b.iter()).filter(|(&x, &y)| x == y).count();
    matches as f32 / a.len().min(b.len()) as f32
}

/// Mean reciprocal rank: rank of target in predictions list.
pub fn reciprocal_rank(predictions: &[u32], target: u32) -> f64 {
    for (i, &p) in predictions.iter().enumerate() {
        if p == target {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

// ── Test prompt sets ──

/// Diverse factual prompts for Category 1 testing.
pub fn factual_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of Brazil is", "Brasilia"),
        ("The capital of Australia is", "Canberra"),
        ("The capital of Canada is", "Ottawa"),
        ("The capital of Egypt is", "Cairo"),
        ("The capital of India is", "New Delhi"),
        ("The currency of Japan is the", "yen"),
        ("The currency of the UK is the", "pound"),
        ("The currency of India is the", "rupee"),
        ("Mozart was born in", "Salzburg"),
        ("Einstein was born in", "Ulm"),
        ("Shakespeare was born in", "Stratford"),
        ("Water freezes at", "0"),
        ("The speed of light is approximately", "300"),
        ("The chemical symbol for gold is", "Au"),
        ("The longest river in Africa is the", "Nile"),
    ]
}

/// Diverse prompts spanning multiple categories.
pub fn diverse_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("To be or not to be, that is the", "question"),
        ("How are you today? I'm doing", "well"),
        ("The square root of 144 is", "12"),
        ("In Python, print('hello') outputs", "hello"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("H2O is the chemical formula for", "water"),
        ("The first president of the United States was", "George"),
        ("One kilometer equals", "1000"),
        ("The Mona Lisa was painted by", "Leonardo"),
        ("In music, there are 12 notes in a", "chromatic"),
    ]
}

/// Generate a haystack with a planted needle.
pub fn generate_haystack(
    total_tokens: usize,
    needle_position: usize,
    needle: &str,
) -> (String, String) {
    let filler_word = "the quick brown fox jumps over the lazy dog ";
    let words_before = needle_position / 6; // ~6 chars per word
    let words_after = (total_tokens - needle_position) / 6;

    let mut context = String::new();
    for _ in 0..words_before {
        context.push_str(filler_word);
    }
    context.push_str(needle);
    context.push(' ');
    for _ in 0..words_after {
        context.push_str(filler_word);
    }

    (context, needle.to_string())
}

/// Build a multi-turn fact retention conversation.
pub fn build_retention_conversation(num_turns: usize) -> Vec<ConversationTurn> {
    let facts = [("My name is Alice and I work at Anthropic.", "name", "Alice"),
        ("I'm based in San Francisco.", "location", "San Francisco"),
        ("My project is called Lighthouse.", "project", "Lighthouse"),
        ("My favorite color is blue.", "color", "blue"),
        ("I have two cats named Luna and Sol.", "pets", "Luna")];

    let queries = vec![
        ("What project am I working on?", "project", "Lighthouse"),
        ("Where do I live?", "location", "San Francisco"),
        ("What's my name?", "name", "Alice"),
        ("What's my favorite color?", "color", "blue"),
        ("What are my cats' names?", "pets", "Luna"),
    ];

    let mut turns = Vec::new();

    // Establish facts in first 3-5 turns
    for (i, (text, key, _val)) in facts.iter().enumerate() {
        if i < 3 || i < num_turns / 5 {
            turns.push(ConversationTurn {
                turn: turns.len() + 1,
                role: "user".to_string(),
                text: text.to_string(),
                is_query: false,
                expected_fact: None,
                fact_key: Some(key.to_string()),
            });
        }
    }

    // Filler turns
    let fillers = [
        "Tell me about the weather.",
        "What's a good recipe for pasta?",
        "How does photosynthesis work?",
        "What are the planets in order?",
        "Explain quantum computing briefly.",
        "What's the tallest mountain?",
        "Who wrote Romeo and Juliet?",
        "What year was the moon landing?",
    ];

    while turns.len() < num_turns.saturating_sub(queries.len()) {
        let filler = fillers[(turns.len() - 3) % fillers.len()];
        turns.push(ConversationTurn {
            turn: turns.len() + 1,
            role: "user".to_string(),
            text: filler.to_string(),
            is_query: false,
            expected_fact: None,
            fact_key: None,
        });
    }

    // Query turns at the end
    for (text, key, expected) in &queries {
        if turns.len() < num_turns {
            turns.push(ConversationTurn {
                turn: turns.len() + 1,
                role: "user".to_string(),
                text: text.to_string(),
                is_query: true,
                expected_fact: Some(expected.to_string()),
                fact_key: Some(key.to_string()),
            });
        }
    }

    turns
}

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationTurn {
    pub turn: usize,
    pub role: String,
    pub text: String,
    pub is_query: bool,
    pub expected_fact: Option<String>,
    pub fact_key: Option<String>,
}

// ── Summary formatting ──

/// Aggregate accuracy results into a summary table.
pub fn format_accuracy_summary(results: &[AccuracyResult]) -> String {
    let mut out = String::new();

    // Group by strategy
    let mut strategies: Vec<String> = results.iter().map(|r| r.strategy.clone()).collect();
    strategies.sort();
    strategies.dedup();

    out.push_str("\n=== Accuracy Summary ===\n\n");
    out.push_str(&format!(
        "{:<20} {:>10} {:>10} {:>10}\n",
        "Strategy", "Top-1 %", "Mean KL", "Needles"
    ));
    out.push_str(&"-".repeat(55));
    out.push('\n');

    for strategy in &strategies {
        let strat_results: Vec<&AccuracyResult> = results
            .iter()
            .filter(|r| &r.strategy == strategy)
            .collect();

        let total = strat_results.len();
        let top1_matches = strat_results.iter().filter(|r| r.top1_match).count();
        let match_pct = if total > 0 {
            top1_matches as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        let kl_values: Vec<f64> = strat_results
            .iter()
            .filter(|r| r.kl_divergence.is_finite())
            .map(|r| r.kl_divergence)
            .collect();
        let mean_kl = if kl_values.is_empty() {
            f64::NAN
        } else {
            kl_values.iter().sum::<f64>() / kl_values.len() as f64
        };

        let needles: Vec<&AccuracyResult> = strat_results
            .iter()
            .filter(|r| r.needle_found.is_some())
            .copied()
            .collect();
        let needles_found = needles.iter().filter(|r| r.needle_found == Some(true)).count();
        let needle_str = if needles.is_empty() {
            "n/a".to_string()
        } else {
            format!("{}/{}", needles_found, needles.len())
        };

        out.push_str(&format!(
            "{:<20} {:>9.1}% {:>10.4} {:>10}\n",
            strategy, match_pct, mean_kl, needle_str,
        ));
    }

    out
}
