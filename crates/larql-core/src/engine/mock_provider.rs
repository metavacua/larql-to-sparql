#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use ::core::sync::atomic::{AtomicUsize, Ordering};
use crate::collections::HashMap;

use super::provider::*;

/// A mock provider for testing. Knowledge is injected at construction time —
/// nothing is hardcoded in the engine. Use `MockProvider::new()` for an empty
/// provider, or `MockProvider::with_knowledge()` to supply test fixtures.
pub struct MockProvider {
    name: String,
    /// Maps (prompt_suffix) -> (answer, probability)
    knowledge: HashMap<String, (String, f64)>,
    call_count: AtomicUsize,
}

impl MockProvider {
    /// Create an empty mock provider with no knowledge.
    pub fn new() -> Self {
        Self {
            name: "mock/empty".to_string(),
            knowledge: HashMap::default(),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a mock provider with the given knowledge entries.
    /// Each entry is (prompt_suffix, answer, probability).
    pub fn with_knowledge(entries: Vec<(String, String, f64)>) -> Self {
        let mut knowledge = HashMap::default();
        for (prompt, answer, prob) in entries {
            knowledge.insert(prompt, (answer, prob));
        }
        Self {
            name: "mock/knowledge-base".to_string(),
            knowledge,
            call_count: AtomicUsize::new(0),
        }
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelProvider for MockProvider {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn predict_next_token(
        &self,
        prompt: &str,
        _top_k: usize,
    ) -> Result<PredictionResult, ProviderError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);

        let trimmed = prompt.trim();
        for (suffix, (answer, prob)) in &self.knowledge {
            if trimmed.ends_with(suffix.trim()) || trimmed == suffix.trim() {
                let first_token = answer.split_whitespace().next().unwrap_or(answer);
                // f64::ln() is a std-only inherent method (core's f64 has no
                // transcendental math -- it needs a libm implementation,
                // which std links against the platform's). libm is the
                // standard no_std-compatible pure-Rust replacement. See
                // docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md,
                // Task 7, pattern 6: core-f64-lacks-transcendental-math.
                #[cfg(target_arch = "wasm32")]
                let logit = libm::log(*prob);
                #[cfg(not(target_arch = "wasm32"))]
                let logit = prob.ln();
                return Ok(PredictionResult {
                    prompt: prompt.to_string(),
                    predictions: vec![TokenPrediction {
                        token: format!(" {first_token}"),
                        token_id: -1,
                        probability: *prob,
                        logit,
                    }],
                });
            }
        }

        Ok(PredictionResult {
            prompt: prompt.to_string(),
            predictions: vec![],
        })
    }
}
