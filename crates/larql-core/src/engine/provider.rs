use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPrediction {
    pub token: String,
    #[serde(default)]
    pub token_id: i64,
    pub probability: f64,
    #[serde(default)]
    pub logit: f64,
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub prompt: String,
    pub predictions: Vec<TokenPrediction>,
}

impl PredictionResult {
    pub fn top(&self) -> Option<&TokenPrediction> {
        self.predictions.first()
    }

    pub fn top_token(&self) -> Option<&str> {
        self.top().map(|p| p.token.as_str())
    }

    pub fn top_probability(&self) -> f64 {
        self.top().map_or(0.0, |p| p.probability)
    }
}

/// The single interface between LARQL and any model backend.
pub trait ModelProvider: Send + Sync {
    fn model_name(&self) -> &str;

    fn predict_next_token(
        &self,
        prompt: &str,
        top_k: usize,
    ) -> Result<PredictionResult, ProviderError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("empty response from model")]
    EmptyResponse,
    #[error("model not loaded")]
    NotLoaded,
    #[error("timeout after {0}ms")]
    Timeout(u64),
    #[error("{0}")]
    Other(String),
}
