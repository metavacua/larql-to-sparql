#[cfg(feature = "http")]
use reqwest::blocking::Client;

use super::provider::*;

/// Connects to any OpenAI-compatible completions API.
/// Works with: ollama, vLLM, llama.cpp server, LM Studio.
#[cfg(feature = "http")]
pub struct HttpProvider {
    client: Client,
    base_url: String,
    name: String,
}

#[cfg(feature = "http")]
impl HttpProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("http client"),
            base_url: base_url.into(),
            name: model.into(),
        }
    }

    /// Ollama at localhost:11434.
    pub fn ollama(model: &str) -> Self {
        Self::new("http://localhost:11434/v1", model)
    }

    /// llama.cpp server at localhost:8080.
    pub fn llama_cpp(model: &str) -> Self {
        Self::new("http://localhost:8080/v1", model)
    }
}

#[cfg(feature = "http")]
impl ModelProvider for HttpProvider {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn predict_next_token(
        &self,
        prompt: &str,
        top_k: usize,
    ) -> Result<PredictionResult, ProviderError> {
        let body = serde_json::json!({
            "model": &self.name,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": top_k,
            "temperature": 0,
        });

        let resp = self
            .client
            .post(format!("{}/completions", self.base_url))
            .json(&body)
            .send()
            .map_err(|e| ProviderError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ProviderError::Http(format!("status {}", resp.status())));
        }

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| ProviderError::Http(e.to_string()))?;

        let choice = &json["choices"][0];
        let mut predictions = Vec::new();

        // Parse logprobs if available
        if let Some(top_logprobs) = choice["logprobs"]["top_logprobs"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_object())
        {
            for (tok, lp) in top_logprobs {
                let logprob = lp.as_f64().unwrap_or(-100.0);
                predictions.push(TokenPrediction {
                    token: tok.clone(),
                    token_id: -1,
                    probability: logprob.exp(),
                    logit: logprob,
                });
            }
        }

        // Fallback: use generated text if no logprobs
        if predictions.is_empty() {
            if let Some(text) = choice["text"].as_str() {
                if !text.is_empty() {
                    predictions.push(TokenPrediction {
                        token: text.to_string(),
                        token_id: -1,
                        probability: 1.0,
                        logit: 0.0,
                    });
                }
            }
        }

        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        Ok(PredictionResult {
            prompt: prompt.to_string(),
            predictions,
        })
    }
}
