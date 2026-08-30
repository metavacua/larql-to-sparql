//! Model dimensions pulled from a target's `config.json`, via the same
//! architecture detection larql's extractor itself uses — so a size
//! estimate reads the same fields the real extraction would.

/// The handful of dimensions [`super::bytes::estimate_component_bytes`]
/// needs. Not the full `larql_models::config::ModelConfig` — just the
/// fields a byte-count formula depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelDims {
    /// Transformer layer count.
    pub num_layers: u64,
    /// Residual stream width.
    pub hidden_size: u64,
    /// Dense FFN intermediate width (or per-expert width for MoE models
    /// that don't declare a separate `moe_intermediate_size`).
    pub intermediate_size: u64,
    /// Vocabulary size. `0` if the config doesn't declare one — the
    /// embed/lm_head components then estimate to `0` rather than the
    /// call failing.
    pub vocab_size: u64,
    /// Expert count for MoE models; `None` for dense.
    pub num_experts: Option<u64>,
}

impl ModelDims {
    /// Detect the architecture from a raw `config.json` value and pull
    /// out the fields a size estimate needs.
    pub fn from_config_json(config: &serde_json::Value) -> Self {
        let arch = larql_models::detect::detect_from_json(config);
        let c = arch.config();
        Self {
            num_layers: c.num_layers as u64,
            hidden_size: c.hidden_size as u64,
            intermediate_size: c.intermediate_size as u64,
            vocab_size: c.vocab_size.unwrap_or(0) as u64,
            num_experts: c.num_experts.map(|n| n as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_dense_model_dims() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
            "vocab_size": 262208,
        });
        let dims = ModelDims::from_config_json(&config);
        assert_eq!(dims.hidden_size, 2560);
        assert_eq!(dims.intermediate_size, 10240);
        assert_eq!(dims.num_layers, 34);
        assert_eq!(dims.vocab_size, 262208);
        assert_eq!(dims.num_experts, None);
    }

    #[test]
    fn reads_moe_model_expert_count() {
        let config = serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_local_experts": 8,
        });
        let dims = ModelDims::from_config_json(&config);
        assert_eq!(dims.num_experts, Some(8));
    }

    #[test]
    fn missing_vocab_size_defaults_to_zero_rather_than_failing() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
        });
        let dims = ModelDims::from_config_json(&config);
        assert_eq!(dims.vocab_size, 0);
    }
}
