//! Mistral architecture.
//!
//! Llama-compatible: same tensor keys, norms, activation, RoPE.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct MistralArch {
    config: ModelConfig,
}

impl MistralArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MistralArch {
    fn family(&self) -> &str {
        "mistral"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
