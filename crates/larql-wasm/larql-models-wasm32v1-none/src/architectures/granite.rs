//! IBM Granite architecture.
//!
//! Llama-compatible: same tensor keys, norms, activation, RoPE.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct GraniteArch {
    config: ModelConfig,
}

impl GraniteArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GraniteArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
