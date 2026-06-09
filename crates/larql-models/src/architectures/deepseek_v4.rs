//! DeepSeek V4 Flash architecture — Stage 1 stub.
//!
//! Full architecture per the 2026-04 release:
//! - 284B total / 13B active MoE, FP4 experts + FP8 elsewhere
//! - Hybrid CSA (Compressed Sparse Attention) + HCA (Heavily Compressed
//!   Attention) dispatched per-layer
//! - Manifold-Constrained Hyper-Connections (mHC) — 4 parallel residual
//!   streams with Sinkhorn-Knopp `comb` matrix per layer
//! - Grouped low-rank output projection (8 head-groups × 1024 rank)
//! - Hash-routed MoE bootstrap (first 3 layers)
//! - Sqrt(Softplus) router activation + clamped SwiGLU
//! - 1M native context
//!
//! Build status: Stage 1 (this file) is GGUF arch detection only. The
//! forward path is **not yet implemented**; loading a DSv4 GGUF and
//! attempting inference will return a clear error from the chat
//! handler. Subsequent stages add mHC residuals, CSA/HCA kernels, FP4/FP8
//! quant support, and the end-to-end forward.
//!
//! PR #186 reverted an earlier MLA-based attempt; this is the rebuild
//! that targets the actual hybrid-compressed-attention architecture.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct DeepSeekV4Arch {
    config: ModelConfig,
}

impl DeepSeekV4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for DeepSeekV4Arch {
    fn family(&self) -> &str {
        "deepseek_v4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// DSv4 attention is low-rank Q + latent KV + grouped O with HCA /
    /// indexer / mHC — not standard Q/K/V/O. Routes the vindex extractor
    /// to the dedicated DSv4 path (and gates it out of the standard
    /// writers until that path exists).
    fn uses_dsv4_attention(&self) -> bool {
        true
    }
}
