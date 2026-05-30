//! Auto-detect model architecture from `config.json`.
//!
//! The module is organised so each concern lives in its own file:
//! - [`config_io`] reads `config.json` from disk and enforces presence of
//!   topology fields that have no defensible architecture-class default.
//! - [`parser`] turns a parsed JSON value into a [`ModelConfig`], honouring
//!   both multimodal nesting (`text_config`) and flat layouts.
//! - This file owns [`ModelError`] and the public entry points, including
//!   the family-routing dispatch that maps `model_type` → concrete
//!   [`ModelArchitecture`].
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::architectures::deepseek::DeepSeekArch;
use crate::architectures::deepseek_v4::DeepSeekV4Arch;
use crate::architectures::gemma2::Gemma2Arch;
use crate::architectures::gemma3::Gemma3Arch;
use crate::architectures::gemma4::Gemma4Arch;
use crate::architectures::generic::GenericArch;
use crate::architectures::gpt2::Gpt2Arch;
use crate::architectures::gpt_oss::GptOssArch;
use crate::architectures::granite::GraniteArch;
use crate::architectures::llama::LlamaArch;
use crate::architectures::mistral::MistralArch;
use crate::architectures::mixtral::MixtralArch;
use crate::architectures::qwen::QwenArch;
use crate::architectures::starcoder2::StarCoder2Arch;
use crate::architectures::tinymodel::TinyModelArch;
use crate::config::ModelArchitecture;
mod config_io;
mod parser;

use config_io::{config_path, read_config_json, require_config_fields};
use parser::parse_model_config;

// `ModelError` now lives in the portable `crate::error` module so quant/dequant
// (which is target-agnostic) can return it. Re-exported here for callers that
// reference `detect::ModelError`.
pub use crate::error::ModelError;

/// Read `config.json` from a model directory and return the architecture.
///
/// Errors with [`ModelError::ConfigMissing`] when the directory has no
/// `config.json`, and with [`ModelError::ConfigFieldsMissing`] when the
/// file exists but lacks topology fields without a defensible
/// architecture-class default. This prevents the silent fallback-to-defaults
/// path from inventing a wrong topology and then panicking deep inside the
/// extract pipeline (issue #22).
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_path = config_path(model_dir);
    let config_json = read_config_json(&config_path)?;
    require_config_fields(&config_json, &config_path)?;
    Ok(detect_from_json(&config_json))
}

/// Read `config.json` from a model directory, detect the architecture, and validate it.
pub fn detect_architecture_validated(
    model_dir: &Path,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_architecture(model_dir)?;
    validate_detected_architecture(arch)
}

/// Detect architecture from an already-parsed `config.json` value.
///
/// Infallible by design: callers building an in-memory config for tests
/// or programmatic loads (e.g. GGUF-derived configs) can keep terse setup
/// and rely on [`ModelArchitecture::validate`] downstream to catch any
/// missing required fields.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    let model_type = model_config.model_type.as_str();

    match model_type {
        // Gemma family
        t if t.starts_with("gemma4") => Box::new(Gemma4Arch::from_config(model_config)),
        t if t.starts_with("gemma3") => Box::new(Gemma3Arch::from_config(model_config)),
        t if t.starts_with("gemma2") || t == "gemma" => {
            Box::new(Gemma2Arch::from_config(model_config))
        }
        // Llama family
        t if t.starts_with("llama") => Box::new(LlamaArch::from_config(model_config)),
        // Mistral (dense)
        "mistral" => Box::new(MistralArch::from_config(model_config)),
        // Mixtral (MoE) — block_sparse_moe pattern
        "mixtral" => Box::new(MixtralArch::from_config(model_config)),
        // GPT-2 (non-gated FFN, LayerNorm, learned positional embeddings)
        "gpt2" => Box::new(Gpt2Arch::from_config(model_config)),
        // GPT-OSS (MoE, MXFP4 packed experts)
        "gpt_oss" => Box::new(GptOssArch::from_config(model_config)),
        // Qwen family (dense and MoE share same keys)
        t if t.starts_with("qwen") => Box::new(QwenArch::from_config(model_config)),
        // DeepSeek-V4 (MoE + MLA + MXFP4 + HCA attention; new tensor naming)
        "deepseek_v4" => Box::new(DeepSeekV4Arch::from_config(model_config)),
        // DeepSeek V2/V3 family (MoE + MLA, model.* prefixed keys)
        t if t.starts_with("deepseek") => Box::new(DeepSeekArch::from_config(model_config)),
        // StarCoder 2
        "starcoder2" => Box::new(StarCoder2Arch::from_config(model_config)),
        // Granite family (dense and MoE share same base keys)
        t if t.starts_with("granite") => Box::new(GraniteArch::from_config(model_config)),
        // TinyModel — research-scale decoder used for LARQL compile/walk work
        "tinymodel" => Box::new(TinyModelArch::from_config(model_config)),
        // Unknown — generic fallback
        _ => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Detect architecture from an already-parsed `config.json` value and validate it.
pub fn detect_from_json_validated(
    config: &serde_json::Value,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_from_json(config);
    validate_detected_architecture(arch)
}

pub(crate) fn validate_detected_architecture(
    arch: Box<dyn ModelArchitecture>,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    match arch.validate() {
        Ok(()) => Ok(arch),
        Err(errors) => Err(ModelError::ConfigValidation(errors)),
    }
}

#[cfg(test)]
mod tests;
