//! Loading `config.json` from a model directory and enforcing presence
//! of fields without a defensible architecture-class default.
//!
//! Silently defaulting topology fields (the historical behaviour) makes a
//! "wrong directory" / "incomplete download" surface as a panic deep in
//! the extract pipeline (issue #22 — `could not broadcast array from
//! shape: [2560] to: [2048]`). Failing here keeps the error message
//! attached to the actual cause.

use std::path::{Path, PathBuf};

use super::ModelError;

/// HF-convention config file name read from a model directory.
pub(super) const CONFIG_FILE_NAME: &str = "config.json";

/// Nested-config wrapper used by multimodal models (Gemma 3 IT, Gemma 4).
pub(super) const CONFIG_KEY_TEXT_CONFIG: &str = "text_config";

// JSON keys for required topology fields. These have no defensible
// architecture-class default — silently substituting a guess masks real
// "wrong directory" / "incomplete download" failure modes and surfaces
// later as a broadcast/matmul panic.
pub(super) const CONFIG_KEY_HIDDEN_SIZE: &str = "hidden_size";
pub(super) const CONFIG_KEY_NUM_HIDDEN_LAYERS: &str = "num_hidden_layers";
pub(super) const CONFIG_KEY_INTERMEDIATE_SIZE: &str = "intermediate_size";

/// Fields whose absence makes the config unsuitable for inferring topology.
/// Other fields (head_dim, num_kv_heads, rope_theta, ...) have well-defined
/// architecture-class defaults in transformers and are intentionally left
/// to fall through.
pub(super) const REQUIRED_CONFIG_FIELDS: &[&str] = &[
    CONFIG_KEY_HIDDEN_SIZE,
    CONFIG_KEY_NUM_HIDDEN_LAYERS,
    CONFIG_KEY_INTERMEDIATE_SIZE,
];

/// Resolve the conventional `<model_dir>/config.json` path.
pub(super) fn config_path(model_dir: &Path) -> PathBuf {
    model_dir.join(CONFIG_FILE_NAME)
}

/// Read and parse a `config.json` at the given path.
///
/// Returns [`ModelError::ConfigMissing`] when the file does not exist,
/// rather than the prior behavior of synthesising an empty `{}` and
/// letting magic-number defaults pretend the model was successfully
/// described.
pub(super) fn read_config_json(config_path: &Path) -> Result<serde_json::Value, ModelError> {
    if !config_path.exists() {
        return Err(ModelError::ConfigMissing(config_path.to_path_buf()));
    }
    let text = std::fs::read_to_string(config_path)?;
    Ok(serde_json::from_str::<serde_json::Value>(&text)?)
}

/// Fail loudly when a parsed config is missing any field whose silent
/// default would diverge from a real model's topology. Both top-level and
/// nested `text_config` (multimodal) layouts are accepted; the field is
/// present when it resolves under either.
pub(super) fn require_config_fields(
    config: &serde_json::Value,
    config_path: &Path,
) -> Result<(), ModelError> {
    let text_config = config.get(CONFIG_KEY_TEXT_CONFIG).unwrap_or(config);
    let missing: Vec<&'static str> = REQUIRED_CONFIG_FIELDS
        .iter()
        .copied()
        .filter(|field| {
            text_config.get(*field).and_then(|v| v.as_u64()).is_none()
                && config.get(*field).and_then(|v| v.as_u64()).is_none()
        })
        .collect();
    if !missing.is_empty() {
        return Err(ModelError::ConfigFieldsMissing {
            path: config_path.to_path_buf(),
            missing,
        });
    }
    Ok(())
}
