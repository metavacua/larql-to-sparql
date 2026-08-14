//! `vec_inject` entry types.
//!
//! An entry represents a single retrievable fact extracted from the document
//! during the store build. At query time, `retrieve` finds entries relevant
//! to the query, and `inject` additively modifies the residual stream at
//! `injection_layer` with the token embedding of the entry's `token_id`,
//! scaled by `coefficient`.
//!
//! Storage layout matches the Python format in
//! `apollo-demo/apollo11_store/entries.npz`:
//!
//! ```text
//! entries: structured array with fields
//!   (token_id: u32, coefficient: f32, window_id: u16,
//!    position_in_window: u16, fact_id: u16)
//! ```

use serde::{Deserialize, Serialize};

/// A single vec_inject entry. One document can have thousands; Apollo 11
/// has 3,585 entries across 176 windows.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VecInjectEntry {
    /// Token ID whose embedding gets injected.
    pub token_id: u32,
    /// Amplification multiplier applied to the embedding before injection.
    /// Apollo's coefficients run ~2-10× the embedding's natural norm.
    pub coefficient: f32,
    /// Window this fact was extracted from.
    pub window_id: u16,
    /// Position within that window (0..window_size).
    pub position_in_window: u16,
    /// Grouping key — multiple entries sharing a fact_id form a
    /// multi-token fact (e.g. a proper noun like "John Coyle").
    pub fact_id: u16,
}

/// Injection knobs used at query time. Configured once per store; the
/// Apollo 11 demo uses `injection_layer=30, inject_coefficient=10.0` on
/// Gemma 3 4B.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Layer at which to add retrieved entries to the residual stream.
    pub injection_layer: usize,
    /// Global multiplier on top of each entry's per-entry coefficient.
    pub inject_coefficient: f32,
    /// Maximum entries to inject per query (top-k after retrieval).
    pub top_k: usize,
    /// BOS token id to strip from the front of the query when building the
    /// `window_tokens ++ query_tokens` context (a BOS mid-context is
    /// nonsense). `None` means "strip nothing" — there is deliberately no
    /// hardcoded model-specific default; the engine falls back to the
    /// structural `weights.arch.bos_token_id()` when this is `None`.
    /// Set via the engine spec (`apollo:bos=2`) for tokenizers that
    /// prepend BOS themselves (e.g. Gemma 2/3, whose arch hook is `None`
    /// because larql never needs to prepend for them).
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        // Apollo 11 defaults from the demo manifest.
        Self {
            injection_layer: 30,
            inject_coefficient: 10.0,
            top_k: 8,
            bos_token_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_injection_matches_apollo() {
        let cfg = InjectionConfig::default();
        assert_eq!(cfg.injection_layer, 30);
        assert_eq!(cfg.inject_coefficient, 10.0);
        assert_eq!(cfg.top_k, 8);
        // No hardcoded model-specific BOS: unset means "strip nothing".
        assert_eq!(cfg.bos_token_id, None);
    }

    #[test]
    fn entry_is_pod_sized() {
        // Must be layout-compatible with the Python structured dtype:
        // token_id u32 (4) + coef f32 (4) + window_id u16 (2) +
        // pos_in_window u16 (2) + fact_id u16 (2) = 14 bytes + padding
        let size = std::mem::size_of::<VecInjectEntry>();
        assert!(size >= 14, "entry smaller than expected: {size}");
        assert!(size <= 20, "entry has too much padding: {size}");
    }
}
