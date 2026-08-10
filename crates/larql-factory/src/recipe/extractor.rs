//! `spec.extractor` — the pinned extractor invocation.

// BTreeMap needs alloc (not core) on wasm32 -- unlike HashMap/HashSet
// (pattern 4), it needs no custom hasher, so a plain cfg'd import
// suffices instead of a wrapper type in collections.rs.
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
#[cfg(target_arch = "wasm32")]
use alloc::collections::BTreeMap;
#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use larql_vindex_spec::{ExtractLevel, StorageDtype};
use serde::{Deserialize, Serialize};

/// Pinned extractor invocation. Part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Extractor {
    /// Extractor tool name. Only `"larql"` is currently supported —
    /// checked against `validate::constants::SUPPORTED_EXTRACTOR_TOOL`.
    pub tool: String,
    /// Released tag only — never a git sha of a branch (PIN §13.3).
    pub version: String,
    /// Extraction tier. Reuses the vindex-v1 manifest's own enum so the
    /// recipe and the produced manifest can't drift.
    pub level: ExtractLevel,
    /// Storage precision for float tensors.
    pub dtype: StorageDtype,
    /// Extractor-specific flags (`down_q4k`, `drop_gate_vectors`, ...).
    /// A `BTreeMap` rather than `serde_json::Map` keeps key order
    /// deterministic for `build_id` regardless of `serde_json`'s
    /// `preserve_order` feature state in a downstream consumer.
    #[serde(default)]
    pub options: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json_including_options() {
        let mut options = BTreeMap::new();
        options.insert("down_q4k".into(), serde_json::Value::Bool(true));
        let e = Extractor {
            tool: "larql".into(),
            version: "0.14.2".into(),
            level: ExtractLevel::Inference,
            dtype: StorageDtype::F16,
            options,
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: Extractor = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool, e.tool);
        assert_eq!(back.version, e.version);
        assert_eq!(back.level, e.level);
        assert_eq!(back.dtype, e.dtype);
        assert_eq!(back.options, e.options);
    }

    #[test]
    fn options_default_to_empty_when_absent() {
        let json = r#"{"tool":"larql","version":"0.14.2","level":"inference","dtype":"f16"}"#;
        let e: Extractor = serde_json::from_str(json).unwrap();
        assert!(e.options.is_empty());
    }
}
