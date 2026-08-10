//! `spec.source` — the pinned upstream model.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use serde::{Deserialize, Serialize};

/// Pinned upstream model. Part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    /// Upstream HF repo, `owner/name`.
    pub hf_repo: String,
    /// REQUIRED: full 40-hex-character commit SHA, never a branch name.
    pub revision: String,
    /// Licence key — must resolve in the recipes repo's
    /// `policy/licences.yaml`. That allowlist check is out of scope for
    /// this crate's structural validator; it runs in `validate.yml`.
    pub licence: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let s = Source {
            hf_repo: "google/gemma-3-4b-it".into(),
            revision: "9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82".into(),
            licence: "gemma".into(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hf_repo, s.hf_repo);
        assert_eq!(back.revision, s.revision);
        assert_eq!(back.licence, s.licence);
    }
}
