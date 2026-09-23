//! `spec.outputs` — the presets to slice and their shard caps.

use serde::{Deserialize, Serialize};

/// One output artifact: a preset slice (or `"full"`, the unsliced
/// extract) plus where it's published. Part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSpec {
    /// `full`, or a `larql slice --preset` name (`client`, `attn`,
    /// `embed`, `server`, `browse`, `router`, `expert-server`, `all`).
    pub preset: String,
    /// Shard cap override. Defaults to the vindex-v1 20 GiB cap when
    /// absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_cap_gib: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_cap_omitted_when_absent() {
        let o = OutputSpec {
            preset: "full".into(),
            shard_cap_gib: None,
        };
        let json = serde_json::to_string(&o).unwrap();
        assert!(!json.contains("shard_cap_gib"));
    }

    #[test]
    fn round_trips_with_shard_cap() {
        let o = OutputSpec {
            preset: "expert-server".into(),
            shard_cap_gib: Some(20),
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: OutputSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.preset, o.preset);
        assert_eq!(back.shard_cap_gib, o.shard_cap_gib);
    }
}
