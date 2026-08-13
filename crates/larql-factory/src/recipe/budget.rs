//! `spec.budget` — executor selection and resource/cost ceilings.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use serde::{Deserialize, Serialize};

/// Executor selection and resource/cost ceilings. Not part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Budget {
    /// `auto`, `inline`, or `leased`.
    pub executor: String,
    /// Hard wall-clock ceiling in minutes.
    pub max_wall_minutes: u32,
    /// Hard USD ceiling.
    pub max_usd: f64,
    /// Declared resource footprint, used for the `inline`/`leased`
    /// executor decision (PIN §13.4) and the PREFLIGHT stage.
    pub requires: BudgetRequires,
}

/// Declared resource footprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetRequires {
    /// Scratch disk required, in GiB.
    pub disk_gib: u32,
    /// RAM required, in GiB.
    pub ram_gib: u32,
    /// Minimum downlink bandwidth, in Mbps.
    pub net_down_mbps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let b = Budget {
            executor: "auto".into(),
            max_wall_minutes: 240,
            max_usd: 12.0,
            requires: BudgetRequires {
                disk_gib: 512,
                ram_gib: 64,
                net_down_mbps: 1000,
            },
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(back.executor, b.executor);
        assert_eq!(back.max_wall_minutes, b.max_wall_minutes);
        assert_eq!(back.requires.disk_gib, b.requires.disk_gib);
    }
}
