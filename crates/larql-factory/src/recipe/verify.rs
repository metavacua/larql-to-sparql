//! `spec.verify` — reconstruction and logit-match thresholds (§8.1).

use serde::{Deserialize, Serialize};

/// Verification thresholds. Not part of `build_id` — tightening a
/// threshold re-verifies, it doesn't force a rebuild (§5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verify {
    /// Bit-identical reconstruction check on sampled layers.
    pub reconstruction: Reconstruction,
    /// Teacher-forced logit agreement against a reference forward pass.
    pub logit_match: LogitMatch,
    /// Re-pull the published bytes and verify those, not the local
    /// build directory (§8.2).
    pub from_hub: bool,
}

/// §8.1 reconstruction-fidelity thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reconstruction {
    /// Number of layers sampled for the check.
    pub layers_sampled: u32,
    /// RNG seed for layer sampling.
    pub seed: u64,
    /// Maximum allowed absolute difference. `0.0` for a lossless carve.
    pub max_abs_diff: f64,
    /// Minimum allowed cosine similarity. `1.0` for a lossless carve.
    pub min_cosine: f64,
}

/// §8.1 logit-match thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitMatch {
    /// Path (within the recipes repo) to the content-addressed prompt set.
    pub prompt_set: String,
    /// Minimum top-1 agreement with the reference forward pass.
    pub top1_agreement_min: f64,
    /// Maximum allowed bits-per-char drift.
    pub bits_per_char_drift_max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let v = Verify {
            reconstruction: Reconstruction {
                layers_sampled: 8,
                seed: 1729,
                max_abs_diff: 0.0,
                min_cosine: 1.0,
            },
            logit_match: LogitMatch {
                prompt_set: "verify/gemma-smoke-32.jsonl".into(),
                top1_agreement_min: 0.99,
                bits_per_char_drift_max: 0.005,
            },
            from_hub: true,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: Verify = serde_json::from_str(&json).unwrap();
        assert_eq!(back.reconstruction.layers_sampled, 8);
        assert_eq!(back.logit_match.prompt_set, "verify/gemma-smoke-32.jsonl");
        assert!(back.from_hub);
    }
}
