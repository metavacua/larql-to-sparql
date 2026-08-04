//! Types unique to card rendering: what a build actually produced.
//! Nothing in the workspace tracks this yet — there's no build driver
//! (docs/vindex-factory.md §7) to produce it from. This is a
//! best-guess shape scoped to what §8/§9 of the spec need for the
//! card; expect it to grow once `larql build` exists and can validate
//! these fields against real output.

use larql_vindex_spec::VindexManifest;
use serde::{Deserialize, Serialize};

use crate::{Recipe, Verify};

/// Everything [`super::render`] needs, bundled so the entry point takes
/// one argument instead of five.
pub struct CardInputs<'a> {
    /// The recipe that produced this build.
    pub recipe: &'a Recipe,
    /// The vindex-v1 manifest the build wrote.
    pub manifest: &'a VindexManifest,
    /// What §8's verification stages found.
    pub verification: &'a VerificationReport,
    /// Every output this build published, for the slice table.
    pub slices: &'a [SliceSummary],
    /// This build's `build_id` (see [`crate::build_id`]).
    pub build_id: &'a str,
}

/// What the §8 verification stages found for a build.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationReport {
    /// §8.1 reconstruction-fidelity result.
    pub reconstruction: ReconstructionResult,
    /// §8.1 logit-match result.
    pub logit_match: LogitMatchResult,
    /// Whether VERIFY-B (§8.2, re-pulled from the Hub) ran, vs only
    /// the local VERIFY-A.
    pub verified_from_hub: bool,
}

/// §8.1 reconstruction-fidelity measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionResult {
    /// Layers actually sampled.
    pub layers_sampled: u32,
    /// Measured maximum absolute difference.
    pub max_abs_diff: f64,
    /// Measured minimum cosine similarity.
    pub min_cosine: f64,
}

/// §8.1 logit-match measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitMatchResult {
    /// Measured top-1 agreement.
    pub top1_agreement: f64,
    /// Measured bits-per-char drift.
    pub bits_per_char_drift: f64,
}

/// One published output's size, for the card's slice table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SliceSummary {
    /// `full`, or a slice preset name — matches
    /// [`crate::OutputSpec::preset`].
    pub preset: String,
    /// Total size on disk / on the Hub, in bytes.
    pub size_bytes: u64,
}

impl VerificationReport {
    /// True if both measurements are within `verify`'s declared
    /// thresholds — reuses the recipe's own thresholds rather than a
    /// second, parallel set of pass/fail numbers.
    pub fn passed(&self, verify: &Verify) -> bool {
        self.reconstruction.max_abs_diff <= verify.reconstruction.max_abs_diff
            && self.reconstruction.min_cosine >= verify.reconstruction.min_cosine
            && self.logit_match.top1_agreement >= verify.logit_match.top1_agreement_min
            && self.logit_match.bits_per_char_drift <= verify.logit_match.bits_per_char_drift_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Verify {
        crate::test_support::sample_recipe().spec.verify
    }

    fn passing_report() -> VerificationReport {
        VerificationReport {
            reconstruction: ReconstructionResult {
                layers_sampled: 8,
                max_abs_diff: 0.0,
                min_cosine: 1.0,
            },
            logit_match: LogitMatchResult {
                top1_agreement: 1.0,
                bits_per_char_drift: 0.0,
            },
            verified_from_hub: true,
        }
    }

    #[test]
    fn passed_true_when_every_measurement_meets_threshold() {
        assert!(passing_report().passed(&thresholds()));
    }

    #[test]
    fn passed_false_when_reconstruction_drifts() {
        let mut report = passing_report();
        report.reconstruction.max_abs_diff = 0.01;
        assert!(!report.passed(&thresholds()));
    }

    #[test]
    fn passed_false_when_cosine_drops_below_threshold() {
        let mut report = passing_report();
        report.reconstruction.min_cosine = 0.99;
        assert!(!report.passed(&thresholds()));
    }

    #[test]
    fn passed_false_when_top1_agreement_drops_below_threshold() {
        let mut report = passing_report();
        report.logit_match.top1_agreement = 0.5;
        assert!(!report.passed(&thresholds()));
    }

    #[test]
    fn passed_false_when_bits_per_char_drift_exceeds_threshold() {
        let mut report = passing_report();
        report.logit_match.bits_per_char_drift = 1.0;
        assert!(!report.passed(&thresholds()));
    }
}
