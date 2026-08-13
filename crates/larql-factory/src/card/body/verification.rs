//! §9's verification-report summary section.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use crate::{VerificationReport, Verify};

/// Render the verification summary — pass/fail plus the measured
/// numbers against the recipe's own declared thresholds.
pub fn render_verification(verify: &Verify, report: &VerificationReport) -> String {
    let status = if report.passed(verify) {
        "PASSED"
    } else {
        "FAILED"
    };
    let hub_check = if report.verified_from_hub {
        "re-pulled from the Hub and re-verified (§8.2)"
    } else {
        "verified locally only"
    };
    format!(
        "## Verification — {status}\n\n\
         - Reconstruction: {sampled} layers sampled, max abs diff {abs_diff}, min cosine {cosine} \
         (thresholds: {abs_diff_thr}, {cosine_thr})\n\
         - Logit match: {top1}% top-1 agreement, {drift} bits/char drift \
         (thresholds: {top1_thr}%, {drift_thr})\n\
         - {hub_check}",
        sampled = report.reconstruction.layers_sampled,
        abs_diff = report.reconstruction.max_abs_diff,
        cosine = report.reconstruction.min_cosine,
        abs_diff_thr = verify.reconstruction.max_abs_diff,
        cosine_thr = verify.reconstruction.min_cosine,
        top1 = report.logit_match.top1_agreement * 100.0,
        drift = report.logit_match.bits_per_char_drift,
        top1_thr = verify.logit_match.top1_agreement_min * 100.0,
        drift_thr = verify.logit_match.bits_per_char_drift_max,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogitMatchResult, ReconstructionResult};

    fn sample_verify() -> Verify {
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
    fn reports_passed_when_within_thresholds() {
        let out = render_verification(&sample_verify(), &passing_report());
        assert!(out.starts_with("## Verification — PASSED"));
        assert!(out.contains("re-pulled from the Hub"));
    }

    #[test]
    fn reports_failed_when_outside_thresholds() {
        let mut report = passing_report();
        report.reconstruction.max_abs_diff = 1.0;
        let out = render_verification(&sample_verify(), &report);
        assert!(out.starts_with("## Verification — FAILED"));
    }

    #[test]
    fn notes_local_only_verification() {
        let mut report = passing_report();
        report.verified_from_hub = false;
        let out = render_verification(&sample_verify(), &report);
        assert!(out.contains("verified locally only"));
    }
}
