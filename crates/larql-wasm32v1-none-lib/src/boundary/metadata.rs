//! Phase 2 — per-boundary confidence metadata.

use alloc::vec::Vec;
use super::frame::BoundaryAgreement;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundaryMetadata {
    pub raw_top1_token: u32,
    pub compressed_top1_token: Option<u32>,
    pub boundary_agreement: BoundaryAgreement,
    pub raw_logit_margin: f32,
    pub raw_log_prob_margin: f32,
    pub raw_top1_prob: f32,
    pub codec_fragile: bool,
    pub boundary_fragile: bool,
}

/// Compute boundary metadata from logit slices.
///
/// # Panics
/// Panics if either slice is empty or if they have different lengths.
pub fn compute(raw_logits: &[f32], hat_logits: Option<&[f32]>) -> BoundaryMetadata {
    assert!(!raw_logits.is_empty(), "raw_logits must not be empty");
    if let Some(hat) = hat_logits {
        assert_eq!(
            raw_logits.len(),
            hat.len(),
            "raw_logits and hat_logits must have the same length"
        );
    }

    let log_probs = log_softmax(raw_logits);
    let (top1, top2) = top2_by_logit(raw_logits);

    let raw_logit_margin = raw_logits[top1] - raw_logits[top2];
    let raw_log_prob_margin = log_probs[top1] - log_probs[top2];
    let raw_top1_prob = libm::expf(log_probs[top1]);

    let (compressed_top1_token, boundary_agreement) = match hat_logits {
        None => (None, BoundaryAgreement::NotChecked),
        Some(hat) => {
            let hat_top1 = argmax(hat);
            let agreement = if hat_top1 == top1 {
                BoundaryAgreement::Agrees
            } else {
                BoundaryAgreement::Disagrees
            };
            (Some(hat_top1 as u32), agreement)
        }
    };

    let codec_fragile = matches!(boundary_agreement, BoundaryAgreement::Disagrees);

    BoundaryMetadata {
        raw_top1_token: top1 as u32,
        compressed_top1_token,
        boundary_agreement,
        raw_logit_margin,
        raw_log_prob_margin,
        raw_top1_prob,
        codec_fragile,
        boundary_fragile: false,
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max).collect();
    let log_sum = libm::logf(shifted.iter().map(|&x| libm::expf(x)).sum::<f32>());
    shifted.iter().map(|&x| x - log_sum).collect()
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn top2_by_logit(logits: &[f32]) -> (usize, usize) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best;
            best = (i, v);
        } else if v > second.1 {
            second = (i, v);
        }
    }
    (best.0, second.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaked_logits(top: usize, v_top: f32, v_rest: f32, n: usize) -> Vec<f32> {
        let mut l = alloc::vec![v_rest; n];
        l[top] = v_top;
        l
    }

    #[test]
    fn agrees_when_hat_same_argmax() {
        let raw = peaked_logits(42, 10.0, 0.0, 1000);
        let hat = raw.clone();
        let meta = compute(&raw, Some(&hat));
        assert_eq!(meta.raw_top1_token, 42);
        assert_eq!(meta.compressed_top1_token, Some(42));
        assert!(matches!(meta.boundary_agreement, BoundaryAgreement::Agrees));
        assert!(!meta.codec_fragile);
    }

    #[test]
    fn disagrees_when_hat_different_argmax() {
        let raw = peaked_logits(42, 10.0, 0.0, 1000);
        let hat = peaked_logits(99, 10.0, 0.0, 1000);
        let meta = compute(&raw, Some(&hat));
        assert!(matches!(
            meta.boundary_agreement,
            BoundaryAgreement::Disagrees
        ));
        assert!(meta.codec_fragile);
    }

    #[test]
    fn not_checked_when_no_hat() {
        let raw = peaked_logits(5, 3.0, 0.0, 100);
        let meta = compute(&raw, None);
        assert!(matches!(
            meta.boundary_agreement,
            BoundaryAgreement::NotChecked
        ));
        assert!(!meta.codec_fragile);
        assert!(meta.compressed_top1_token.is_none());
    }

    #[test]
    fn log_prob_margin_always_positive() {
        let raw = alloc::vec![3.0f32, 2.0, 1.0, 0.5];
        let meta = compute(&raw, None);
        assert!(meta.raw_log_prob_margin > 0.0);
    }

    #[test]
    fn uniform_logits_have_small_margin() {
        let raw = alloc::vec![1.0f32; 1000];
        let meta = compute(&raw, None);
        assert!(meta.raw_log_prob_margin < 0.01);
    }
}
