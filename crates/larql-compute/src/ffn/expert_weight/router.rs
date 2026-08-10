//! Expert selection for the reference MoE backend.
//!
//! Whether the selected weights sum to 1 is a **per-architecture** fact, not a
//! convention, and it is the difference between two behaviours:
//!
//! - [`ExpertRoutingPolicy::SoftmaxThenSelect`] — softmax over all `E` logits,
//!   keep the top `k` probabilities as they are. They sum to *less* than 1, by
//!   however much mass the unselected experts hold. Mixtral, and OLMoE with
//!   `norm_topk_prob: false`.
//! - [`ExpertRoutingPolicy::NormalisedOverSelected`] — the weights sum to
//!   exactly 1, whether by renormalising the top-k probabilities (Gemma 4) or
//!   by softmaxing over just the selected logits (GPT-OSS). Those two are
//!   algebraically the same operation.
//!
//! Hardcoding either one is a rescale of the whole expert branch. This module
//! did hardcode the normalised order when first written, and GB caught it
//! immediately: OLMoE went from 0.390 bits/char (HF reference) to 2.677.
//! See `docs/k3-funnel.md` §4.7.

use larql_models::ExpertRoutingPolicy;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One token's expert assignment: `(expert_id, weight)` pairs, `k` of them.
pub type Routing = Vec<(usize, f32)>;

/// Select the top `k` experts for one token and weight them per `policy`.
///
/// `logits` is `[num_experts]` and already includes the router bias. Returns at
/// most `k` entries, ordered by descending logit.
///
/// Both branches softmax with the standard max-shift, so a dominant logit
/// cannot overflow. They differ only in *which* set the denominator sums over:
/// all experts, or the selected ones.
pub fn select(logits: &[f32], k: usize, policy: ExpertRoutingPolicy) -> Routing {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }

    // Rank by logit. Softmax is monotone, so ranking by logit and ranking by
    // probability select the same set — the policy changes the weights, never
    // the selection. `k` is 4-ish and `logits` 32-ish, so a full sort beats a
    // heap and ties break by first index, matching torch.topk.
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    let denominator_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator: f32 = match policy {
        // Sum over ALL experts: the selected probabilities keep whatever
        // fraction of the distribution they actually hold.
        ExpertRoutingPolicy::SoftmaxThenSelect => {
            logits.iter().map(|l| (l - denominator_max).exp()).sum()
        }
        // Sum over the selected experts only, so they normalise to 1.
        ExpertRoutingPolicy::NormalisedOverSelected => ranked
            .iter()
            .take(k)
            .map(|&(_, l)| (l - denominator_max).exp())
            .sum(),
    };

    ranked.truncate(k);
    for (_, w) in ranked.iter_mut() {
        *w = if denominator > 0.0 {
            (*w - denominator_max).exp() / denominator
        } else {
            0.0
        };
    }
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    const NORMALISED: ExpertRoutingPolicy = ExpertRoutingPolicy::NormalisedOverSelected;
    const RAW: ExpertRoutingPolicy = ExpertRoutingPolicy::SoftmaxThenSelect;

    /// 32 experts with a clear winning group — the 4-of-32 shape GPT-OSS uses.
    fn logits_32() -> Vec<f32> {
        (0..32).map(|i| if i < 4 { 2.0 } else { 1.0 }).collect()
    }

    // ── the two policies are genuinely different ────────────────────────

    /// The defining difference: normalised weights sum to 1, raw ones sum to
    /// whatever mass the selected experts actually hold. Conflating them
    /// rescales the whole expert branch — the bug GB caught on OLMoE.
    #[test]
    fn the_two_policies_disagree_on_total_mass() {
        let logits = logits_32();
        let normalised: f32 = select(&logits, 4, NORMALISED).iter().map(|&(_, w)| w).sum();
        let raw: f32 = select(&logits, 4, RAW).iter().map(|&(_, w)| w).sum();

        assert!((normalised - 1.0).abs() < 1e-6, "sum was {normalised}");
        assert!(raw < 0.7, "raw policy should hold well under 1, got {raw}");
    }

    /// Raw weights must equal a plain softmax over all experts, restricted to
    /// the selected ones — computed here independently of the implementation.
    #[test]
    fn raw_policy_matches_a_full_softmax() {
        let logits = logits_32();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let denom: f32 = exps.iter().sum();

        for (expert, weight) in select(&logits, 4, RAW) {
            let want = exps[expert] / denom;
            assert!(
                (weight - want).abs() < 1e-6,
                "expert {expert}: got {weight}, want {want}"
            );
        }
    }

    /// Normalising the raw weights must reproduce the normalised policy
    /// exactly — the algebraic identity the enum's docs claim.
    #[test]
    fn normalising_the_raw_weights_reproduces_the_other_policy() {
        let logits = logits_32();
        let raw = select(&logits, 4, RAW);
        let total: f32 = raw.iter().map(|&(_, w)| w).sum();
        let normalised = select(&logits, 4, NORMALISED);

        for ((e_raw, w_raw), (e_norm, w_norm)) in raw.iter().zip(normalised.iter()) {
            assert_eq!(e_raw, e_norm);
            assert!(
                (w_raw / total - w_norm).abs() < 1e-6,
                "expert {e_raw}: {} vs {w_norm}",
                w_raw / total
            );
        }
    }

    /// The policy changes the weights, never the selection.
    #[test]
    fn both_policies_select_the_same_experts() {
        let logits = [0.1, 5.0, 0.2, 4.0, 0.3, 3.0, 0.4, 2.0];
        let ids = |p| -> Vec<usize> { select(&logits, 3, p).into_iter().map(|(e, _)| e).collect() };
        assert_eq!(ids(NORMALISED), ids(RAW));
    }

    // ── selection ──────────────────────────────────────────────────────

    #[test]
    fn picks_the_highest_logits() {
        let logits = [1.0, 9.0, 2.0, 8.0];
        let ids: Vec<usize> = select(&logits, 2, NORMALISED)
            .into_iter()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn ordered_by_descending_logit() {
        let logits = [3.0, 1.0, 2.0];
        let routing = select(&logits, 3, NORMALISED);
        assert_eq!(routing[0].0, 0);
        assert_eq!(routing[1].0, 2);
        assert_eq!(routing[2].0, 1);
        assert!(routing[0].1 > routing[1].1);
        assert!(routing[1].1 > routing[2].1);
    }

    // ── numerical behaviour ────────────────────────────────────────────

    /// Softmax depends only on logit differences, so adding a constant to
    /// every logit must not move any weight — under either policy. This is
    /// also what makes a uniform router bias a no-op.
    #[test]
    fn invariant_to_a_constant_logit_shift() {
        let base = [0.5, 2.0, 1.0, 3.0];
        let shifted: Vec<f32> = base.iter().map(|l| l + 100.0).collect();
        for policy in [NORMALISED, RAW] {
            let a = select(&base, 2, policy);
            let b = select(&shifted, 2, policy);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.0, y.0);
                assert!((x.1 - y.1).abs() < 1e-6, "policy {policy:?} shifted");
            }
        }
    }

    #[test]
    fn a_dominant_logit_cannot_overflow() {
        let logits = [1000.0, 1.0, 2.0];
        for policy in [NORMALISED, RAW] {
            let routing = select(&logits, 2, policy);
            assert!(routing.iter().all(|&(_, w)| w.is_finite()));
            // The dominant expert takes essentially all the mass either way.
            assert!((routing[0].1 - 1.0).abs() < 1e-6);
        }
    }

    // ── degenerate inputs ──────────────────────────────────────────────

    /// With k >= num_experts the two policies must coincide: selecting every
    /// expert makes the two denominators the same sum.
    #[test]
    fn k_larger_than_expert_count_is_clamped_and_policies_agree() {
        let logits = [1.0, 2.0];
        for policy in [NORMALISED, RAW] {
            let routing = select(&logits, 8, policy);
            assert_eq!(routing.len(), 2);
            let total: f32 = routing.iter().map(|&(_, w)| w).sum();
            assert!((total - 1.0).abs() < 1e-6, "policy {policy:?}: {total}");
        }
    }

    #[test]
    fn zero_k_selects_nothing() {
        assert!(select(&[1.0, 2.0], 0, NORMALISED).is_empty());
        assert!(select(&[1.0, 2.0], 0, RAW).is_empty());
    }

    #[test]
    fn empty_logits_select_nothing() {
        assert!(select(&[], 4, NORMALISED).is_empty());
        assert!(select(&[], 4, RAW).is_empty());
    }

    #[test]
    fn top_one_normalised_is_a_certainty() {
        let routing = select(&[1.0, 5.0, 3.0], 1, NORMALISED);
        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].0, 1);
        assert!((routing[0].1 - 1.0).abs() < 1e-6);
    }

    /// The same single selection under the raw policy keeps its real
    /// probability, which is strictly below 1.
    #[test]
    fn top_one_raw_keeps_its_own_probability() {
        let routing = select(&[1.0, 5.0, 3.0], 1, RAW);
        assert_eq!(routing[0].0, 1);
        assert!(
            routing[0].1 < 0.95,
            "raw top-1 should not be a certainty, got {}",
            routing[0].1
        );
    }
}
