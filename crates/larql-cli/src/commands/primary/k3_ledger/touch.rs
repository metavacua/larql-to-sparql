//! DEC-8.6 — per-token weight touch once the bytes are resident, split into the
//! part expert-side levers can reduce and the part they cannot.
//!
//! Pure. Residency removes the storage link but not the weight-touch ceiling:
//! every token still reads whatever the forward pass touches, at memory
//! bandwidth. The split is the whole point — K3's always-on half is the *larger*
//! half, and no expert-side lever touches it.

use serde::Serialize;

use super::frontier::ServingPremises;
use super::geometry::K3Geometry;

/// The demo tier per `vindex-factory.md` §15.3: hot experts only, down-row
/// payloads, up folded to per-feature scalars. Note this tier's own definition
/// already banks the row-sparsity and up-fold results that other rungs exist to
/// test — they are load-bearing for it, not a post-mortem.
#[derive(Debug, Clone, Copy)]
pub struct SliceTier {
    pub size_bytes: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TouchLedger {
    pub dense_params: u64,
    pub routed_params: u64,
    pub activated_params: u64,
    pub dense_share: f64,
    pub dense_bytes: f64,
    pub routed_bytes: f64,
    pub total_bytes: f64,
    pub tok_s_total: f64,
    /// What remains reachable when routed bytes go to zero — the R4 ceiling.
    pub tok_s_dense_only: f64,
    /// Total reduction expert-side levers can supply, by construction.
    pub expert_side_ceiling: f64,
    pub reduction_needed: Option<f64>,
}

pub fn touch_ledger(geom: &K3Geometry, p: &ServingPremises, target_tok_s: f64) -> TouchLedger {
    let dense_params = geom.dense_params();
    let routed_params = geom.routed_activated_params();
    let dense = geom.dense_bytes(p.dense_all_in_bits);
    let routed = geom.routed_bytes_per_position();
    let total = dense + routed;
    let budget = p.budget_per_token(target_tok_s);

    TouchLedger {
        dense_params,
        routed_params,
        activated_params: dense_params + routed_params,
        dense_share: dense_params as f64 / (dense_params + routed_params) as f64,
        dense_bytes: dense,
        routed_bytes: routed,
        total_bytes: total,
        tok_s_total: p.effective_bw() / total,
        tok_s_dense_only: p.effective_bw() / dense,
        expert_side_ceiling: total / dense,
        reduction_needed: (budget > 0.0).then(|| total / budget),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceComposition {
    pub slice_bytes: f64,
    pub resident_experts: f64,
    pub resident_fraction: f64,
    pub uniform_hits_per_layer: f64,
}

/// What fits in a slice after the dense half, and how much routing it covers.
///
/// The per-expert demo payload is the down branch alone; the up branch folds to
/// scalars and the gate is served from a resident sketch.
pub fn slice_composition(
    geom: &K3Geometry,
    p: &ServingPremises,
    tier: &SliceTier,
) -> SliceComposition {
    let payload = geom.branch_bytes[1] as f64;
    let expert_budget = (tier.size_bytes - geom.dense_bytes(p.dense_all_in_bits)).max(0.0);
    let resident = expert_budget / payload;
    let total_slots = (geom.n_experts * geom.n_moe_layers()) as f64;
    let fraction = resident / total_slots;
    SliceComposition {
        slice_bytes: tier.size_bytes,
        resident_experts: resident,
        resident_fraction: fraction,
        uniform_hits_per_layer: geom.top_k as f64 * fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    fn ledger() -> TouchLedger {
        touch_ledger(&k3_reference(), &ServingPremises::default(), 20.0)
    }

    #[test]
    fn dense_is_the_larger_half() {
        let l = ledger();
        assert!(l.dense_share > 0.5, "dense share {}", l.dense_share);
        assert!(l.dense_bytes > l.routed_bytes);
    }

    #[test]
    fn the_measured_touch_figures_hold() {
        let l = ledger();
        approx(l.dense_bytes / 1e9, 29.86, 0.5);
        approx(l.routed_bytes / 1e9, 25.83, 0.01);
        approx(l.tok_s_total, 6.6, 0.2);
    }

    #[test]
    fn dense_alone_exceeds_the_whole_twenty_tok_s_budget() {
        let (g, p) = (k3_reference(), ServingPremises::default());
        let l = touch_ledger(&g, &p, 20.0);
        assert!(l.dense_bytes > p.budget_per_token(20.0));
    }

    #[test]
    fn expert_side_levers_cap_below_two_x() {
        // Routed experts are under half the bytes, so driving them to zero is
        // worth less than 2x on the total. This is why no expert-side result
        // could ever have decided the headline target.
        let l = ledger();
        assert!(
            l.expert_side_ceiling < 2.0,
            "ceiling {}",
            l.expert_side_ceiling
        );
    }

    #[test]
    fn the_needed_reduction_exceeds_what_experts_can_give() {
        let l = ledger();
        assert!(l.reduction_needed.unwrap() > l.expert_side_ceiling);
    }

    #[test]
    fn a_demo_slice_holds_only_a_few_percent_of_experts() {
        let (g, p) = (k3_reference(), ServingPremises::default());
        for gb in [55.0, 60.0, 65.0] {
            let c = slice_composition(
                &g,
                &p,
                &SliceTier {
                    size_bytes: gb * 1e9,
                },
            );
            assert!(
                (0.04..0.09).contains(&c.resident_fraction),
                "{gb} GB slice holds {} of experts",
                c.resident_fraction
            );
            assert!(c.uniform_hits_per_layer < 2.0);
        }
    }

    #[test]
    fn a_slice_smaller_than_the_dense_half_holds_no_experts() {
        let (g, p) = (k3_reference(), ServingPremises::default());
        let c = slice_composition(&g, &p, &SliceTier { size_bytes: 10.0e9 });
        approx(c.resident_experts, 0.0, 1e-9);
    }
}
