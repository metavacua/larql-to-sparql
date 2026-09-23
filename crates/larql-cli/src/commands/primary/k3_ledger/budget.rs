//! DEC-8.0 — the per-token fetch budget when experts live on external storage.
//!
//! Pure. Divides a link's bytes/second by a target tokens/second to get a miss
//! budget, and compares it to what a token actually has to fetch.
//!
//! Two things this module exists to keep honest. A lever's divisor must not
//! already be inside the baseline it divides (the original ~50x figure was the
//! down-branch-only fetch, so crediting a further branch-drop double-counted
//! it). And row-granular fetch is bound by *request rate*, not bandwidth, which
//! the byte arithmetic alone will never show.

use serde::Serialize;

use super::geometry::K3Geometry;

/// A page is the smallest unit the storage stack actually moves.
pub const PAGE_BYTES: f64 = 4096.0;

#[derive(Debug, Clone, Copy)]
pub struct LinkPremises {
    pub gb_s: f64,
    pub target_tok_s: f64,
    pub ports: u32,
}

impl Default for LinkPremises {
    fn default() -> Self {
        Self {
            gb_s: 3.5,
            target_tok_s: 20.0,
            ports: 1,
        }
    }
}

impl LinkPremises {
    pub fn bytes_per_second(&self) -> f64 {
        self.gb_s * 1e9 * self.ports as f64
    }

    /// Bytes that may cross the link per token.
    pub fn miss_budget(&self) -> f64 {
        self.bytes_per_second() / self.target_tok_s
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MissBudget {
    pub miss_budget_bytes: f64,
    pub required_fetch_bytes: f64,
    pub gap: f64,
    pub expert_visits: u64,
    pub bytes_allowed_per_visit: f64,
    pub fraction_of_expert_allowed: f64,
}

pub fn miss_budget(geom: &K3Geometry, link: &LinkPremises) -> MissBudget {
    let budget = link.miss_budget();
    let required = geom.routed_bytes_per_position();
    let visits = (geom.top_k * geom.n_moe_layers()) as u64;
    let per_visit = budget / visits as f64;
    MissBudget {
        miss_budget_bytes: budget,
        required_fetch_bytes: required,
        gap: required / budget,
        expert_visits: visits,
        bytes_allowed_per_visit: per_visit,
        fraction_of_expert_allowed: per_visit / geom.expert_bytes as f64,
    }
}

/// The second constraint: what row-granular access costs in *requests*.
#[derive(Debug, Clone, Serialize)]
pub struct Granularity {
    pub feature_row_bytes: f64,
    pub features_selected: f64,
    pub ideal_bytes_per_visit: f64,
    pub paged_bytes_per_visit: f64,
    pub read_amplification: f64,
    pub iops_required: f64,
    /// What the link could do at page granularity if it were purely
    /// bandwidth-limited — an upper bound that ignores per-command overhead.
    pub iops_bandwidth_equivalent: f64,
    pub iops_shortfall: f64,
}

/// Price row-granular access at a given feature fraction.
///
/// `matrices` is how many of the three branches a feature must be read from.
/// `locality` discounts the request count for measured routing re-use.
pub fn granularity(
    geom: &K3Geometry,
    link: &LinkPremises,
    feature_fraction: f64,
    matrices: f64,
    locality: f64,
) -> Granularity {
    let row = geom.branch_bytes[1] as f64 / geom.moe_intermediate as f64;
    let n_sel = feature_fraction * geom.moe_intermediate as f64;
    let visits = (geom.top_k * geom.n_moe_layers()) as f64;

    let ideal = n_sel * matrices * row;
    // An unaligned extent of length L spans 1 + L/PAGE pages in expectation.
    let pages_per_extent = 1.0 + row / PAGE_BYTES;
    let paged = n_sel * matrices * pages_per_extent * PAGE_BYTES;

    let iops = n_sel * matrices * visits / locality * link.target_tok_s;
    let bw_equiv = link.bytes_per_second() / PAGE_BYTES;

    Granularity {
        feature_row_bytes: row,
        features_selected: n_sel,
        ideal_bytes_per_visit: ideal,
        paged_bytes_per_visit: paged,
        read_amplification: paged / ideal,
        iops_required: iops,
        iops_bandwidth_equivalent: bw_equiv,
        iops_shortfall: iops / bw_equiv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn the_gap_is_the_measured_one() {
        let m = miss_budget(&k3_reference(), &LinkPremises::default());
        approx(m.miss_budget_bytes / 1e6, 175.0, 0.1);
        approx(m.required_fetch_bytes / 1e9, 25.83, 0.01);
        approx(m.gap, 147.6, 0.2);
    }

    #[test]
    fn a_token_may_fetch_well_under_one_percent_of_each_expert() {
        let m = miss_budget(&k3_reference(), &LinkPremises::default());
        assert_eq!(m.expert_visits, 1472);
        assert!(m.fraction_of_expert_allowed < 0.01);
    }

    #[test]
    fn down_branch_only_reproduces_the_baseline_that_was_double_counted() {
        // The original ~50x was this figure, so crediting a further branch-drop
        // lever on top of it counted the same saving twice.
        let g = k3_reference();
        let link = LinkPremises::default();
        let down_only = g.branch_bytes[1] as f64 * (g.top_k * g.n_moe_layers()) as f64;
        approx(down_only / link.miss_budget(), 49.2, 0.3);
    }

    #[test]
    fn striping_ports_scales_the_budget_linearly() {
        let g = k3_reference();
        let one = miss_budget(&g, &LinkPremises::default());
        let three = miss_budget(
            &g,
            &LinkPremises {
                ports: 3,
                ..Default::default()
            },
        );
        approx(one.gap / three.gap, 3.0, 1e-9);
    }

    #[test]
    fn sub_page_rows_cost_read_amplification() {
        let g = k3_reference();
        let gr = granularity(
            &g,
            &LinkPremises {
                ports: 3,
                ..Default::default()
            },
            0.064,
            2.0,
            1.5,
        );
        approx(gr.feature_row_bytes, 1904.0, 1.0);
        assert!(
            gr.read_amplification > 3.0,
            "amplification {}",
            gr.read_amplification
        );
    }

    #[test]
    fn row_fetch_is_request_rate_bound_not_bandwidth_bound() {
        // The finding that promoted the systems check ahead of the fidelity one.
        let g = k3_reference();
        let gr = granularity(
            &g,
            &LinkPremises {
                ports: 3,
                ..Default::default()
            },
            0.064,
            2.0,
            1.5,
        );
        assert!(
            gr.iops_shortfall > 1.0,
            "request rate should bind before bandwidth, shortfall {}",
            gr.iops_shortfall
        );
    }

    #[test]
    fn locality_discounts_the_request_rate() {
        let g = k3_reference();
        let link = LinkPremises::default();
        let none = granularity(&g, &link, 0.064, 2.0, 1.0);
        let with = granularity(&g, &link, 0.064, 2.0, 1.5);
        approx(none.iops_required / with.iops_required, 1.5, 1e-9);
    }
}
