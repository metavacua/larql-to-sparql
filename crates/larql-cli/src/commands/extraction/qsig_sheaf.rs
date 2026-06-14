//! Contextual fraction (sheaf global-section obstruction) via linear programming.
//! CF = 1 − max noncontextual weight. CF=0 ⟺ a global section exists ⟺
//! non-contextual (conclusive negative). Subsumes Bell (CHSH) as a cover.

use larql_hilbert::EmpiricalModel;
use minilp::{ComparisonOp, OptimizationDirection, Problem};

/// Contextual fraction of an empirical model via LP. CF = 1 − λ*, where λ* is the
/// max total weight of a noncontextual (global-section-supported) subdistribution
/// consistent with every context marginal. CF=0 ⟺ a global section exists.
#[allow(dead_code)]
pub fn contextual_fraction(model: &EmpiricalModel) -> f64 {
    let mut prob = Problem::new(OptimizationDirection::Maximize);
    // 16 global assignments g over measurements {0,1,2,3}; bit k = outcome of meas k.
    let vars: Vec<_> = (0..16).map(|_| prob.add_var(1.0, (0.0, f64::INFINITY))).collect();
    let bit = |g: usize, k: usize| (g >> k) & 1; // 0 => +1, 1 => −1
    for (ctx, (ai, bj)) in model.settings.iter().enumerate() {
        for o in 0..4 {
            let a = (o >> 1) & 1; // 0=+, 1=−  (major)
            let b = o & 1;        // minor
            let terms: Vec<(_, f64)> = (0..16)
                .filter(|&g| bit(g, *ai) == a && bit(g, *bj) == b)
                .map(|g| (vars[g], 1.0))
                .collect();
            // Σ consistent d_g ≤ empirical P(a,b | ctx)
            prob.add_constraint(terms, ComparisonOp::Le, model.dists[ctx][o].max(0.0));
        }
    }
    match prob.solve() {
        Ok(sol) => (1.0 - sol.objective()).max(0.0),
        Err(_) => 1.0, // infeasible ⇒ fully contextual (shouldn't happen with Le constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_hilbert::{bell_empirical_model, witness::{bell_rho2, product_rho2}};

    #[test]
    fn product_has_a_global_section() {
        let cf = contextual_fraction(&bell_empirical_model(&product_rho2()));
        assert!(cf.abs() < 1e-6, "product admits a global section ⇒ CF=0, got {cf}");
    }

    #[test]
    fn bell_is_contextual() {
        let cf = contextual_fraction(&bell_empirical_model(&bell_rho2()));
        assert!(cf > 1e-3, "Bell on the CHSH cover is contextual ⇒ CF>0, got {cf}");
    }

    #[test]
    fn cf_agrees_with_chsh_on_the_bell_cover() {
        use larql_hilbert::witness::chsh_max;
        let bell = bell_rho2();
        assert!(contextual_fraction(&bell_empirical_model(&bell)) > 1e-3 && chsh_max(&bell) > 2.0);
        let prod = product_rho2();
        assert!(contextual_fraction(&bell_empirical_model(&prod)) < 1e-6 && chsh_max(&prod) <= 2.0 + 1e-9);
    }
}
