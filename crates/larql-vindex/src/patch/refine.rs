//! Gate refine pass for `COMPILE INTO VINDEX WITH REFINE`.
//!
//! The refine pass orthogonalises each patched gate against (a) the other
//! patched gates at the same layer and (b) any optional decoy residuals
//! supplied by the caller. It is the load-bearing fix for cross-fact
//! bleed at compile time, validated end-to-end in
//! `experiments/14_vindex_compilation` on Gemma 3 4B (10/10 retrieval,
//! 0/4 regression bleed under refine + decoys).
//!
//! The math is plain Gram-Schmidt: for each target gate, subtract the
//! projection onto every suppression vector (other gates + decoys), one
//! at a time. The result has reduced inner product with each suppression
//! direction so the compiled gate doesn't fire on residuals it
//! shouldn't fire on.
//!
//! No model dependency, no forward passes — pure linear algebra over
//! `Array1<f32>` slices the caller already has in hand.

use ndarray::Array1;

/// Input to the refine pass: a single fact's identifying coordinates
/// plus the unrefined gate vector synthesised at INSERT time.
#[derive(Debug, Clone)]
pub struct RefineInput {
    pub layer: usize,
    pub feature: usize,
    pub gate: Array1<f32>,
}

/// Output of the refine pass for a single fact: the refined (gate-only)
/// vector and the fraction of the original norm retained after
/// orthogonalisation. Norm retained < 1.0 means the refine pass
/// projected non-trivial overlap with neighbours; values near 1.0 mean
/// the fact had little template overlap with the rest of the
/// constellation.
#[derive(Debug, Clone)]
pub struct RefinedGate {
    pub layer: usize,
    pub feature: usize,
    pub gate: Array1<f32>,
    pub retained_norm: f32,
}

/// Aggregate result of running refine over a constellation. Carries the
/// per-fact refined gates plus summary stats so the executor can report
/// what happened to the user.
#[derive(Debug, Clone)]
pub struct RefineResult {
    pub gates: Vec<RefinedGate>,
    pub min_retained: f32,
    pub max_retained: f32,
    pub median_retained: f32,
    pub n_decoys: usize,
}

/// Refine a constellation of patched gates.
///
/// `inputs` is the full set of facts being baked. Within each layer, the
/// refine pass orthogonalises each fact's gate against the other facts
/// in the same layer plus any `decoy_residuals` supplied by the caller.
///
/// Cross-layer interaction is intentionally not modelled: facts at
/// different layers fire on different residual signatures and don't
/// interfere with each other under the FFN math, so refining them
/// against each other would be both expensive and incorrect.
///
/// The decoy residuals are taken in their full original space (no
/// per-layer slicing). Callers should supply residuals captured at the
/// install layer, which is what `larql-inference` produces.
///
/// Empty input returns an empty result with `median_retained = 1.0`.
pub fn refine_gates(
    inputs: &[RefineInput],
    decoy_residuals: &[Array1<f32>],
) -> RefineResult {
    if inputs.is_empty() {
        return RefineResult {
            gates: Vec::new(),
            min_retained: 1.0,
            max_retained: 1.0,
            median_retained: 1.0,
            n_decoys: decoy_residuals.len(),
        };
    }

    // Group facts by layer so each fact's suppression set is built from
    // peers at the same layer. The decoys apply to every layer because
    // the caller is responsible for capturing them at the right depth.
    let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, fact) in inputs.iter().enumerate() {
        by_layer.entry(fact.layer).or_default().push(i);
    }

    let mut refined: Vec<Option<RefinedGate>> = vec![None; inputs.len()];
    for indices in by_layer.values() {
        for &i in indices {
            let target = &inputs[i].gate;
            let original_norm = target.dot(target).sqrt();

            // Suppression set = other facts at this layer + all decoys.
            let mut suppress: Vec<&Array1<f32>> = indices
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| &inputs[j].gate)
                .collect();
            for d in decoy_residuals {
                suppress.push(d);
            }

            let refined_gate = orthogonalise(target, &suppress);
            let retained = if original_norm > 1e-8 {
                refined_gate.dot(&refined_gate).sqrt() / original_norm
            } else {
                0.0
            };

            refined[i] = Some(RefinedGate {
                layer: inputs[i].layer,
                feature: inputs[i].feature,
                gate: refined_gate,
                retained_norm: retained,
            });
        }
    }

    let gates: Vec<RefinedGate> = refined.into_iter().map(|r| r.unwrap()).collect();
    let mut retained: Vec<f32> = gates.iter().map(|g| g.retained_norm).collect();
    retained.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = retained[retained.len() / 2];

    RefineResult {
        min_retained: *retained.first().unwrap(),
        max_retained: *retained.last().unwrap(),
        median_retained: median,
        n_decoys: decoy_residuals.len(),
        gates,
    }
}

/// Project `target` onto the orthogonal complement of `span(suppress)`.
///
/// Does proper modified Gram-Schmidt: first orthonormalises the
/// suppress vectors (so correlated vectors don't lead to incorrect
/// projections), then subtracts each orthonormal component from the
/// target. With an orthonormal basis `q_1..q_k` of `span(suppress)`,
/// the result `v = target - Σ (target·q_i) q_i` is exactly orthogonal
/// to every original suppress vector — even when the suppress set was
/// highly correlated (cos ~ 0.99 at compose-time on Gemma L26).
///
/// The naive single-pass version (`v -= (v·u)/||u||² · u` for each raw
/// `u`) only guarantees `v ⊥ u_last`; earlier orthogonality is lost as
/// later projections re-introduce components. At N=50 with correlated
/// template-dominated residuals this produced cross-slot interference
/// strong enough to collapse compose to ~10 usable facts.
fn orthogonalise(target: &Array1<f32>, suppress: &[&Array1<f32>]) -> Array1<f32> {
    // Step 1: build an orthonormal basis of span(suppress) via
    // Gram-Schmidt over the suppress set itself. Numerical
    // near-dependencies are dropped (||q|| < 1e-6 after projection).
    let mut basis: Vec<Array1<f32>> = Vec::with_capacity(suppress.len());
    for u in suppress {
        let mut q = (*u).clone();
        for b in &basis {
            let coef = q.dot(b);
            q = &q - &(coef * b);
        }
        let qn = q.dot(&q).sqrt();
        if qn > 1e-6 {
            q.mapv_inplace(|v| v / qn);
            basis.push(q);
        }
    }

    // Step 2: project target onto the orthogonal complement of
    // span(suppress) = span(basis).
    let mut v = target.clone();
    for q in &basis {
        let coef = v.dot(q);
        v = &v - &(coef * q);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn vec(xs: &[f32]) -> Array1<f32> {
        Array1::from_vec(xs.to_vec())
    }

    #[test]
    fn empty_input_returns_empty_result() {
        let r = refine_gates(&[], &[]);
        assert!(r.gates.is_empty());
        assert_eq!(r.median_retained, 1.0);
        assert_eq!(r.n_decoys, 0);
    }

    #[test]
    fn orthogonal_inputs_retain_full_norm() {
        // Two unit vectors that are already orthogonal — refine should
        // not change them and retained_norm should be ~1.0 for both.
        let inputs = vec![
            RefineInput { layer: 0, feature: 0, gate: vec(&[1.0, 0.0, 0.0]) },
            RefineInput { layer: 0, feature: 1, gate: vec(&[0.0, 1.0, 0.0]) },
        ];
        let r = refine_gates(&inputs, &[]);
        assert!((r.gates[0].retained_norm - 1.0).abs() < 1e-5);
        assert!((r.gates[1].retained_norm - 1.0).abs() < 1e-5);
        assert!((r.gates[0].gate[0] - 1.0).abs() < 1e-5);
        assert!((r.gates[1].gate[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parallel_inputs_lose_norm() {
        // Two parallel vectors — the second one should be projected to
        // (almost) zero after refining against the first.
        let inputs = vec![
            RefineInput { layer: 0, feature: 0, gate: vec(&[1.0, 0.0]) },
            RefineInput { layer: 0, feature: 1, gate: vec(&[2.0, 0.0]) },
        ];
        let r = refine_gates(&inputs, &[]);
        // The first fact projects out the second, and vice-versa. Both
        // collapse because they share the same direction and there is
        // nothing else to anchor on.
        assert!(r.gates[0].retained_norm < 0.01,
                "first fact retained norm {} should be ~0", r.gates[0].retained_norm);
        assert!(r.gates[1].retained_norm < 0.01);
    }

    #[test]
    fn overlapping_peers_lose_norm_under_refine() {
        // Two facts that share a direction should both retain less than
        // their full norm after refining against each other. We assert
        // norm retention rather than post-refine cosine because
        // symmetric Gram-Schmidt over a 2-element set preserves |cos|
        // by construction (both facts get reflected through the
        // orthogonal complement, flipping sign but not magnitude). The
        // load-bearing property is that the projection happened — i.e.
        // retained_norm < 1.0 — and that's what the executor uses for
        // its alpha-effective accounting.
        let inputs = vec![
            RefineInput { layer: 0, feature: 0, gate: vec(&[1.0, 0.5, 0.0, 0.0]) },
            RefineInput { layer: 0, feature: 1, gate: vec(&[0.5, 1.0, 0.0, 0.0]) },
        ];
        let r = refine_gates(&inputs, &[]);
        assert!(r.gates[0].retained_norm < 1.0,
                "fact 0 should lose norm to peer projection, got {}",
                r.gates[0].retained_norm);
        assert!(r.gates[1].retained_norm < 1.0);
        assert!(r.gates[0].retained_norm > 0.1,
                "fact 0 collapsed too far ({}), peers aren't parallel",
                r.gates[0].retained_norm);
    }

    #[test]
    fn decoy_residuals_remove_decoy_overlap() {
        // A single fact with overlap onto a decoy direction should
        // lose that overlap after refining against the decoy.
        let inputs = vec![
            RefineInput { layer: 0, feature: 0, gate: vec(&[1.0, 0.5]) },
        ];
        let decoy = vec(&[0.0, 1.0]);
        let cos_before = cos(&inputs[0].gate, &decoy);
        let r = refine_gates(&inputs, std::slice::from_ref(&decoy));
        let cos_after = cos(&r.gates[0].gate, &decoy);
        assert!(cos_after.abs() < 1e-5,
                "decoy overlap should drop to ~0, got {}", cos_after.abs());
        assert!(cos_before.abs() > 0.1, "test setup broken: no overlap to start");
    }

    #[test]
    fn cross_layer_facts_dont_interfere() {
        // Two facts at different layers that share a direction should
        // both retain their full norm — refine never crosses layers.
        let inputs = vec![
            RefineInput { layer: 5, feature: 0, gate: vec(&[1.0, 0.0]) },
            RefineInput { layer: 9, feature: 1, gate: vec(&[1.0, 0.0]) },
        ];
        let r = refine_gates(&inputs, &[]);
        assert!((r.gates[0].retained_norm - 1.0).abs() < 1e-5);
        assert!((r.gates[1].retained_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn summary_stats_are_sensible() {
        let inputs = vec![
            RefineInput { layer: 0, feature: 0, gate: vec(&[1.0, 0.0, 0.0]) },
            RefineInput { layer: 0, feature: 1, gate: vec(&[0.5, 0.5, 0.0]) },
            RefineInput { layer: 0, feature: 2, gate: vec(&[0.1, 0.1, 1.0]) },
        ];
        let r = refine_gates(&inputs, &[]);
        assert_eq!(r.gates.len(), 3);
        assert!(r.min_retained <= r.median_retained);
        assert!(r.median_retained <= r.max_retained);
        assert!(r.max_retained <= 1.0 + 1e-5);
    }

    #[test]
    fn passthrough_for_array_input_form() {
        // Smoke test that the API accepts the actual ndarray macros
        // (catches signature drift).
        let g = array![1.0_f32, 2.0, 3.0];
        let inputs = vec![RefineInput { layer: 0, feature: 0, gate: g.clone() }];
        let r = refine_gates(&inputs, &[]);
        assert_eq!(r.gates[0].gate, g);
    }

    fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let na = a.dot(a).sqrt();
        let nb = b.dot(b).sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        a.dot(b) / (na * nb)
    }
}
