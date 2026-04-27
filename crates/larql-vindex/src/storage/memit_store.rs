//! L2 storage: MEMIT-compacted facts with decomposed (k, d) pairs for graph walk.
//!
//! Also hosts `memit_solve` — the vanilla closed-form decomposition (no
//! covariance whitening) used to populate `MemitStore` during COMPACT MAJOR.
//! The underlying ridge-system solve is `larql_compute::cpu::ops::linalg::
//! ridge_decomposition_solve`; this module wraps it with the MEMIT-domain
//! interpretation (keys = END residuals, targets = embed nudges, per-fact
//! reconstruction quality).
//!
//! For production weight edits with covariance whitening + per-fact
//! optimised target deltas (the validated v11 200/200 pipeline), see
//! `larql-inference/src/forward/memit.rs`.

use ndarray::{Array1, Array2};

use larql_compute::cpu::ops::linalg::ridge_decomposition_solve;

/// A single MEMIT compaction cycle's result.
#[derive(Debug, Clone)]
pub struct MemitCycle {
    pub cycle_id: u64,
    pub layer: usize,
    pub facts: Vec<MemitFact>,
    pub frobenius_norm: f32,
    pub min_reconstruction_cos: f32,
    pub max_off_diagonal: f32,
}

/// A fact stored in L2 via MEMIT decomposition.
#[derive(Debug, Clone)]
pub struct MemitFact {
    pub entity: String,
    pub relation: String,
    pub target: String,
    /// Decomposed key: the END-position residual at install layer.
    pub key: Array1<f32>,
    /// Decomposed contribution: ΔW · k_i.
    pub decomposed_down: Array1<f32>,
    /// Reconstruction quality: cos(decomposed_down, target_direction).
    pub reconstruction_cos: f32,
}

/// Persistent store for MEMIT-compacted facts across multiple cycles.
#[derive(Debug, Default)]
pub struct MemitStore {
    cycles: Vec<MemitCycle>,
    next_cycle_id: u64,
}

impl MemitStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_cycle(&mut self, layer: usize, facts: Vec<MemitFact>, frobenius_norm: f32, min_cos: f32, max_off_diag: f32) -> u64 {
        let id = self.next_cycle_id;
        self.next_cycle_id += 1;
        self.cycles.push(MemitCycle {
            cycle_id: id,
            layer,
            facts,
            frobenius_norm,
            min_reconstruction_cos: min_cos,
            max_off_diagonal: max_off_diag,
        });
        id
    }

    pub fn total_facts(&self) -> usize {
        self.cycles.iter().map(|c| c.facts.len()).sum()
    }

    pub fn num_cycles(&self) -> usize {
        self.cycles.len()
    }

    pub fn cycles(&self) -> &[MemitCycle] {
        &self.cycles
    }

    /// Lookup all facts for an entity across all cycles.
    pub fn facts_for_entity(&self, entity: &str) -> Vec<&MemitFact> {
        let mut out = Vec::new();
        for cycle in &self.cycles {
            for fact in &cycle.facts {
                if fact.entity.eq_ignore_ascii_case(entity) {
                    out.push(fact);
                }
            }
        }
        out
    }

    /// Lookup all facts matching (entity, relation) across all cycles.
    pub fn lookup(&self, entity: &str, relation: &str) -> Vec<&MemitFact> {
        let mut out = Vec::new();
        for cycle in &self.cycles {
            for fact in &cycle.facts {
                if fact.entity.eq_ignore_ascii_case(entity) && fact.relation.eq_ignore_ascii_case(relation) {
                    out.push(fact);
                }
            }
        }
        out
    }
}

/// Result of a vanilla MEMIT solve — the dense weight delta plus
/// per-fact decomposition diagnostics ready to feed `MemitStore`.
#[derive(Debug, Clone)]
pub struct MemitSolveResult {
    /// ΔW: (d, d) weight update matrix.
    pub delta_w: Array2<f32>,
    /// Per-fact decomposed contributions: d_i = ΔW @ k_i.
    pub decomposed: Vec<Array1<f32>>,
    /// Per-fact reconstruction cosine: cos(d_i, t_i).
    pub reconstruction_cos: Vec<f32>,
    /// Maximum off-diagonal cosine (cross-fact interference).
    pub max_off_diagonal: f32,
    /// Frobenius norm of ΔW.
    pub frobenius_norm: f32,
}

/// Vanilla MEMIT closed-form solve.
///
/// Wraps `larql_compute::cpu::ops::linalg::ridge_decomposition_solve`
/// with the MEMIT interpretation: each row of `keys` is the END-position
/// residual at the install layer, each row of `targets` is the desired
/// residual delta, and the per-fact decomposition `ΔW @ k_i` is what
/// gets persisted as a `(key, decomposed_down)` pair in `MemitStore`.
///
/// **Vanilla** = no covariance whitening. Cross-template bleed grows
/// with N when keys share a dominant direction. For production weight
/// edits with C⁻¹ whitening, use `larql-inference::forward::memit`.
pub fn memit_solve(
    keys: &Array2<f32>,
    targets: &Array2<f32>,
    lambda: f32,
) -> Result<MemitSolveResult, String> {
    let n = keys.nrows();
    let delta_w = ridge_decomposition_solve(keys, targets, lambda)
        .map_err(|e| format!("memit_solve: {e}"))?;

    // Batched per-fact decomposition: D = K @ ΔW^T  → (N, d), where
    // row i is `ΔW @ k_i` (the i-th fact's contribution). One BLAS sgemm
    // beats N hand-rolled matvecs by ~5-10× at d=2560.
    let d_matrix: Array2<f32> = keys.dot(&delta_w.t());

    let decomposed: Vec<Array1<f32>> = (0..n).map(|i| d_matrix.row(i).to_owned()).collect();

    let reconstruction_cos: Vec<f32> = (0..n)
        .map(|i| cosine_sim_views(&d_matrix.row(i), &targets.row(i)))
        .collect();

    let max_off_diagonal = max_off_diagonal_batched(&d_matrix, targets);
    let frobenius_norm = delta_w.iter().map(|x| x * x).sum::<f32>().sqrt();

    Ok(MemitSolveResult {
        delta_w,
        decomposed,
        reconstruction_cos,
        max_off_diagonal,
        frobenius_norm,
    })
}

fn cosine_sim_views(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    let dot = a.dot(b);
    let na = a.dot(a).sqrt();
    let nb = b.dot(b).sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Batched cross-similarity: `C[i,j] = cos(D.row(i), T.row(j))`. The
/// matrix is computed as one BLAS sgemm over row-normalised D and T,
/// then the max absolute off-diagonal value is returned. Replaces the
/// O(N² d) per-pair cosine loop with one (N, d) × (d, N) matmul.
fn max_off_diagonal_batched(d_matrix: &Array2<f32>, targets: &Array2<f32>) -> f32 {
    let n = d_matrix.nrows();
    if n < 2 {
        return 0.0;
    }
    let d_dim = d_matrix.ncols();
    debug_assert_eq!(targets.ncols(), d_dim);

    let normalise_rows = |m: &Array2<f32>| -> Array2<f32> {
        let mut out = m.clone();
        for i in 0..n {
            let row = out.row(i);
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                out.row_mut(i).mapv_inplace(|v| v * inv);
            }
        }
        out
    };

    let d_n = normalise_rows(d_matrix);
    let t_n = normalise_rows(targets);
    let c = d_n.dot(&t_n.t()); // (N, N) cross-cosine matrix

    let mut max_off = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = c[[i, j]].abs();
            if v > max_off {
                max_off = v;
            }
        }
    }
    max_off
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fact(entity: &str, relation: &str, target: &str) -> MemitFact {
        MemitFact {
            entity: entity.into(),
            relation: relation.into(),
            target: target.into(),
            key: Array1::zeros(4),
            decomposed_down: Array1::zeros(4),
            reconstruction_cos: 1.0,
        }
    }

    #[test]
    fn empty_store() {
        let s = MemitStore::new();
        assert_eq!(s.total_facts(), 0);
        assert_eq!(s.num_cycles(), 0);
    }

    #[test]
    fn add_cycle_and_lookup() {
        let mut s = MemitStore::new();
        let facts = vec![
            make_fact("France", "capital", "Paris"),
            make_fact("Germany", "capital", "Berlin"),
        ];
        let id = s.add_cycle(33, facts, 0.01, 0.99, 0.001);
        assert_eq!(id, 0);
        assert_eq!(s.total_facts(), 2);
        assert_eq!(s.num_cycles(), 1);

        let france = s.lookup("France", "capital");
        assert_eq!(france.len(), 1);
        assert_eq!(france[0].target, "Paris");

        let all_france = s.facts_for_entity("france");
        assert_eq!(all_france.len(), 1);
    }

    #[test]
    fn multi_cycle() {
        let mut s = MemitStore::new();
        s.add_cycle(33, vec![make_fact("France", "capital", "Paris")], 0.01, 0.99, 0.001);
        s.add_cycle(33, vec![make_fact("France", "language", "French")], 0.01, 0.99, 0.001);
        assert_eq!(s.total_facts(), 2);
        assert_eq!(s.num_cycles(), 2);

        let all = s.facts_for_entity("France");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn memit_solve_orthonormal_round_trip() {
        let n = 4;
        let d = 8;
        let mut keys = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            keys[[i, i]] = 1.0;
        }
        let mut targets = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            targets[[i, (i + n) % d]] = 1.0;
        }
        let r = memit_solve(&keys, &targets, 1e-6).unwrap();
        for cos in &r.reconstruction_cos {
            assert!(*cos > 0.99, "cos {cos}");
        }
        assert!(r.max_off_diagonal < 0.01, "off-diag {}", r.max_off_diagonal);
    }

    #[test]
    fn memit_solve_populates_diagnostics() {
        let n = 3;
        let d = 6;
        let mut keys = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            keys[[i, i]] = 1.0;
        }
        let mut targets = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            targets[[i, (i + 3) % d]] = 1.0;
        }
        let r = memit_solve(&keys, &targets, 1e-6).unwrap();

        assert_eq!(r.delta_w.shape(), [d, d]);
        assert_eq!(r.decomposed.len(), n);
        assert_eq!(r.reconstruction_cos.len(), n);
        for d_i in &r.decomposed {
            assert_eq!(d_i.len(), d);
        }
        assert!(r.frobenius_norm > 0.0);
        // ΔW @ k_i should match decomposed[i] exactly (within f32 noise).
        for i in 0..n {
            let direct = r.delta_w.dot(&keys.row(i));
            let diff: f32 = direct
                .iter()
                .zip(r.decomposed[i].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(diff < 1e-4, "fact {i}: diff {diff}");
        }
    }

    #[test]
    fn memit_solve_feeds_store() {
        // Round-trip: solve, package into MemitFact, add to MemitStore, look up.
        let n = 2;
        let d = 4;
        let mut keys = Array2::<f32>::zeros((n, d));
        keys[[0, 0]] = 1.0;
        keys[[1, 1]] = 1.0;
        let mut targets = Array2::<f32>::zeros((n, d));
        targets[[0, 2]] = 1.0;
        targets[[1, 3]] = 1.0;
        let r = memit_solve(&keys, &targets, 1e-6).unwrap();

        let labels = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
        ];
        let facts: Vec<MemitFact> = labels
            .iter()
            .enumerate()
            .map(|(i, (e, rel, t))| MemitFact {
                entity: (*e).into(),
                relation: (*rel).into(),
                target: (*t).into(),
                key: keys.row(i).to_owned(),
                decomposed_down: r.decomposed[i].clone(),
                reconstruction_cos: r.reconstruction_cos[i],
            })
            .collect();

        let mut store = MemitStore::new();
        store.add_cycle(
            33,
            facts,
            r.frobenius_norm,
            r.reconstruction_cos.iter().cloned().fold(1.0, f32::min),
            r.max_off_diagonal,
        );

        assert_eq!(store.total_facts(), 2);
        let france = store.lookup("France", "capital");
        assert_eq!(france.len(), 1);
        assert_eq!(france[0].target, "Paris");
        assert!(france[0].reconstruction_cos > 0.99);
    }
}
