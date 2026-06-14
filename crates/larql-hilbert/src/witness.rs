//! Exact structural quantum-signature witnesses on a 2-qubit reduced density
//! matrix ρ₂ (4×4 Hermitian, basis index 2·a+b). All deterministic — no
//! sampling. See the SP2 spec for W1–W8 and the randomness regime.

use ndarray::Array2;
use num_complex::Complex64;

use crate::complex_structure::{commutator_residual, split_half_j};
use crate::density::{density_matrix, partial_trace, von_neumann_entropy};
use crate::entropy::entanglement_entropy_bipartition;
use crate::nqubit::{row_qubits, NQubit};
use crate::register::classical_bits;
use crate::eig::hermitian_eigenvalues;
use crate::eig::symmetric_eigenvalues;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Bell |Φ⁺⟩⟨Φ⁺| as a 4×4 density matrix (the entangled/quantum pole).
pub fn bell_rho2() -> Array2<Complex64> {
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    for &(i, j) in &[(0, 0), (0, 3), (3, 0), (3, 3)] {
        rho[[i, j]] = c(0.5, 0.0);
    }
    rho
}

/// Product pole |00⟩⟨00| (separable, independent, local).
pub fn product_rho2() -> Array2<Complex64> {
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    rho[[0, 0]] = c(1.0, 0.0);
    rho
}

/// Werner state p|Φ⁺⟩⟨Φ⁺| + (1−p)/4 · I — entangled iff p>1/3, CHSH-violating
/// iff p>1/√2. The lattice cell `N>0, M≤1` lives at e.g. p=0.6.
pub fn werner_state(p: f64) -> Array2<Complex64> {
    let mut rho = bell_rho2().mapv(|z| z * c(p, 0.0));
    for i in 0..4 {
        rho[[i, i]] += c((1.0 - p) / 4.0, 0.0);
    }
    rho
}

/// W1 — mutual information I(A:B) = S(ρ_A) + S(ρ_B) − S(ρ₂), in bits. 0 ⟺ product.
pub fn mutual_information(rho2: &Array2<Complex64>) -> f64 {
    let s_a = von_neumann_entropy(&partial_trace(rho2, 2, &[0]));
    let s_b = von_neumann_entropy(&partial_trace(rho2, 2, &[1]));
    let s_ab = von_neumann_entropy(rho2);
    (s_a + s_b - s_ab).max(0.0)
}

/// Partial transpose over subsystem B: ρ^{T_B}[2a+b, 2a'+b'] = ρ[2a+b', 2a'+b].
fn partial_transpose_b(rho2: &Array2<Complex64>) -> Array2<Complex64> {
    let mut pt = Array2::<Complex64>::zeros((4, 4));
    for a in 0..2 {
        for ap in 0..2 {
            for b in 0..2 {
                for bp in 0..2 {
                    pt[[2 * a + b, 2 * ap + bp]] = rho2[[2 * a + bp, 2 * ap + b]];
                }
            }
        }
    }
    pt
}

/// W2 — negativity N(ρ₂) = Σ|negative eigenvalues of ρ^{T_B}|. For a 2-qubit
/// state, N=0 ⟺ separable (Peres–Horodecki, necessary & sufficient).
pub fn negativity(rho2: &Array2<Complex64>) -> f64 {
    hermitian_eigenvalues(&partial_transpose_b(rho2))
        .into_iter()
        .filter(|&l| l < 0.0)
        .map(|l| -l)
        .sum()
}

/// The three single-qubit Pauli matrices X, Y, Z (index 0,1,2).
fn pauli(i: usize) -> [[Complex64; 2]; 2] {
    match i {
        0 => [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]], // X
        1 => [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]], // Y
        _ => [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]], // Z
    }
}

/// Tr[ρ₂ · (σ_i ⊗ σ_j)] with σ at Pauli indices i, j. Real for Hermitian ρ.
#[allow(clippy::needless_range_loop)] // explicit 2×2 tensor index arithmetic
fn pauli_expectation(rho2: &Array2<Complex64>, i: usize, j: usize) -> f64 {
    let (si, sj) = (pauli(i), pauli(j));
    let mut acc = Complex64::new(0.0, 0.0);
    // (σ_i⊗σ_j)[2a+b, 2a'+b'] = si[a][a'] * sj[b][b']; Tr(ρ·M) = Σ ρ[r,s] M[s,r].
    for a in 0..2 {
        for b in 0..2 {
            for ap in 0..2 {
                for bp in 0..2 {
                    let r = 2 * a + b;
                    let s = 2 * ap + bp;
                    let m_sr = si[ap][a] * sj[bp][b]; // M[s, r]
                    acc += rho2[[r, s]] * m_sr;
                }
            }
        }
    }
    acc.re
}

/// W3 — maximal CHSH value via the Horodecki criterion: `2√M`, where `M` is the
/// sum of the two largest eigenvalues of `TᵀT`, `T_ij = Tr[ρ₂ σ_i⊗σ_j]`.
/// Violates the classical bound iff `M > 1` (Tsirelson caps `M ≤ 2` → CHSH ≤ 2√2).
pub fn chsh_max(rho2: &Array2<Complex64>) -> f64 {
    let mut t = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            t[[i, j]] = pauli_expectation(rho2, i, j);
        }
    }
    let tt = t.t().dot(&t);
    let mut ev = symmetric_eigenvalues(&tt);
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap()); // descending
    let m = ev[0].max(0.0) + ev[1].max(0.0);
    2.0 * m.sqrt()
}

const LATTICE_TOL: f64 = 1e-6;

/// Check the witness implication lattice from raw values: nonlocal ⟹ entangled
/// ⟹ correlated (and the dual product ⟹ separable ⟹ local). Returns false if
/// any implication is violated — which would indict the apparatus, not the data.
pub fn lattice_check(mutual_info: f64, neg: f64, chsh: f64) -> bool {
    let correlated = mutual_info > LATTICE_TOL;
    let entangled = neg > LATTICE_TOL;
    let nonlocal = chsh > 2.0 + LATTICE_TOL;
    if nonlocal && !entangled {
        return false; // nonlocal ⟹ entangled
    }
    if entangled && !correlated {
        return false; // entangled ⟹ correlated
    }
    true
}

/// Evaluate W1–W3 on ρ₂ and check the lattice.
pub fn lattice_consistent(rho2: &Array2<Complex64>) -> bool {
    lattice_check(mutual_information(rho2), negativity(rho2), chsh_max(rho2))
}

/// Two-qubit Pauli product σ_p ⊗ σ_q as a 4×4 matrix; p,q ∈ {0=I,1=X,2=Y,3=Z}.
fn pauli2(p: usize, q: usize) -> Array2<Complex64> {
    let s = |k: usize| -> [[Complex64; 2]; 2] {
        match k {
            0 => [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
            1 => [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
            2 => [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]],
            _ => [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]],
        }
    };
    let (a, b) = (s(p), s(q));
    let mut m = Array2::<Complex64>::zeros((4, 4));
    for i0 in 0..2 { for i1 in 0..2 { for j0 in 0..2 { for j1 in 0..2 {
        m[[2 * i0 + i1, 2 * j0 + j1]] = a[i0][j0] * b[i1][j1];
    }}}}
    m
}

/// The 9 Peres–Mermin observables (I=0,X=1,Y=2,Z=3), row-major 3×3:
///  X⊗I  I⊗X  X⊗X
///  I⊗Z  Z⊗I  Z⊗Z
///  X⊗Z  Z⊗X  Y⊗Y
fn pm_observables() -> [(usize, usize); 9] {
    [(1,0),(0,1),(1,1), (0,3),(3,0),(3,3), (1,3),(3,1),(2,2)]
}

/// The 6 contexts (3 rows then 3 columns) as index-triples into pm_observables().
fn pm_contexts() -> [[usize; 3]; 6] {
    [[0,1,2],[3,4,5],[6,7,8], [0,3,6],[1,4,7],[2,5,8]]
}

/// The sign s_c ∈ {+1,−1} such that the product of a context's observables = s_c·I.
/// Self-checks that the product really is proportional to identity.
fn pm_context_sign(ctx: &[usize; 3], obs: &[(usize, usize); 9]) -> f64 {
    let mut prod = Array2::<Complex64>::eye(4);
    for &k in ctx {
        let (p, q) = obs[k];
        prod = prod.dot(&pauli2(p, q));
    }
    let s = prod[[0, 0]].re;
    // assert prod == s·I (the PM algebra guarantees this).
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { s } else { 0.0 };
            debug_assert!((prod[[i, j]] - c(expected, 0.0)).norm() < 1e-9,
                "PM context product must be ±I");
        }
    }
    s.signum()
}

/// W8 quantum value: Σ_c χ_c·⟨∏ context⟩ with χ_c = s_c (the ±I sign). Each term
/// is s_c·s_c = +1 (state-independent) ⇒ total 6.
pub fn peres_mermin_quantum_value() -> f64 {
    let obs = pm_observables();
    pm_contexts().iter().map(|ctx| {
        let s = pm_context_sign(ctx, &obs);
        s * s // χ_c · ⟨B_c⟩ = +1
    }).sum()
}

/// W8 noncontextual bound: max over the 2⁹ deterministic ±1 value-assignments of
/// Σ_c χ_c · ∏_{k∈c} v(k). Kochen–Specker forces this to 4 < 6.
pub fn peres_mermin_noncontextual_bound() -> f64 {
    let obs = pm_observables();
    let contexts = pm_contexts();
    let signs: Vec<f64> = contexts.iter().map(|ctx| pm_context_sign(ctx, &obs)).collect();
    let mut best = f64::NEG_INFINITY;
    for assign in 0..(1u32 << 9) {
        let v = |k: usize| if (assign >> k) & 1 == 1 { 1.0 } else { -1.0 };
        let val: f64 = contexts.iter().zip(&signs)
            .map(|(ctx, &chi)| chi * v(ctx[0]) * v(ctx[1]) * v(ctx[2]))
            .sum();
        best = best.max(val);
    }
    best
}

/// The pre-registered 2-qubit reduction ρ₂ used by W1–W3 and the sheaf cover:
/// embed `C` via the Choi map (`from_matrix`), then partial-trace onto the
/// pre-registered pair {0, n/2} (one row qubit + one column qubit). Exposed so the
/// contextual-fraction cover uses the *identical* reduction as the scalar
/// witnesses (predicativity).
pub fn reduced_rho2(coupling: &Array2<f64>) -> Array2<Complex64> {
    let state = NQubit::from_matrix(coupling);
    let n = state.n();
    partial_trace(&density_matrix(&state), n, &[0, n / 2])
}

/// The structural witness battery (W1–W6) for one coupling matrix `C`.
#[derive(Debug, Clone, Copy)]
pub struct Witnesses {
    pub mutual_information: f64,   // W1
    pub negativity: f64,          // W2
    pub chsh: f64,                // W3
    pub entanglement_entropy: f64, // W4 (full-state row/col bipartition)
    pub gap: f64,                 // W5 (classical bits − W4)
    pub hilbertian_residual: f64, // W6
    pub lattice_ok: bool,
}

impl Witnesses {
    /// Evaluate W1–W6 on a square coupling `C`. State = `from_matrix(C)` (the Choi
    /// embedding); ρ₂ = partial trace onto the pre-registered pair {0, n/2} (one
    /// row qubit + one column qubit) for W1–W3; W4/W5 use the full row-vs-column
    /// bipartition; W6 is the Hilbertian residual on `C` directly.
    pub fn from_coupling(coupling: &Array2<f64>) -> Witnesses {
        assert_eq!(
            coupling.shape()[0],
            coupling.shape()[1],
            "coupling must be square (head_dim×head_dim)"
        );
        let rows = coupling.shape()[0];
        let state = NQubit::from_matrix(coupling);
        let rho2 = reduced_rho2(coupling);
        let mi = mutual_information(&rho2);
        let neg = negativity(&rho2);
        let chsh = chsh_max(&rho2);
        let ent = entanglement_entropy_bipartition(&state, &row_qubits(rows));
        let h = classical_bits(&state);
        let j = split_half_j(rows); // C is square head_dim×head_dim (even)
        let resid = commutator_residual(coupling, &j);
        Witnesses {
            mutual_information: mi,
            negativity: neg,
            chsh,
            entanglement_entropy: ent,
            gap: (h - ent).max(0.0),
            hilbertian_residual: resid,
            lattice_ok: lattice_check(mi, neg, chsh),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_rho2_matches_the_internal_reduction() {
        let c = Array2::from_shape_vec((4, 4), (0..16).map(|i| (i as f64 * 0.5).cos()).collect()).unwrap();
        let rho = reduced_rho2(&c);
        assert_eq!(rho.shape(), &[4, 4]);
        let tr: Complex64 = (0..4).map(|i| rho[[i, i]]).sum();
        assert!((tr.re - 1.0).abs() < 1e-9 && tr.im.abs() < 1e-9, "trace must be 1, got {tr}");
        // It is the SAME ρ₂ W1 uses inside from_coupling (predicativity).
        assert!((mutual_information(&rho) - Witnesses::from_coupling(&c).mutual_information).abs() < 1e-9);
    }

    #[test]
    fn peres_mermin_quantum_exceeds_noncontextual_bound() {
        // State-independent KS contextuality: quantum 6 > noncontextual 4.
        assert!((peres_mermin_quantum_value() - 6.0).abs() < 1e-9);
        assert!((peres_mermin_noncontextual_bound() - 4.0).abs() < 1e-9);
        assert!(peres_mermin_quantum_value() > peres_mermin_noncontextual_bound());
    }

    #[test]
    fn mutual_information_bell_is_two_bits() {
        assert!((mutual_information(&bell_rho2()) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn mutual_information_product_is_zero() {
        assert!(mutual_information(&product_rho2()).abs() < 1e-9);
    }

    #[test]
    fn negativity_bell_is_half() {
        assert!((negativity(&bell_rho2()) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn negativity_product_is_zero() {
        assert!(negativity(&product_rho2()).abs() < 1e-9);
    }

    #[test]
    fn negativity_werner_entangled_above_one_third() {
        assert!(negativity(&werner_state(0.6)) > 1e-6, "p=0.6 Werner is entangled");
        assert!(negativity(&werner_state(0.2)).abs() < 1e-9, "p=0.2 Werner is separable");
    }

    #[test]
    fn chsh_bell_saturates_tsirelson() {
        assert!((chsh_max(&bell_rho2()) - 2.0 * 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn chsh_product_does_not_violate() {
        assert!(chsh_max(&product_rho2()) <= 2.0 + 1e-9);
    }

    #[test]
    fn chsh_werner_entangled_but_local_cell() {
        // p=0.6: entangled (negativity>0) yet CHSH ≤ 2 — the Werner cell proving
        // negativity (W2) and CHSH (W3) are independent witnesses.
        assert!(negativity(&werner_state(0.6)) > 1e-6);
        assert!(chsh_max(&werner_state(0.6)) <= 2.0 + 1e-9);
    }

    #[test]
    fn lattice_holds_on_poles_and_werner() {
        for rho in [bell_rho2(), product_rho2(), werner_state(0.6), werner_state(0.2)] {
            assert!(lattice_consistent(&rho), "implication lattice must hold");
        }
    }

    #[test]
    fn lattice_detects_an_inconsistent_triple() {
        // nonlocal but separable is impossible → inconsistent (indicts the apparatus).
        assert!(!lattice_check(0.0 /*MI*/, 0.0 /*N*/, 2.83 /*CHSH>2*/));
        // a consistent triple (correlated, entangled, nonlocal) holds.
        assert!(lattice_check(2.0, 0.5, 2.83));
    }

    #[test]
    fn battery_on_identity_coupling_is_entangled_quantum() {
        use ndarray::array;
        // C = I₂ → Choi state = Bell → entangled, nonlocal, correlated.
        let cmat = array![[1.0, 0.0], [0.0, 1.0]];
        let w = Witnesses::from_coupling(&cmat);
        assert!(w.negativity > 1e-6);          // entangled
        assert!(w.chsh > 2.0);                  // nonlocal-structure
        assert!(w.mutual_information > 1e-6);   // correlated
        assert!(w.lattice_ok);
    }

    #[test]
    fn battery_on_rank_one_coupling_is_separable() {
        use ndarray::array;
        let cmat = array![[1.0, 2.0], [2.0, 4.0]]; // rank 1
        let w = Witnesses::from_coupling(&cmat);
        assert!(w.negativity.abs() < 1e-6);     // provably separable (conclusive)
        assert!(w.chsh <= 2.0 + 1e-6);
        assert!(w.lattice_ok);
    }

    #[test]
    fn random_coupling_is_generically_entangled_via_embedding() {
        use crate::entropy::entanglement_entropy_bipartition;
        use crate::nqubit::{row_qubits, NQubit};
        // Deterministic pseudo-random 4×4 "coupling" (no rand dep here).
        let mut vals = [0.0f64; 16];
        let (mut x, a, cc, m) = (12345u64, 1664525u64, 1013904223u64, 1u64 << 31);
        for v in vals.iter_mut() {
            x = (a.wrapping_mul(x).wrapping_add(cc)) % m;
            *v = x as f64 / m as f64 - 0.5;
        }
        let cmat = Array2::from_shape_vec((4, 4), vals.to_vec()).unwrap();
        let state = NQubit::from_matrix(&cmat); // 4 qubits (2 row + 2 col)
        // The Choi embedding manufactures entanglement in the ROW-vs-COLUMN cut
        // (the Schmidt entropy of C), NOT between the two row qubits — so we
        // assert the full bipartition entanglement. This is WHY the random null
        // is mandatory: a generic matrix is entangled purely via the embedding.
        let s = entanglement_entropy_bipartition(&state, &row_qubits(4));
        assert!(
            s > 1e-6,
            "generic coupling reads as entangled via the Choi embedding (S={s} ebits)"
        );
    }
}
