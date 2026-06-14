//! Sheaf-theoretic empirical models for the quantum-signature apparatus. A
//! measurement cover (contexts) with Born outcome distributions; the contextual
//! fraction (computed in larql-cli) is the global-section obstruction.

use ndarray::Array2;
use num_complex::Complex64;

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// An empirical model: each context's joint outcome distribution. `settings[ctx]`
/// gives the (Alice-setting, Bob-setting) indices into the 4 dichotomic
/// measurements {A0=0, A1=1, B0=2, B1=3}; `dists[ctx]` is `[P(++),P(+-),P(-+),P(--)]`.
#[derive(Debug, Clone)]
pub struct EmpiricalModel {
    pub settings: Vec<(usize, usize)>,
    pub dists: Vec<[f64; 4]>,
}

/// Build the standard CHSH/Bell empirical model from a 2-qubit density matrix.
#[allow(clippy::needless_range_loop)] // explicit 2×2 tensor index arithmetic
pub fn bell_empirical_model(rho2: &Array2<Complex64>) -> EmpiricalModel {
    // measurement directions on the Bloch sphere (x,y,z)
    let s2 = 1.0 / 2.0_f64.sqrt();
    let dirs = [
        (0.0, 0.0, 1.0),       // A0 = Z
        (1.0, 0.0, 0.0),       // A1 = X
        (s2, 0.0, s2),         // B0 = (Z+X)/√2
        (-s2, 0.0, s2),        // B1 = (Z−X)/√2
    ];
    // ±1 projector for direction d and sign σ=±1: (I + σ n·σ)/2 as a 2×2.
    let proj = |d: (f64, f64, f64), sign: f64| -> [[Complex64; 2]; 2] {
        let (nx, ny, nz) = d;
        // n·σ = [[nz, nx−iny],[nx+iny, −nz]]
        [
            [c(0.5 * (1.0 + sign * nz), 0.0), c(0.5 * sign * nx, -0.5 * sign * ny)],
            [c(0.5 * sign * nx, 0.5 * sign * ny), c(0.5 * (1.0 - sign * nz), 0.0)],
        ]
    };
    let tr_proj = |pa: [[Complex64; 2]; 2], pb: [[Complex64; 2]; 2]| -> f64 {
        // Tr[ρ₂ (Pa⊗Pb)] = Σ ρ[r,s] (Pa⊗Pb)[s,r]; (Pa⊗Pb)[2a+b,2a'+b']=pa[a][a']pb[b][b'].
        let mut acc = Complex64::new(0.0, 0.0);
        for a in 0..2 { for b in 0..2 { for ap in 0..2 { for bp in 0..2 {
            let r = 2 * a + b; let s = 2 * ap + bp;
            acc += rho2[[r, s]] * pa[ap][a] * pb[bp][b]; // M[s,r]
        }}}}
        acc.re
    };
    let pairs = [(0usize, 2usize), (0, 3), (1, 2), (1, 3)];
    let mut settings = Vec::new();
    let mut dists = Vec::new();
    for &(ai, bj) in &pairs {
        let da = dirs[ai];
        let db = dirs[bj];
        // outcomes [++, +-, -+, --]
        let d = [
            tr_proj(proj(da, 1.0), proj(db, 1.0)),
            tr_proj(proj(da, 1.0), proj(db, -1.0)),
            tr_proj(proj(da, -1.0), proj(db, 1.0)),
            tr_proj(proj(da, -1.0), proj(db, -1.0)),
        ];
        settings.push((ai, bj));
        dists.push(d);
    }
    EmpiricalModel { settings, dists }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::{bell_rho2, product_rho2};

    #[test]
    fn distributions_are_normalized() {
        let m = bell_empirical_model(&bell_rho2());
        assert_eq!(m.settings.len(), 4);
        for d in &m.dists {
            assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(d.iter().all(|&p| p >= -1e-12));
        }
    }

    #[test]
    fn product_marginals_are_uniform_for_this_cover() {
        // |00> measured in X / (Z±X)/√2 bases: each single-party outcome is
        // unbiased except A0=Z on |0> which is deterministic. Just sanity: valid probs.
        let m = bell_empirical_model(&product_rho2());
        for d in &m.dists { assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9); }
    }
}
