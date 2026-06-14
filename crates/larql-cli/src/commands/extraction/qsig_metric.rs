//! Canonical (whitened) metric for the quantum-signature pipe. The dagger is a
//! choice; raw Euclidean vs canonical-whitened (Cholesky of the embedding
//! covariance, canonical_meta) is the metric dual. M = L⁻ᵀ applied to the hidden
//! axis: C_canon = (W_Q·M)(W_K·M)ᵀ.

use larql_vindex::ndarray::Array2;

/// Apply the whitening M = L⁻ᵀ to the hidden (column) axis of a weight matrix:
/// returns W·M, i.e. each row r of W solved against Lᵀ. `l` is lower-triangular.
fn whiten_rows(w: &Array2<f64>, l: &Array2<f64>) -> Array2<f64> {
    let (rows, d) = (w.shape()[0], w.shape()[1]);
    // Want (W·M)[r,:] = w_row · L⁻ᵀ. As a column: out_rowᵀ = L⁻¹ w_rowᵀ, i.e.
    // solve L · z = w_rowᵀ — forward substitution (L lower-triangular) — and store z.
    let mut out = Array2::<f64>::zeros((rows, d));
    for r in 0..rows {
        let mut z = vec![0.0; d];
        for i in 0..d {
            let mut acc = w[[r, i]];
            for j in 0..i {
                acc -= l[[i, j]] * z[j];
            }
            z[i] = acc / l[[i, i]];
        }
        for i in 0..d {
            out[[r, i]] = z[i];
        }
    }
    out
}

/// Canonical (metric-corrected) head coupling: C_canon = (W_Q M)(W_K M)ᵀ, M=L⁻ᵀ.
pub fn canonical_coupling(wq: &Array2<f64>, wk: &Array2<f64>, l: &Array2<f64>) -> Array2<f64> {
    let wqm = whiten_rows(wq, l);
    let wkm = whiten_rows(wk, l);
    wqm.dot(&wkm.t())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_cholesky_is_the_raw_coupling() {
        // L = I ⇒ canonical coupling == raw coupling (the metric dual collapses).
        let wq = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let wk = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let l = Array2::<f64>::eye(2);
        let canon = canonical_coupling(&wq, &wk, &l);
        let raw = wq.dot(&wk.t());
        for (a, b) in canon.iter().zip(raw.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn canonical_coupling_of_identity_is_inverse_gram_exact() {
        // canonical_coupling(I, I, L) = M Mᵀ = L⁻ᵀL⁻¹ = (L Lᵀ)⁻¹ = Σ⁻¹ (exact gate
        // — catches the L⁻¹-vs-L⁻ᵀ transpose bug the ≠-raw test cannot).
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap(); // lower-tri
        let i2 = Array2::<f64>::eye(2);
        let canon = canonical_coupling(&i2, &i2, &l);
        // Σ = L Lᵀ = [[4,2],[2,10]] ⇒ Σ⁻¹ = (1/36)[[10,−2],[−2,4]].
        let expected = Array2::from_shape_vec((2, 2), vec![10.0 / 36.0, -2.0 / 36.0, -2.0 / 36.0, 4.0 / 36.0]).unwrap();
        for (a, b) in canon.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "canon {a} vs Σ⁻¹ {b}");
        }
        // And it genuinely differs from the raw coupling (= I here).
        assert!((canon[[0, 1]]).abs() > 1e-6);
    }
}
