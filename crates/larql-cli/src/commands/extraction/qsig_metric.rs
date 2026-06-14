//! Canonical (whitened) metric for the quantum-signature pipe. The dagger is a
//! choice; raw Euclidean vs canonical-whitened (Cholesky of the embedding
//! covariance, canonical_meta) is the metric dual. M = L⁻ᵀ applied to the hidden
//! axis: C_canon = (W_Q·M)(W_K·M)ᵀ.

use larql_vindex::ndarray::Array2;

/// Apply the whitening M = L⁻ᵀ to the hidden (column) axis of a weight matrix:
/// returns W·M, i.e. each row r of W solved against Lᵀ. `l` is lower-triangular.
#[allow(dead_code)]
fn whiten_rows(w: &Array2<f64>, l: &Array2<f64>) -> Array2<f64> {
    let (rows, d) = (w.shape()[0], w.shape()[1]);
    // M = L⁻ᵀ. (W·M)[r,:] solves Lᵀ·y = w_rowᵀ (Lᵀ upper-triangular ⇒ back-substitution).
    let mut out = Array2::<f64>::zeros((rows, d));
    for r in 0..rows {
        let mut y = vec![0.0; d];
        for i in (0..d).rev() {
            let mut acc = w[[r, i]];
            for j in (i + 1)..d {
                acc -= l[[j, i]] * y[j]; // (Lᵀ)[i,j] = l[j,i]
            }
            y[i] = acc / l[[i, i]];
        }
        for i in 0..d {
            out[[r, i]] = y[i];
        }
    }
    out
}

/// Canonical (metric-corrected) head coupling: C_canon = (W_Q M)(W_K M)ᵀ, M=L⁻ᵀ.
#[allow(dead_code)]
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
    fn whitening_changes_the_coupling_for_nontrivial_l() {
        let wq = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let wk = wq.clone();
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap(); // lower-tri
        let canon = canonical_coupling(&wq, &wk, &l);
        let raw = wq.dot(&wk.t());
        assert!(canon.iter().zip(raw.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
