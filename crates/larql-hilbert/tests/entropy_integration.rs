//! End-to-end: entanglement_entropy orders matrices by compressibility —
//! rank-1 (0 ebits, fully compressible) < skewed spectrum < flat spectrum
//! (maximal, incompressible) — through the public crate API.

use larql_hilbert::{entanglement_entropy, spectral_entropy};
use ndarray::{array, Array2};

#[test]
fn entropy_orders_matrices_by_compressibility() {
    // Rank-1: zero entanglement (one Schmidt term).
    let rank1 = array![[1.0, 2.0], [2.0, 4.0]];
    // Skewed singular values [1, 0.1] → squared [1, 0.01]: low but nonzero.
    let skewed = Array2::from_diag(&array![1.0, 0.1]);
    // Flat: identity, maximal entropy for 2×2 (1 ebit).
    let flat = Array2::<f64>::eye(2);

    let s_rank1 = entanglement_entropy(&rank1);
    let s_skewed = entanglement_entropy(&skewed);
    let s_flat = entanglement_entropy(&flat);

    assert!(s_rank1 < 1e-9, "rank-1 should be ~0, got {s_rank1}");
    assert!(s_rank1 < s_skewed, "{s_rank1} !< {s_skewed}");
    assert!(s_skewed < s_flat, "{s_skewed} !< {s_flat}");
    assert!((s_flat - 1.0).abs() < 1e-9, "flat 2×2 should be 1 ebit, got {s_flat}");
}

#[test]
fn matrix_meter_agrees_with_spectral_entropy_on_squared_singular_values() {
    // The matrix meter must equal spectral_entropy applied to the squared
    // singular values directly. For a diagonal matrix those are the squared
    // diagonal entries.
    let m = Array2::from_diag(&array![2.0, 1.0]);
    let direct = spectral_entropy(&[4.0, 1.0]); // squares of 2 and 1
    assert!((entanglement_entropy(&m) - direct).abs() < 1e-9);
}
