//! Per-head "Hilbertian" residual: how close an attention head's query/key
//! coupling is to being complex-linear w.r.t. the split-half complex
//! structure J (J² = −I) that RoPE uses. See the plan doc for the math.

use ndarray::{s, Array2};

/// Build the split-half complex structure J on R^n (n must be even):
///   J e_i        =  e_{i+half}     for i in [0, half)
///   J e_{i+half} = −e_i
/// so that J·J = −I. Panics if n is odd.
pub fn complex_structure_split_half(n: usize) -> Array2<f64> {
    assert!(n.is_multiple_of(2), "complex structure requires even dimension, got {n}");
    let half = n / 2;
    let mut j = Array2::<f64>::zeros((n, n));
    for i in 0..half {
        j[[half + i, i]] = 1.0; // J e_i = e_{i+half}
        j[[i, half + i]] = -1.0; // J e_{i+half} = -e_i
    }
    j
}

fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Relative commutator residual ‖M J − J M‖_F / ‖M‖_F ∈ [0, 2].
/// 0 ⟺ M commutes with J ⟺ M is complex-linear w.r.t. J.
/// Returns 0.0 for the zero matrix (no division by zero).
pub fn commutator_residual(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let comm = &m.dot(j) - &j.dot(m);
    let den = frob_norm(m);
    if den == 0.0 {
        0.0
    } else {
        frob_norm(&comm) / den
    }
}

/// Map a query-head index to its KV-head index under grouped-query attention.
/// `num_q_heads` must be a multiple of `num_kv_heads`.
pub fn kv_head_for_query(query_head: usize, num_q_heads: usize, num_kv_heads: usize) -> usize {
    let group = num_q_heads / num_kv_heads;
    query_head / group.max(1)
}

/// Extract head `head`'s `[head_dim, hidden]` block from a stacked projection
/// matrix `[n*head_dim, hidden]` (PyTorch `[out, in]` orientation).
pub fn head_block(proj: &Array2<f64>, head: usize, head_dim: usize) -> Array2<f64> {
    proj.slice(s![head * head_dim..(head + 1) * head_dim, ..]).to_owned()
}

/// Per-head query/key coupling C = W_Q · W_Kᵀ, shape `[head_dim, head_dim]`.
/// `wq_head` and `wk_head` are both `[head_dim, hidden]`.
pub fn head_coupling(wq_head: &Array2<f64>, wk_head: &Array2<f64>) -> Array2<f64> {
    wq_head.dot(&wk_head.t())
}

/// Hilbertian residual for one head: ‖[C, J]‖_F / ‖C‖_F where C = W_Q W_Kᵀ.
/// `j` must be the split-half complex structure of dimension `head_dim`.
pub fn head_hilbertian_residual(
    wq_head: &Array2<f64>,
    wk_head: &Array2<f64>,
    j: &Array2<f64>,
) -> f64 {
    let c = head_coupling(wq_head, wk_head);
    commutator_residual(&c, j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn j_squares_to_negative_identity() {
        let j = complex_structure_split_half(4);
        let jj = j.dot(&j);
        let neg_i = -Array2::<f64>::eye(4);
        for i in 0..4 {
            for k in 0..4 {
                assert!((jj[[i, k]] - neg_i[[i, k]]).abs() < 1e-12,
                    "J^2 != -I at ({i},{k})");
            }
        }
    }

    #[test]
    fn realified_complex_matrix_has_zero_residual() {
        // M = [[A, -B], [B, A]] (2x2 blocks) commutes with split-half J on R^4.
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut m = Array2::<f64>::zeros((4, 4));
        m.slice_mut(s![0..2, 0..2]).assign(&a);
        m.slice_mut(s![0..2, 2..4]).assign(&(-&b));
        m.slice_mut(s![2..4, 0..2]).assign(&b);
        m.slice_mut(s![2..4, 2..4]).assign(&a);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn diagonal_matrix_has_positive_residual() {
        // diag(1,2,3,4) does not commute with J (it mixes paired coords).
        let m = Array2::from_diag(&array![1.0, 2.0, 3.0, 4.0]);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) > 0.1);
    }

    #[test]
    fn identity_has_zero_residual() {
        let m = Array2::<f64>::eye(4);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn zero_matrix_has_zero_residual_not_nan() {
        let m = Array2::<f64>::zeros((4, 4));
        let j = complex_structure_split_half(4);
        let r = commutator_residual(&m, &j);
        assert_eq!(r, 0.0);
    }

    #[test]
    #[should_panic]
    fn odd_dimension_panics() {
        let _ = complex_structure_split_half(3);
    }

    #[test]
    fn kv_head_for_query_maps_gqa_groups() {
        // 15 query heads, 5 kv heads -> group size 3.
        assert_eq!(kv_head_for_query(0, 15, 5), 0);
        assert_eq!(kv_head_for_query(2, 15, 5), 0);
        assert_eq!(kv_head_for_query(3, 15, 5), 1);
        assert_eq!(kv_head_for_query(14, 15, 5), 4);
        // No GQA (num_kv == num_q): identity.
        assert_eq!(kv_head_for_query(2, 4, 4), 2);
        // Single kv head: everything maps to 0.
        assert_eq!(kv_head_for_query(7, 8, 1), 0);
    }

    #[test]
    fn head_block_extracts_rows() {
        // proj is [4, 3] = 2 heads of head_dim 2.
        let proj = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let h0 = head_block(&proj, 0, 2);
        let h1 = head_block(&proj, 1, 2);
        assert_eq!(h0, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(h1, array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
    }

    #[test]
    fn head_coupling_is_wq_times_wk_transpose() {
        // wq, wk are [d_head=2, hidden=3]; C = wq · wkᵀ is [2,2].
        let wq = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let wk = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let c = head_coupling(&wq, &wk);
        // row0·wkᵀ = [1,4]; row1·wkᵀ = [2,5]
        assert_eq!(c, array![[1.0, 4.0], [2.0, 5.0]]);
    }

    #[test]
    fn head_hilbertian_residual_matches_manual_composition() {
        let wq = array![[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]];
        let wk = array![[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 3.0]];
        let j = complex_structure_split_half(2); // d_head = 2
        let direct = head_hilbertian_residual(&wq, &wk, &j);
        let manual = commutator_residual(&head_coupling(&wq, &wk), &j);
        assert!((direct - manual).abs() < 1e-15);
    }
}
