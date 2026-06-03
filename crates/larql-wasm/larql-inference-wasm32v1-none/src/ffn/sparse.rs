//! Sparse FFN backend — gate matmul selects top-K features, architecture-correct.
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;

#[cfg(not(target_arch = "wasm32"))]
use ndarray::Array2;

#[cfg(not(target_arch = "wasm32"))]
use super::sparse_compute::{select_top_k_features, sparse_ffn_forward};
use super::FfnBackend;
#[cfg(not(target_arch = "wasm32"))]
use crate::model::ModelWeights;
#[cfg(not(target_arch = "wasm32"))]
/// Sparse FFN: compute all gate activations, select top-K, then
/// compute gate/up/down for those K features only.
///
/// Uses the model architecture trait for activation function and bias.
/// Falls back to dense BLAS when K >= 80% of features.
pub struct SparseFfn<'a> {
    pub weights: &'a ModelWeights,
    pub top_k: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> FfnBackend for SparseFfn<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = x.shape()[0];

        // Select features per position, union them, then compute once
        let mut all_features = alloc::collections::BTreeSet::new();
        for s in 0..seq_len {
            let x_row = x.row(s);
            let feats = select_top_k_features(self.weights, layer, &x_row, self.top_k);
            all_features.extend(feats);
        }
        let features: Vec<usize> = all_features.into_iter().collect();

        sparse_ffn_forward(self.weights, layer, x, &features)
    }

    fn name(&self) -> &str {
        "sparse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    #[cfg(not(target_arch = "wasm32"))]
    use ndarray::Array2;
    #[cfg(not(target_arch = "wasm32"))]
    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    #[test]
    fn sparse_ffn_name() {
        let weights = make_test_weights();
        let ffn = SparseFfn {
            weights: &weights,
            top_k: 4,
        };
        assert_eq!(ffn.name(), "sparse");
    }

    #[test]
    fn sparse_ffn_forward_shape_single_token() {
        let weights = make_test_weights();
        let ffn = SparseFfn {
            weights: &weights,
            top_k: 4,
        };
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_ffn_forward_shape_multi_token() {
        let weights = make_test_weights();
        let ffn = SparseFfn {
            weights: &weights,
            top_k: 4,
        };
        let x = input(3, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_ffn_forward_all_layers() {
        let weights = make_test_weights();
        let ffn = SparseFfn {
            weights: &weights,
            top_k: 8,
        };
        let x = input(1, weights.hidden_size);
        for layer in 0..weights.num_layers {
            let out = ffn.forward(layer, &x);
            assert_eq!(out.shape(), &[1, weights.hidden_size], "layer {layer}");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "layer {layer} non-finite"
            );
        }
    }

    #[test]
    fn sparse_ffn_with_activation_returns_correct_shapes() {
        let weights = make_test_weights();
        let ffn = SparseFfn {
            weights: &weights,
            top_k: 4,
        };
        let x = input(2, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &x);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert_eq!(act.shape()[0], 2);
    }

    #[test]
    fn sparse_ffn_top_k_gt_intermediate_falls_back_to_dense() {
        let weights = make_test_weights();
        // top_k > intermediate triggers dense fallback in sparse_ffn_forward
        let ffn_big = SparseFfn {
            weights: &weights,
            top_k: weights.intermediate_size + 100,
        };
        let ffn_dense = crate::ffn::weight::WeightFfn { weights: &weights };
        let x = input(1, weights.hidden_size);
        let out_sparse = ffn_big.forward(0, &x);
        let out_dense = ffn_dense.forward(0, &x);
        // With all features selected, results match dense
        for (s, d) in out_sparse.iter().zip(out_dense.iter()) {
            assert!((s - d).abs() < 1e-3, "big-k sparse vs dense: {s} != {d}");
        }
    }
}
