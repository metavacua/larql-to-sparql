//! Sparse FFN backend — gate matmul selects top-K features, architecture-correct.

use ndarray::Array2;

use crate::model::ModelWeights;
use super::FfnBackend;
use super::sparse_compute::{sparse_ffn_forward, select_top_k_features};

/// Sparse FFN: compute all gate activations, select top-K, then
/// compute gate/up/down for those K features only.
///
/// Uses the model architecture trait for activation function and bias.
/// Falls back to dense BLAS when K >= 80% of features.
pub struct SparseFfn<'a> {
    pub weights: &'a ModelWeights,
    pub top_k: usize,
}

impl<'a> FfnBackend for SparseFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = x.shape()[0];

        // Select features per position, union them, then compute once
        let mut all_features = std::collections::BTreeSet::new();
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
