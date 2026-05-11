//! Full mmap walk — gate + up + down from three separate mmap files.
//! Zero safetensor reads. Three BLAS gemms over mmap'd matrices.
//!
//! Used by vindexes that have `up_features.bin` and `down_features.bin`
//! but not the interleaved layout. Same FLOP count as dense; the only
//! win is that all weight reads come from the vindex so the safetensors
//! can be unloaded after extraction.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_full_mmap(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_scores = self.index.gate_scores_batch(layer, x)?;
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "full_mmap");
        Some((out, activation))
    }
}
