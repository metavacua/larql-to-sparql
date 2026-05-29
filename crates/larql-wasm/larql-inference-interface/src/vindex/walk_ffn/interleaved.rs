//! f32 interleaved walk — gate + up + down in one contiguous mmap per
//! layer. Eliminates TLB thrash from 3 separate files and prefetches
//! the next layer.
//!
//! Three dense matmuls: gate_scores = x · W_gate.T, up_scores = x ·
//! W_up.T, out = silu(gate) * up · W_down.T. Identical computation to
//! dense, but all reads come from a single mmap region — the OS page
//! cache can keep a hot layer resident without filling descriptors.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_view = self.index.interleaved_gate(layer)?;
        let up_view = self.index.interleaved_up(layer)?;
        let down_view = self.index.interleaved_down(layer)?;

        self.index.prefetch_interleaved_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let gate_scores = larql_compute::dot_proj_gpu(x, &gate_view, self.backend);
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

        self.trace_path(layer, "interleaved");
        Some((out, activation))
    }
}
