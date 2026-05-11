//! Exact walk — gate + up from model (safetensors) weights, down from
//! mmap'd feature-major matrix.
//!
//! The fallback when the vindex has `down_features.bin` but no
//! interleaved layout, and we still have the dense f32 weights loaded
//! (e.g. during a one-off correctness sanity check). Same FLOP count
//! as dense; reads 315 MB per layer. The one advantage is that the
//! down read is mmap-backed, so a hot layer's down matrix can stay
//! resident across calls without reloading safetensors shards.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_exact(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;

        // If FFN weights were dropped (walk-only mode), fall through to full mmap.
        let w_up = match self.weights.tensors.get(&arch.ffn_up_key(layer)) {
            Some(w) => w,
            None => {
                if let Some(result) = self.walk_ffn_full_mmap(layer, x) {
                    return result;
                }
                panic!("walk_ffn_exact: no FFN weights and no mmap data for layer {layer}");
            }
        };

        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let activation = if is_gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = crate::forward::dot_proj(x, w_gate);
            let up = crate::forward::dot_proj(x, w_up);
            if use_gelu {
                crate::ffn::gelu_tanh_gate_up(&gate, &up)
            } else {
                crate::ffn::silu_gate_up(&gate, &up)
            }
        } else {
            let mut proj = crate::forward::dot_proj(x, w_up);
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk))
            {
                crate::forward::add_bias(&mut proj, bias);
            }
            if use_gelu {
                proj.mapv(crate::ffn::gelu_tanh)
            } else {
                proj.mapv(|v| v * crate::ffn::sigmoid(v))
            }
        };

        let out = if let Some(down_view) = self.index.down_layer_matrix(layer) {
            larql_compute::matmul_gpu(&activation, &down_view, self.backend)
        } else {
            let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
            larql_compute::dot_proj_gpu(&activation, w_down, self.backend)
        };

        let mut out = out;
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "exact");
        (out, activation)
    }
}
