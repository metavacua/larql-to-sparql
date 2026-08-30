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
        let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

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

#[cfg(test)]
mod tests {
    //! First tests for this path (2026-07-30 review, roadmap item 20).
    //! The f32-interleaved fixture stores the MODEL's own gate/up/down
    //! tensors (`attach_interleaved_f32_to_test_vindex` copies
    //! `weights.tensors`), so the walk computes the identical three
    //! matmuls as the dense baseline — parity vs `WeightFfn` is the
    //! correctness bar.

    use crate::ffn::{FfnBackend, WeightFfn};
    use crate::test_utils::{make_test_vindex, InterleavedF32TestFixtures};
    use crate::vindex::WalkFfn;
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Full forward through the ladder lands on `interleaved` (step 6)
    /// and matches the dense `WeightFfn` on the same weights — output
    /// AND activation, both layers.
    #[test]
    fn interleaved_matches_dense_weight_ffn() {
        let f = InterleavedF32TestFixtures::build();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index).with_dispatch_trace();
        let dense = WeightFfn {
            weights: &f.weights,
        };
        let input = x(2, f.weights.hidden_size);

        for layer in 0..f.weights.num_layers {
            let (out_w, obs_w) = walk.forward_observed(layer, &input);
            let act_w = obs_w
                .into_dense()
                .expect("dense walk path observes densely");
            let (out_d, obs_d) = dense.forward_observed(layer, &input);
            let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
            assert_eq!(out_w.shape(), out_d.shape());
            for (w, d) in act_w.iter().zip(act_d.iter()) {
                assert!(
                    (w - d).abs() <= 1e-5 * d.abs().max(1.0),
                    "layer {layer}: activation {w} vs dense {d}"
                );
            }
            for (w, d) in out_w.iter().zip(out_d.iter()) {
                assert!(
                    (w - d).abs() <= 1e-5 * d.abs().max(1.0),
                    "layer {layer}: output {w} vs dense {d}"
                );
            }
            assert!(out_d.iter().any(|v| v.abs() > 0.0), "non-trivial reference");
        }
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), f.weights.num_layers);
        assert!(
            trace.iter().all(|e| e.path == "interleaved"),
            "f32-interleaved storage must route every layer via step 6, got {trace:?}"
        );
    }

    /// Without interleaved storage the path declines (`?` early exits)
    /// so the ladder can fall through.
    #[test]
    fn interleaved_declines_without_interleaved_storage() {
        let f = InterleavedF32TestFixtures::build();
        let bare = make_test_vindex(&f.weights);
        let walk = WalkFfn::new_unlimited(&f.weights, &bare);
        assert!(walk
            .walk_ffn_interleaved(0, &x(1, f.weights.hidden_size))
            .is_none());
    }
}
