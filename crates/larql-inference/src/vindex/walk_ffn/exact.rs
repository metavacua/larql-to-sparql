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
        let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

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

#[cfg(test)]
mod tests {
    //! First tests for this path (2026-07-30 review, roadmap item 20).
    //! `exact` computes gate/up from the safetensors weights and down
    //! from the feature-major mmap — the same math as dense from mixed
    //! storage, so parity vs `WeightFfn` is the correctness bar.

    use crate::ffn::{FfnBackend, WeightFfn};
    use crate::test_utils::{
        attach_down_features_f32_only_to_test_vindex, attach_feature_major_f32_to_test_vindex,
        make_model_gate_vindex, make_starcoder2_test_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::WalkFfn;
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    fn assert_close(w: &Array2<f32>, d: &Array2<f32>, what: &str) {
        assert_eq!(w.shape(), d.shape());
        for (a, b) in w.iter().zip(d.iter()) {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "{what}: walk {a} vs dense {b}"
            );
        }
        assert!(d.iter().any(|v| v.abs() > 0.0), "{what}: trivial reference");
    }

    /// Ladder step 9: `down_features.bin` only (no up_features →
    /// `has_full_mmap_ffn` false) routes the forward to `exact`, whose
    /// output matches dense `WeightFfn` — gate/up identical by
    /// construction, down through the mmap'd transpose.
    #[test]
    fn exact_with_mmap_down_matches_dense_weight_ffn() {
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_down_features_f32_only_to_test_vindex(&weights, &mut index);
        assert!(index.has_down_features());
        assert!(
            !index.has_full_mmap_ffn(),
            "down-only fixture must not qualify for step 7 or the ladder \
             never reaches `exact`"
        );
        let walk = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();
        let dense = WeightFfn { weights: &weights };
        let input = x(2, weights.hidden_size);

        let (out_w, obs_w) = walk.forward_observed(0, &input);
        let act_w = obs_w
            .into_dense()
            .expect("dense walk path observes densely");
        let (out_d, obs_d) = dense.forward_observed(0, &input);
        let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
        assert_close(&act_w, &act_d, "activation");
        assert_close(&out_w, &out_d, "output");
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].path, "exact");
    }

    /// Without a down mmap, `exact` falls back to the safetensors down
    /// tensor — still dense parity (this is the direct-call shape; the
    /// ladder only reaches `exact` when `has_down_features()` is true).
    #[test]
    fn exact_without_mmap_down_uses_safetensors_down() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let walk = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();
        let dense = WeightFfn { weights: &weights };
        let input = x(1, weights.hidden_size);
        let (out_w, act_w) = walk.walk_ffn_exact(0, &input);
        let (out_d, obs_d) = dense.forward_observed(0, &input);
        let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
        assert_close(&act_w, &act_d, "activation");
        assert_close(&out_w, &out_d, "output");
        assert_eq!(walk.take_dispatch_trace()[0].path, "exact");
    }

    /// Non-gated arm (StarCoder2: Standard FFN, up + down biases):
    /// `exact` follows the same `up+bias → act → down+bias` pipeline as
    /// dense — parity again.
    #[test]
    fn exact_non_gated_starcoder2_matches_dense_with_biases() {
        let weights = make_starcoder2_test_weights();
        let index = make_test_vindex(&weights);
        assert!(
            weights.arch.ffn_up_bias_key(0).is_some()
                && weights.arch.ffn_down_bias_key(0).is_some(),
            "starcoder2 fixture must exercise both bias arms"
        );
        let walk = WalkFfn::new_unlimited(&weights, &index);
        let dense = WeightFfn { weights: &weights };
        let input = x(1, weights.hidden_size);
        let (out_w, act_w) = walk.walk_ffn_exact(0, &input);
        let (out_d, obs_d) = dense.forward_observed(0, &input);
        let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
        assert_close(&act_w, &act_d, "activation");
        assert_close(&out_w, &out_d, "output");
    }

    /// Walk-only mode (FFN weights dropped): `exact` must hand the layer
    /// to `full_mmap` when the up tensor is missing — and the result is
    /// still dense-parity because the full-mmap fixture serves the same
    /// (model) weights from mmap.
    #[test]
    fn exact_falls_back_to_full_mmap_when_ffn_weights_dropped() {
        let mut weights = make_test_weights();
        let mut index = make_model_gate_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        for layer in 0..weights.num_layers {
            let key = weights.arch.ffn_up_key(layer);
            assert!(
                weights.tensors.remove(&key).is_some(),
                "fixture must have had the up tensor to drop"
            );
        }
        // `make_test_weights` is deterministic — a fresh call reproduces
        // the dropped tensors for the dense reference.
        let reference_weights = make_test_weights();
        let dense = WeightFfn {
            weights: &reference_weights,
        };

        let walk = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();
        let input = x(1, weights.hidden_size);
        let (out_w, act_w) = walk.walk_ffn_exact(0, &input);
        assert_eq!(
            walk.take_dispatch_trace()[0].path,
            "full_mmap",
            "dropped FFN weights must divert to the full-mmap walk"
        );
        let (out_d, obs_d) = dense.forward_observed(0, &input);
        let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
        assert_close(&act_w, &act_d, "activation");
        assert_close(&out_w, &out_d, "output");
    }

    /// With neither FFN weights nor mmap data the layer is unservable —
    /// `exact` panics rather than fabricating output.
    #[test]
    #[should_panic(expected = "walk_ffn_exact: no FFN weights and no mmap data")]
    fn exact_panics_without_weights_or_mmap() {
        let mut weights = make_test_weights();
        let index = make_test_vindex(&weights);
        for layer in 0..weights.num_layers {
            weights.tensors.remove(&weights.arch.ffn_up_key(layer));
        }
        let walk = WalkFfn::new_unlimited(&weights, &index);
        let _ = walk.walk_ffn_exact(0, &x(1, weights.hidden_size));
    }
}
