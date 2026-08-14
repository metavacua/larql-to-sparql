//! Layer normalization and residual stream operations.
//!
//! Re-exports from `larql_compute::residual` — Step 2e moved both the
//! leaf math AND the arch-aware `*_for_arch` wrappers down once
//! `forward_overrides` followed. This shim preserves `crate::residual::*`
//! paths.

pub use larql_compute::residual::{
    layer_norm, layer_norm_eps, layer_norm_for_arch, rms_norm, rms_norm_eps, rms_norm_for_arch,
    rms_norm_full_projection_eps, rms_norm_heads, rms_norm_heads_eps, rms_norm_heads_no_weight,
    rms_norm_heads_no_weight_eps, rms_norm_qk, rms_norm_qk_eps, rms_norm_qk_for_arch, DEFAULT_EPS,
};

#[cfg(test)]
mod tests {
    //! Arch-aware integration smoke tests stay here (not in compute) because
    //! they exercise the full chain: `arch.norm_eps()` parsing in
    //! `larql-models` → `forward_overrides::norm_eps_override` env-var
    //! resolution → leaf `*_eps` math. The compute-side `*_for_arch` unit
    //! tests cover the arch-and-eps wiring in isolation.

    use super::*;
    use ndarray::Array2;

    fn build_arch_with_eps(eps: f64) -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": eps,
        }))
    }

    #[test]
    fn rms_norm_for_arch_reads_arch_eps_through_inference_shim() {
        let x = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.001, 0.001, 0.001]).unwrap();
        let strict = build_arch_with_eps(1e-6);
        let loose = build_arch_with_eps(1e-5);
        let out_strict = rms_norm_for_arch(&x, None, 0.0, &*strict);
        let out_loose = rms_norm_for_arch(&x, None, 0.0, &*loose);
        let max_diff = out_strict
            .iter()
            .zip(out_loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 0.01,
            "shim re-export must reach compute's eps-aware path (max diff {max_diff})"
        );
    }
}
