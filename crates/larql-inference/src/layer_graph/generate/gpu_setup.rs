//! Shared setup for GPU/vindex-backed generation paths.

use super::types::GenerateError;
use crate::layer_graph::pipeline_layer::{kv_cache_shapes_for_arch, DEFAULT_GPU_KV_CACHE_MAX_SEQ};
use crate::model::ModelWeights;
use larql_compute::backend::Capability;
use larql_compute::{prelude::*, FullPipelineLayer};
use std::ops::Range;

/// True when the model has at least two layers and any per-layer
/// attention parameter differs from layer 0. Catches Gemma 4 31B's
/// sliding/global geometry alternation, the canonical heterogeneous
/// case (50 sliding-attention layers at head_dim=256/16-kv plus 10
/// global-attention layers at head_dim=512/4-kv).
pub(crate) fn has_heterogeneous_attention(weights: &ModelWeights) -> bool {
    let arch = &*weights.arch;
    let n = weights.num_layers;
    if n < 2 {
        return false;
    }
    let l0_head = arch.head_dim_for_layer(0);
    let l0_kv = arch.num_kv_heads_for_layer(0);
    let l0_q = arch.num_q_heads_for_layer(0);
    let l0_rope = arch.rope_base_for_layer(0);
    let l0_rotary = arch.rotary_fraction_for_layer(0);
    let l0_sliding = arch.is_sliding_window_layer(0);
    (1..n).any(|l| {
        arch.head_dim_for_layer(l) != l0_head
            || arch.num_kv_heads_for_layer(l) != l0_kv
            || arch.num_q_heads_for_layer(l) != l0_q
            || arch.rope_base_for_layer(l) != l0_rope
            || arch.rotary_fraction_for_layer(l) != l0_rotary
            || arch.is_sliding_window_layer(l) != l0_sliding
    })
}

/// Reject heterogeneous models on backends that haven't opted in via
/// `Capability::HeterogeneousAttention`. The error fires *before* the
/// first decode dispatch so the caller sees a precise unsupported-backend
/// message rather than corrupted KV state at the first non-uniform layer.
pub(crate) fn ensure_attention_supported(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
) -> Result<(), GenerateError> {
    if has_heterogeneous_attention(weights) && !backend.supports(Capability::HeterogeneousAttention)
    {
        return Err(GenerateError::unsupported_backend(format!(
            "{} model has heterogeneous attention geometry but the active backend ({}) does not advertise Capability::HeterogeneousAttention",
            weights.arch.family(),
            backend.name(),
        )));
    }
    Ok(())
}

pub(super) struct GpuDecodeSetup<'a> {
    pub layers: Vec<FullPipelineLayer<'a>>,
    pub hidden: usize,
    pub intermediate: usize,
}

pub(super) fn build_gpu_decode_setup<'a>(
    weights: &'a ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: Range<usize>,
    constrained: bool,
    // Expert-bank authority: routed container if supplied, otherwise
    // spine. The override replaces ONLY the bank facts on each routed
    // layer; attention, norms, router state and topology stay
    // spine-owned. Generic — no representation is named on this path.
    routed: Option<&'a crate::ffn::ContainerRoutedBackend>,
) -> Result<GpuDecodeSetup<'a>, GenerateError> {
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_kquant_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };

    if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) || q4_ffn.is_none() {
        return Err(GenerateError::unsupported_backend(format!(
            "{}GPU generation requires backend Q4 support and interleaved Q4 FFN weights",
            if constrained { "constrained " } else { "" }
        )));
    }
    ensure_attention_supported(weights, backend)?;

    let first_layer = layer_range.start;
    let intermediate = gate_index.num_features(first_layer);
    let has_q4k = index.attn_kquant_layer_data(first_layer).is_some();
    let has_q8 = index.attn_q8_layer_data(first_layer).is_some();
    if intermediate == 0 || (!has_q4k && !has_q8) {
        return Err(GenerateError::missing_weights(format!(
            "{}GPU generation requires non-empty FFN features and Q4/Q8 attention weights",
            if constrained { "constrained " } else { "" }
        )));
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| {
            GenerateError::missing_weights("Q4 interleaved FFN format has invalid packed geometry")
        })?;
    let mut layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..weights.num_layers,
        q4_ffn.expect("checked above"),
        q4_ffn_per_matrix,
        ffn_format,
    );
    if let Some(routed) = routed {
        let mut overridden = 0usize;
        for (l, layer) in layers.iter_mut().enumerate() {
            let Some(moe) = layer.moe.as_mut() else {
                continue; // dense layers owe the container nothing
            };
            let ovr = routed
                .expert_bank_override(moe.num_experts, moe.expert_data_format, l)
                .map_err(|e| {
                    GenerateError::missing_weights(format!(
                        "routed container cannot serve layer {l}: {e}"
                    ))
                })?;
            moe.apply_expert_bank_override(ovr);
            overridden += 1;
        }
        // Fired evidence, not silence: the composed run states what it
        // actually swapped, once per setup.
        eprintln!(
            "[routed-metal] expert banks overridden on {overridden} layer(s); \
             format L0 = {:?}",
            layers
                .iter()
                .find_map(|l| l.moe.as_ref().map(|m| m.expert_data_format)),
        );
    }

    Ok(GpuDecodeSetup {
        layers,
        hidden,
        intermediate,
    })
}

pub(super) fn ensure_prompt_fits(seq_len: usize) -> Result<(), GenerateError> {
    if seq_len > DEFAULT_GPU_KV_CACHE_MAX_SEQ {
        return Err(GenerateError::prompt_too_long(
            seq_len,
            DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        ));
    }
    Ok(())
}

pub(super) fn reset_and_preallocate_kv_cache(weights: &ModelWeights, backend: &dyn ComputeBackend) {
    backend.reset_kv_cache();
    let kv_shapes = kv_cache_shapes_for_arch(weights);
    // Every layer at the full default, sliding or not. The window-derived
    // capacity (`kv_capacities_for_arch`) presumes compaction, and the GPU
    // decode path never compacts — it appends at `current_len` and attends
    // absolute rows — so a sliding layer sized to its window ran off the
    // end of its buffer past position `window x slack` (#229). The Metal
    // backend applies the same rule in `ensure_kv_cache_for_layers`; stating
    // it here too keeps the preallocation from being grown on the first
    // step.
    let kv_capacities = vec![DEFAULT_GPU_KV_CACHE_MAX_SEQ; kv_shapes.len()];
    backend.preallocate_kv_cache_per_layer_with_capacity(&kv_shapes, &kv_capacities);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_kquant_prompt(
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer<'_>],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
    seq_len: usize,
    qk_norm: bool,
    softcap: f32,
    failure_reason: &'static str,
) -> Result<Vec<f32>, GenerateError> {
    backend
        .prefill_kquant(layers, x, hidden, intermediate, seq_len, qk_norm, softcap)
        .ok_or_else(|| GenerateError::prefill_failed(failure_reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, make_test_weights};
    use larql_compute::CpuBackend;

    // ── has_heterogeneous_attention ──────────────────────────────────────

    #[test]
    fn has_heterogeneous_attention_false_for_uniform_arch() {
        let weights = make_test_weights();
        assert!(!has_heterogeneous_attention(&weights));
    }

    #[test]
    fn has_heterogeneous_attention_false_for_single_layer_models() {
        // The function short-circuits when num_layers < 2.
        let mut weights = make_test_weights();
        weights.num_layers = 1;
        assert!(!has_heterogeneous_attention(&weights));
    }

    #[test]
    fn has_heterogeneous_attention_false_for_q4k_synthetic() {
        // Q4K Gemma3 synthetic fixture is also uniform across layers.
        let weights = make_test_q4k_weights();
        assert!(!has_heterogeneous_attention(&weights));
    }

    // ── ensure_attention_supported ────────────────────────────────────────

    #[test]
    fn ensure_attention_supported_ok_for_uniform_model_on_cpu() {
        // Homogeneous model + CpuBackend (no HeterogeneousAttention cap) is fine.
        let weights = make_test_weights();
        let backend = CpuBackend;
        assert!(ensure_attention_supported(&weights, &backend).is_ok());
    }

    // ── ensure_prompt_fits ────────────────────────────────────────────────

    #[test]
    fn ensure_prompt_fits_ok_when_within_cache() {
        assert!(ensure_prompt_fits(0).is_ok());
        assert!(ensure_prompt_fits(1).is_ok());
        assert!(ensure_prompt_fits(DEFAULT_GPU_KV_CACHE_MAX_SEQ).is_ok());
    }

    #[test]
    fn ensure_prompt_fits_errors_when_exceeds_cache() {
        let result = ensure_prompt_fits(DEFAULT_GPU_KV_CACHE_MAX_SEQ + 1);
        let err = result.unwrap_err();
        assert!(matches!(err, GenerateError::PromptTooLong { .. }));
    }

    // ── build_gpu_decode_setup ────────────────────────────────────────────

    #[test]
    fn build_gpu_decode_setup_succeeds_on_cpu_backend_with_q4k_vindex() {
        // CpuBackend now advertises `supports_quant(Q4_K)=true`, and the
        // Q4K synthetic vindex has interleaved_kquant bytes + attn_kquant bytes.
        // The setup builds successfully — exercises the happy path
        // through every per-layer manifest read.
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = CpuBackend;
        let setup = build_gpu_decode_setup(
            &weights,
            &index,
            &backend,
            0..weights.num_layers,
            false,
            None,
        )
        .expect("Q4K fixture + CpuBackend should build setup");
        assert_eq!(setup.layers.len(), weights.num_layers);
        assert_eq!(setup.hidden, weights.hidden_size);
        assert!(setup.intermediate > 0);
    }

    #[test]
    fn build_gpu_decode_setup_succeeds_for_constrained_variant() {
        // The constrained branch only changes the error-message prefix,
        // and the success path is identical to the unconstrained one.
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = CpuBackend;
        let setup = build_gpu_decode_setup(
            &weights,
            &index,
            &backend,
            0..weights.num_layers,
            true,
            None,
        )
        .expect("constrained variant should build on Q4K fixture");
        assert_eq!(setup.layers.len(), weights.num_layers);
    }

    #[test]
    fn build_gpu_decode_setup_errors_when_no_q4_ffn_mmap() {
        // Empty VectorIndex with no FFN data → the
        // `if q4_ffn.is_none()` arm triggers the unsupported-backend
        // error path.
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let backend = CpuBackend;
        let result = build_gpu_decode_setup(
            &weights,
            &empty_index,
            &backend,
            0..weights.num_layers,
            false,
            None,
        );
        let err = match result {
            Ok(_) => panic!("empty vindex must be rejected"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("Q4"), "error must mention Q4: {msg}");
    }

    // ── reset_and_preallocate_kv_cache ────────────────────────────────────

    #[test]
    fn reset_and_preallocate_kv_cache_runs_on_cpu_backend() {
        // CpuBackend's reset/preallocate are no-ops — just confirms the
        // function executes without panicking and exercises both
        // backend calls.
        let weights = make_test_weights();
        let backend = CpuBackend;
        reset_and_preallocate_kv_cache(&weights, &backend);
    }

    // ── prefill_kquant_prompt ─────────────────────────────────────────────────

    #[test]
    fn prefill_q4_prompt_errors_when_backend_returns_none() {
        // CpuBackend's prefill_kquant default returns None → wrapper produces
        // a typed PrefillFailed error with the supplied reason.
        let backend = CpuBackend;
        // Empty layer slice + zero x — the backend returns None
        // immediately regardless of input shape.
        let result = prefill_kquant_prompt(
            &backend,
            &[],
            &[],
            16,
            32,
            1,
            false,
            0.0,
            "test-failure-reason",
        );
        let err = result.expect_err("CpuBackend prefill_kquant default returns None");
        assert!(matches!(err, GenerateError::PrefillFailed { .. }));
        assert_eq!(format!("{err}"), "test-failure-reason");
    }
}
