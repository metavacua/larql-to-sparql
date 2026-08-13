use ndarray::Array2;

use crate::attention::SharedKV;

use super::gqa_step::gqa_attention_decode_step_windowed;
use super::q4k_direct::run_attention_block_decode_step_q4k_direct;

/// Decode-step attention with optional GPU-accelerated projections
/// (Q/K/V/O matmuls route through `ComputeBackend::matmul_transb` when
/// `backend` is `Some`). GQA softmax + weighted-V stays on CPU —
/// that's O(cached_len × head_dim × num_q) per step and rarely the
/// bottleneck vs the hidden×hidden projection gemms.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_backend(
    weights: larql_models::WeightsView,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::dot_proj_gpu;
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;

    let h_norm = crate::forward::apply_norm(
        &weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let w_q = weights.tensor(&arch.attn_q_key(layer))?;
    let w_o = weights.tensor(&arch.attn_o_key(layer))?;
    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    // M1: honour every scaling family (llama3 / YaRN / linear), not just
    // llama3 — the same resolver the Metal pipeline spec reads.
    let rope_scaling = crate::forward_overrides::effective_rope_freq_scaling(arch);
    let q_rope = crate::attention::rope::apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        rope_scaling,
    );

    // New token's K, V — RoPE'd at `position`, then appended to cache.
    let w_k = weights.tensor(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.has_tensor(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensor(&arch.attn_v_key(layer))?
    };

    let mut k_full_new = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full_new = dot_proj_gpu(&h_norm, w_v, backend);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = crate::attention::rope::apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        rope_scaling,
    );

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let kv_dim = num_kv * head_dim;
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    // Per-layer sliding window from the shared rule; `None` leaves
    // this bit-identical to the unwindowed step.
    let window = crate::forward_overrides::effective_attention_window_for_layer(arch, layer);
    let attn_out = gqa_attention_decode_step_windowed(
        &q_rope,
        &k_concat,
        &v_concat,
        num_q,
        head_dim,
        reps,
        scale,
        softcap,
        crate::attention::sinks::resolve(
            arch.attn_sinks_key(layer),
            |k| weights.vectors.get(k).map(|v| v.as_slice()),
            num_q,
            layer,
        ),
        window,
    );

    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            &weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

/// `LARQL_Q4K_DIRECT_ATTN=1`: route decode-step attention projections through
/// the Q4K-direct kernels (packed bytes from the index) instead of the
/// f32-BLAS path over pre-dequantised `weights.tensors`. Single source of
/// truth for the flag — `CpuBackend::attention_step` and the engine walk
/// loops (via [`run_attention_block_decode_step_auto`]) must make the same
/// choice. Cached once; never in the hot loop.
pub fn q4k_direct_attn_enabled() -> bool {
    crate::options::q4k_direct_attn_enabled()
}

/// Best-available decode-step attention for callers that own their cache as
/// `SharedKV` tuples (engine walk loops, the cached-generation parity
/// oracle): Q4K-direct projections (int8 under `LARQL_Q4K_ATTN_INT8`, asm
/// under `LARQL_Q4K_ASM`) when the flag is on and an index with attention
/// bytes is supplied, else the f32 path — the SAME per-layer choice
/// `CpuBackend::attention_step` makes on the dispatch path, so engines and
/// the oracle stay numerically aligned. With the flag off (default) this is
/// byte-identical to calling `run_attention_block_decode_step_backend`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_auto(
    weights: larql_models::WeightsView,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
    index: Option<&dyn crate::KvIndex>,
) -> Option<(Array2<f32>, SharedKV)> {
    if q4k_direct_attn_enabled() {
        if let (Some(be), Some(idx)) = (backend, index) {
            // Q4K-direct reads native packed bytes from `index` (not the
            // dequant scratch) — canonical weights for norms/config.
            if let Some(r) = run_attention_block_decode_step_q4k_direct(
                weights.canonical(),
                h_new,
                layer,
                kv_entry,
                abs_position,
                be,
                idx,
            ) {
                return Some(r);
            }
        }
    }
    run_attention_block_decode_step_backend(weights, h_new, layer, kv_entry, abs_position, backend)
}

/// `LARQL_Q4K_ATTN_INT8=1`: upgrade the Q4K-direct attention projections from
/// the f32-activation kernels (`q4k_matvec`/`q6k_matvec` via `quant_matvec`)
/// to the int8 Q8_K SDOT kernels (`q4k_q8k_matvec_into`/`q6k_q8k_matvec_into`,
/// asm-aware under `LARQL_Q4K_ASM`) — the same numerics the dense-model
/// production attention (`attention_decode_step_native`) has always used.
/// The 26B stage split showed attention at ~54% of decode while moving only
/// ~26% of the bytes: the f32-activation kernel is ~3× worse per byte than
/// the expert path's int8 kernels. Default off = the existing f32-activation
/// behaviour, byte-identical.
pub(super) fn attn_int8_enabled() -> bool {
    crate::options::q4k_attn_int8_enabled()
}
