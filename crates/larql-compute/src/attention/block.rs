//! CPU attention block — full layer attention computation.
//!
//! norm → Q/K/V projection → bias → V-norm → QK-norm → RoPE → GQA → O projection → residual.
//! Supports KV sharing (reuse K/V from a source layer).

use super::gqa::gqa_reduced_qk_all_weights;
use super::{AttentionAllWeights, AttentionWeights, SharedKV};
use ndarray::{s, Array2};

/// Run the full attention block. Returns (h_post_attn, attn_projected, optional_weights).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    run_attention_block_shared(weights, h, layer, capture_attention, None)
}

/// Run attention with optional shared K/V, returning K/V for caching.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_with_kv_out(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<AttentionWeights>,
    Array2<f32>,
    Array2<f32>,
)> {
    let (h_post, attn_proj, attn_w, k, v, _pre_o, _) = run_attention_block_core(
        weights,
        h,
        layer,
        capture_attention,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        None,
    )?;
    Some((h_post, attn_proj, attn_w, k, v))
}

/// Run attention with optional shared K/V (discards K/V output).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_shared(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    let (h_post, attn_proj, attn_w, _, _, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        capture_attention,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        None,
    )?;
    Some((h_post, attn_proj, attn_w))
}

/// Run attention, returning the pre-O-projection output per head.
/// Returns `(h_post_attn, pre_o)` where `pre_o` has shape `[seq, num_q * head_dim]`.
/// This is the equivalent of Python's `o_proj.register_forward_pre_hook`.
pub fn run_attention_block_with_pre_o(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let (h_post, _, _, _, _, pre_o, _) = run_attention_block_core(
        weights, h, layer, false, None, None, None, None, None, false, None,
    )?;
    Some((h_post, pre_o))
}

/// Run attention with optional shared K/V and return the pre-O-projection
/// output per query head.
///
/// This is the shared-KV-safe variant used by research/intervention adapters
/// that need to inspect a pre-W_O head before deciding how to replace it.
pub fn run_attention_block_shared_with_pre_o(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let (h_post, _, _, _, _, pre_o, _) = run_attention_block_core(
        weights, h, layer, false, shared_kv, None, None, None, None, false, None,
    )?;
    Some((h_post, pre_o))
}

/// Run attention with optional shared K/V and return both the pre-O output and
/// all per-query-position attention distributions.
///
/// This is a diagnostic surface for relation/address probes. It is separate
/// from normal attention capture because all-position weights are
/// O(heads * seq^2) memory.
pub fn run_attention_block_with_pre_o_and_all_attention_weights(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, AttentionAllWeights)> {
    let (h_post, _, _, _, _, pre_o, all_weights) = run_attention_block_core(
        weights, h, layer, false, shared_kv, None, None, None, None, true, None,
    )?;
    Some((h_post, pre_o, all_weights?))
}

/// Run attention with optional shared K/V and return the pre-O output plus
/// all-position attention distributions computed from a reduced QK dot product.
///
/// The real attention output remains full-rank. Only the diagnostic attention
/// weights use `qk_rank`, so this can test reduced address computation without
/// changing the model forward path.
pub fn run_attention_block_with_pre_o_and_reduced_qk_attention_weights(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
    qk_rank: usize,
) -> Option<(Array2<f32>, Array2<f32>, AttentionAllWeights)> {
    let (h_post, _, _, _, _, pre_o, all_weights) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        Some(qk_rank),
    )?;
    Some((h_post, pre_o, all_weights?))
}

/// Run attention while zeroing selected pre-O-projection query heads before W_O.
///
/// Returns the post-attention residual and, when K/V were computed by this call,
/// the K/V pair for cross-layer sharing.
pub fn run_attention_block_zero_pre_o_heads(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    heads: &[usize],
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        Some(heads),
        None,
        None,
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while replacing one pre-O-projection query head before W_O.
///
/// `replacement` must have shape `[seq_len, head_dim]`.
pub fn run_attention_block_replace_pre_o_head(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    head: usize,
    replacement: &Array2<f32>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        Some((head, replacement)),
        None,
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while explicitly subtracting selected query-head
/// contributions from the O-projected tensor before the attention residual path.
///
/// This is numerically equivalent to zeroing those pre-W_O heads, but it checks
/// the head-to-W_O block indexing independently.
pub fn run_attention_block_subtract_pre_o_heads(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    heads: &[usize],
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        Some(heads),
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while replacing one query-head residual-space contribution
/// after W_O projection and before the attention residual path.
///
/// `replacement_delta` must have shape `[seq_len, hidden_size]` and represents
/// the residual-space contribution that should replace `W_O^head y_head`.
/// This is the Mode D validation surface: runtime lookup/add tables can bypass
/// W_O entirely while the rest of the layer remains unchanged.
pub fn run_attention_block_replace_head_residual_delta(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    head: usize,
    replacement_delta: &Array2<f32>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        None,
        Some((head, replacement_delta)),
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Core attention block implementation.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn run_attention_block_core(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
    zero_pre_o_heads: Option<&[usize]>,
    replace_pre_o_head: Option<(usize, &Array2<f32>)>,
    subtract_pre_o_heads: Option<&[usize]>,
    replace_head_residual_delta: Option<(usize, &Array2<f32>)>,
    capture_all_attention: bool,
    reduced_qk_rank: Option<usize>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<AttentionWeights>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Option<AttentionAllWeights>,
)> {
    use crate::forward::{add_bias, dot_proj};
    use crate::residual::{rms_norm_heads_no_weight, rms_norm_qk_for_arch};

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
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    // Per-layer stage dumps, paired with Metal via LARQL_CPU_STAGE_DUMP=<dir>.
    // Default is layer 0 (noise budget); set LARQL_STAGE_DUMP_LAYER=<N> to
    // capture a specific layer instead — Gemma 4 global layers (5, 11, …)
    // are useful for bisecting partial-RoPE / V-norm interactions.
    let dump_cfg = crate::forward::dump_config::DumpConfig::get();
    let stage_dump = dump_cfg.stage_dir(layer);
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };

    // Input norm
    let h_norm =
        crate::forward::apply_norm(&weights, h, &arch.input_layernorm_key(layer), norm_offset);
    dump_f32("norm_out", &h_norm);

    // Q projection (always from current hidden state)
    let w_q = weights.tensor(&arch.attn_q_key(layer))?;
    let w_o = weights.tensor(&arch.attn_o_key(layer)).unwrap();
    let mut q_full = dot_proj(&h_norm, w_q);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }
    dump_f32("q_out_raw", &q_full);

    // QK norm on Q
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
        Some(norm_w) => rms_norm_qk_for_arch(&q_full, norm_w, num_q, head_dim, qk_norm_off, arch),
        None => q_full,
    };
    dump_f32("q_out_after_qk_norm", &q_normed);

    // RoPE on Q
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
        0,
        pos_divisor,
        rope_scaling,
    );

    // K/V: either from shared cache or computed fresh
    let (k_rope, v_final) = if let Some((cached_k, cached_v)) = shared_kv {
        (cached_k.clone(), cached_v.clone())
    } else {
        let w_k = weights.tensor(&arch.attn_k_key(layer)).unwrap();

        let mut k_full = dot_proj(&h_norm, w_k);
        if let Some(bias) = arch
            .attn_k_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            add_bias(&mut k_full, bias);
        }

        let k_normed = match arch
            .attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            Some(norm_w) => {
                rms_norm_qk_for_arch(&k_full, norm_w, num_kv, head_dim, qk_norm_off, arch)
            }
            None => k_full.clone(),
        };

        // V projection. Always go through the stored W_v tensor when it
        // exists — including on `attention_k_eq_v` (Gemma 4 global) layers
        // where the bytes in W_v were derived from W_k at extraction time.
        // The reason: the vindex re-quantises V as Q6_K while K stays Q4_K
        // (see `format/weights/write.rs`: `is_v { quantize_q6_k } else {
        // quantize_q4_k }`), so `Q6_K_dequant(K_bytes)` is numerically
        // closer to the original bf16 weight than `Q4_K_dequant(K_bytes)`.
        // Metal's V projection uses the Q6_K path; the old CPU shortcut
        // (`v = k_full`) was ~0.25 off per element on Gemma 4 31B L5+,
        // which is what L5's attn_out drift was tracking.
        //
        // Fallback: when W_v is genuinely absent from the vindex (older
        // extracts with no v_proj tensor for `attention_k_eq_v` layers),
        // reuse `k_full` — matches pre-Q6K-V behaviour.
        let v_full = if let Some(w_v) = weights.tensor(&arch.attn_v_key(layer)) {
            let mut v = dot_proj(&h_norm, w_v);
            if let Some(bias) = arch
                .attn_v_bias_key(layer)
                .and_then(|k| weights.vectors.get(&k))
            {
                add_bias(&mut v, bias);
            }
            if arch.has_v_norm() {
                v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
            }
            v
        } else if arch.has_v_norm() {
            rms_norm_heads_no_weight(&k_full, num_kv, head_dim)
        } else {
            k_full.clone()
        };

        let k_r = crate::attention::rope::apply_rope_partial_at_full(
            &k_normed,
            num_kv,
            head_dim,
            layer_rope_base,
            rotary_frac,
            0,
            pos_divisor,
            rope_scaling,
        );
        (k_r, v_full)
    };

    dump_f32("q_out_after_rope", &q_rope);
    dump_f32("k_out_after_rope", &k_rope);
    dump_f32("v_out", &v_final);

    // GQA attention
    let softcap = arch.attn_logit_softcapping();
    // Attention sinks: one learned logit per query head, competing in the
    // softmax and then discarded (GPT-OSS). Absent for every other
    // architecture, in which case the softmax is the ordinary one.
    let sinks = super::sinks::resolve(arch.attn_sinks_key(layer), &weights.vectors, num_q, layer);
    let reduced_qk_weights = reduced_qk_rank.map(|rank| {
        gqa_reduced_qk_all_weights(
            &q_rope, &k_rope, num_q, head_dim, reps, scale, seq_len, softcap, sinks, rank,
        )
    });
    // Sliding window for THIS layer, from the shared rule the Metal
    // pipeline spec also uses. `None` on a full-attention layer (or an
    // architecture without windows) leaves the maths bit-identical to
    // the unwindowed path.
    let window = crate::forward_overrides::effective_attention_window_for_layer(arch, layer);
    let (mut attn_out, attn_weights, full_all_attn_weights) = {
        let (out, last, all) = super::gqa::gqa_attention_capture(
            &q_rope,
            &k_rope,
            &v_final,
            num_q,
            head_dim,
            reps,
            scale,
            seq_len,
            capture_attention && !capture_all_attention,
            capture_all_attention,
            softcap,
            sinks,
            window,
        );
        (out, last, all)
    };
    let all_attn_weights = reduced_qk_weights.or(full_all_attn_weights);
    if let Some(heads) = zero_pre_o_heads {
        for &head in heads {
            if head >= num_q {
                return None;
            }
            let start = head * head_dim;
            let end = start + head_dim;
            attn_out.slice_mut(s![.., start..end]).fill(0.0);
        }
    }
    if let Some((head, replacement)) = replace_pre_o_head {
        if head >= num_q || replacement.nrows() != seq_len || replacement.ncols() != head_dim {
            return None;
        }
        let start = head * head_dim;
        let end = start + head_dim;
        attn_out
            .slice_mut(s![.., start..end])
            .assign(&replacement.view());
    }
    dump_f32("attn_out", &attn_out);

    // O projection
    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(heads) = subtract_pre_o_heads {
        for &head in heads {
            if head >= num_q {
                return None;
            }
            let start = head * head_dim;
            let end = start + head_dim;
            let head_out = attn_out.slice(s![.., start..end]);
            let w_o_head = w_o.slice(s![.., start..end]);
            let contribution = dot_proj(&head_out, &w_o_head);
            attn_projected -= &contribution;
        }
    }
    if let Some((head, replacement_delta)) = replace_head_residual_delta {
        if head >= num_q
            || replacement_delta.nrows() != seq_len
            || replacement_delta.ncols() != weights.hidden_size
        {
            return None;
        }
        let start = head * head_dim;
        let end = start + head_dim;
        let head_out = attn_out.slice(s![.., start..end]);
        let w_o_head = w_o.slice(s![.., start..end]);
        let original_contribution = dot_proj(&head_out, &w_o_head);
        attn_projected -= &original_contribution;
        attn_projected += replacement_delta;
    }
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }
    dump_f32("o_out", &attn_projected);

    // Residual connection
    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            &weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h + &(&normed * res_mult)
        } else {
            h + &normed
        }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((
        h_post_attn,
        attn_projected,
        attn_weights,
        k_rope,
        v_final,
        attn_out,
        all_attn_weights,
    ))
}

#[cfg(test)]
mod tests;
