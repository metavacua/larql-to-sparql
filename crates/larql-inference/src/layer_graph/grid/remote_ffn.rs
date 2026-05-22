use super::config::GridRuntimeConfig;
use super::setup::{build_grid_pipeline_setup, reset_and_preallocate_grid_kv, RemotePatch};
use super::GridGenerateResult;
use crate::ffn::moe_remote::RemoteMoeError;
use crate::ffn::{FfnBackend, LayerShardedBackend};
use crate::forward::apply_norm;
use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::policy::{
    build_special_suppress_set_with_policy, pick_next_filtered_with_policy,
};
use crate::residual::rms_norm;
use larql_compute::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k, Q8KActivation};
use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

/// Autoregressive generation with Metal GPU attention and remote dense FFN.
///
/// For dense models (not MoE) where the entire FFN should be offloaded to a
/// remote server (`--ffn URL`). Metal handles attention on the local GPU;
/// every layer's FFN is a round trip to `remote` via `LayerShardedBackend::forward`.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    remote: &LayerShardedBackend,
    eos: &EosConfig,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Ffn)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    reset_and_preallocate_grid_kv(weights, backend);

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);

    for &tok_id in &prompt_ids {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let h_normed = apply_norm_for_ffn(weights, h_post_attn, layer);
            let x = ndarray::Array2::from_shape_vec((1, hidden), h_normed)
                .expect("shape must match hidden");
            let raw_out = remote.forward(layer, &x).row(0).to_vec();
            apply_post_ffn_norm(weights, &raw_out, layer)
        };

        let h = backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn);
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
    }

    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let mut ffn_rtt_ms = Vec::new();

    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = pick_next_filtered_with_policy(
        index,
        weights,
        &last0,
        backend,
        &suppress,
        tokenizer,
        &runtime.token_policy,
    );

    let first_tok = detok.push(first_id);
    let first_is_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        let tok_embed = crate::forward::embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let step_ffn_cell = std::cell::Cell::new(0.0f64);
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let t_ffn = std::time::Instant::now();
            // Pre-norm once, regardless of dispatch branch. The remote
            // FFN server expects pre-normed input (matches local forward
            // contract) and the previous code skipped pre-norm in the
            // f32 fallback paths — see chrishayuk/larql#114 (zero
            // hidden states on multi-token decode).
            let h_normed = apply_norm_for_ffn(weights, h_post_attn, layer);
            let raw_out = if hidden % crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS == 0 {
                let q8k = quantize_x_to_q8k(&h_normed);
                remote.forward_single_q8k(layer, &q8k).unwrap_or_else(|| {
                    let x = ndarray::Array2::from_shape_vec((1, hidden), h_normed.clone())
                        .expect("shape must match hidden");
                    remote.forward(layer, &x).row(0).to_vec()
                })
            } else {
                let x = ndarray::Array2::from_shape_vec((1, hidden), h_normed.clone())
                    .expect("shape must match hidden");
                remote.forward(layer, &x).row(0).to_vec()
            };
            let result = apply_post_ffn_norm(weights, &raw_out, layer);
            step_ffn_cell.set(step_ffn_cell.get() + t_ffn.elapsed().as_secs_f64() * 1000.0);
            result
        };

        let h_vec = backend
            .decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn)
            .ok_or_else(|| {
                RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
            })?;

        last_hidden_vec = h_vec;
        ffn_rtt_ms.push(step_ffn_cell.get());

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();

        let next_id = pick_next_filtered_with_policy(
            index,
            weights,
            &last_hidden,
            backend,
            &suppress,
            tokenizer,
            &runtime.token_policy,
        );

        let token_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(token_wall_ms);

        let tok_str = detok.push(next_id);
        let is_eos = eos.is_eos_with_tokenizer(next_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms,
    })
}

fn apply_norm_for_ffn(weights: &ModelWeights, h_post_attn: &[f32], layer: usize) -> Vec<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };

    let h = ndarray::Array2::from_shape_vec((1, h_post_attn.len()), h_post_attn.to_vec())
        .expect("apply_norm_for_ffn: shape error");

    let normed = match pre_ffn_key {
        Some(ref key) => apply_norm(weights, &h, key, norm_offset),
        None => rms_norm(&h, None, norm_offset),
    };
    normed.row(0).to_vec()
}

/// Apply the post-FFN normalisation that the local forward path
/// applies to a layer's FFN output before the residual add.
///
/// Remote FFN servers return the raw FFN result; the caller is
/// responsible for the matching post-norm so that remote and local
/// paths stay bit-equivalent. Three cases:
///   * arch has no post-norms (older Llama/Mistral-style) → pass
///     through unchanged. Checked first because the default
///     `post_feedforward_layernorm_key` impl returns `Some` for
///     every arch, so without this gate every pre-norm arch would
///     get an unwanted identity-weight RMS norm and diverge from
///     the local forward path.
///   * arch advertises a `post_feedforward_layernorm_key` for this
///     layer → apply that named norm.
///   * arch has post-norms but no per-layer key → identity-weight
///     RMS norm.
fn apply_post_ffn_norm(weights: &ModelWeights, ffn_out: &[f32], layer: usize) -> Vec<f32> {
    let arch = &*weights.arch;
    if !arch.has_post_norms() {
        return ffn_out.to_vec();
    }
    let norm_offset = arch.norm_weight_offset();
    let key = arch.post_feedforward_layernorm_key(layer);
    let h = ndarray::Array2::from_shape_vec((1, ffn_out.len()), ffn_out.to_vec())
        .expect("apply_post_ffn_norm: shape error");
    let normed = match key {
        Some(ref k) => apply_norm(weights, &h, k, norm_offset),
        None => rms_norm(&h, None, norm_offset),
    };
    normed.row(0).to_vec()
}

/// Pre-norm every layer's residual once, before the FFN dispatch.
///
/// Pulled out of `dispatch_ffn_with_q8k_fallback` and the streaming
/// decode loop so the composition is unit-testable independently of
/// the remote transport. Both branches of the dispatch (Q8K direct,
/// f32 fallback) consume the same pre-normed buffer — the previous
/// code only pre-normed in the Q8K branch, leaving the f32 fallback
/// sending raw residuals to the remote FFN (chrishayuk/larql#114).
fn prenorm_layers(weights: &ModelWeights, h_capture: &[Vec<f32>]) -> Vec<Vec<f32>> {
    h_capture
        .iter()
        .enumerate()
        .map(|(layer, h)| apply_norm_for_ffn(weights, h, layer))
        .collect()
}

/// Post-norm every layer's raw FFN output before it goes back into
/// the residual stream. Mirrors what the local forward path does and
/// keeps remote-FFN output bit-equivalent on post-norm archs
/// (Gemma 3 / 4). No-op on pre-norm archs — `apply_post_ffn_norm`
/// short-circuits via `has_post_norms`.
fn postnorm_layers(weights: &ModelWeights, raw_results: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    raw_results
        .into_iter()
        .enumerate()
        .map(|(layer, out)| apply_post_ffn_norm(weights, &out, layer))
        .collect()
}

fn dispatch_ffn_with_q8k_fallback(
    remote: &LayerShardedBackend,
    weights: &ModelWeights,
    h_capture: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let hidden = h_capture.first().map(|v| v.len()).unwrap_or(0);
    let h_normed = prenorm_layers(weights, h_capture);

    let raw_results = if hidden == 0 || !hidden.is_multiple_of(crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS)
    {
        remote.forward_predispatch_all(&h_normed)
    } else {
        let q8k_all: Vec<Q8KActivation> = h_normed.iter().map(|h| quantize_x_to_q8k(h)).collect();
        let results = remote.forward_predispatch_all_q8k(&q8k_all);
        let any_zero_result = results.iter().any(|v| v.iter().all(|&x| x == 0.0));
        if any_zero_result {
            remote.forward_predispatch_all(&h_normed)
        } else {
            results
        }
    };

    postnorm_layers(weights, raw_results)
}

/// Batch pre-dispatch variant of [`generate_with_remote_ffn`].
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_ffn_batch(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    remote: &LayerShardedBackend,
    eos: &EosConfig,
    predispatch_iters: usize,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let predispatch_iters = predispatch_iters.max(1);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Ffn)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;
    let num_layers = setup.num_layers;
    reset_and_preallocate_grid_kv(weights, backend);

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);

    for &tok_id in &prompt_ids {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut cap_fn);
        }
        backend.truncate_kv_cache(kv_len);

        let mut h2_final: Option<Vec<f32>> = None;
        for iter in 0..predispatch_iters {
            let is_final = iter + 1 == predispatch_iters;
            let h2 = dispatch_ffn_with_q8k_fallback(remote, weights, &h_capture);

            if !is_final {
                let mut new_cap: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let h2r = &h2;
                let nc = &mut new_cap;
                let mut fn_apply = |l: usize, h: &[f32]| -> Vec<f32> {
                    if nc.len() == l {
                        nc.push(h.to_vec());
                    }
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut fn_apply);
                backend.truncate_kv_cache(kv_len);
                h_capture = new_cap;
            } else {
                let h2r = &h2;
                let mut fn_final = |l: usize, _: &[f32]| -> Vec<f32> {
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                h2_final = backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut fn_final,
                );
            }
        }
        last_hidden_vec = h2_final.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode returned None during prefill".into())
        })?;
    }

    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let first_id = pick_next_filtered_with_policy(
        index,
        weights,
        &h_norm0.row(0).to_owned(),
        backend,
        &suppress,
        tokenizer,
        &runtime.token_policy,
    );
    let first_tok = detok.push(first_id);
    let first_is_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    let mut ffn_rtt_ms: Vec<f64> = Vec::new();
    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut cap_fn);
        }
        backend.truncate_kv_cache(kv_len);

        let mut h_out_opt: Option<Vec<f32>> = None;
        let mut step_ffn_ms = 0.0f64;

        for iter in 0..predispatch_iters {
            let is_final = iter + 1 == predispatch_iters;
            let t_ffn = std::time::Instant::now();
            let h2 = dispatch_ffn_with_q8k_fallback(remote, weights, &h_capture);
            step_ffn_ms += t_ffn.elapsed().as_secs_f64() * 1000.0;

            if !is_final {
                let h2r = &h2;
                let mut new_h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let new_h = &mut new_h_capture;
                let mut fn_apply = |l: usize, h: &[f32]| -> Vec<f32> {
                    if new_h.len() == l {
                        new_h.push(h.to_vec());
                    }
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut fn_apply);
                backend.truncate_kv_cache(kv_len);
                h_capture = new_h_capture;
            } else {
                let h2r = &h2;
                let mut fn_final = |l: usize, _: &[f32]| -> Vec<f32> {
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                h_out_opt = backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut fn_final,
                );
            }
        }

        let h_vec = h_out_opt.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
        })?;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_vec)
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();

        let next_id = pick_next_filtered_with_policy(
            index,
            weights,
            &last_hidden,
            backend,
            &suppress,
            tokenizer,
            &runtime.token_policy,
        );

        let token_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(token_wall_ms);
        ffn_rtt_ms.push(step_ffn_ms);

        let tok_str = detok.push(next_id);
        let is_eos = eos.is_eos_with_tokenizer(next_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::{apply_norm_for_ffn, apply_post_ffn_norm};
    use larql_models::test_fixtures::{make_gemma3_test_weights, make_test_weights};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── apply_post_ffn_norm ─────────────────────────────────────────

    #[test]
    fn apply_post_ffn_norm_passthrough_when_arch_has_no_post_norms() {
        // TinyModel arch reports `has_post_norms() == false`. The
        // helper must short-circuit and return the input verbatim:
        // remote FFN servers running against pre-Gemma-style archs
        // see the FFN output flow straight into the residual add.
        let w = make_test_weights();
        let hidden = w.hidden_size;
        let ffn_out: Vec<f32> = (0..hidden).map(|i| i as f32 - 7.0).collect();
        let out = apply_post_ffn_norm(&w, &ffn_out, 0);
        assert_eq!(out, ffn_out);
    }

    #[test]
    fn apply_post_ffn_norm_applies_named_norm_when_key_present() {
        // Gemma 3 arch supplies `post_feedforward_layernorm_key` for
        // every layer; the helper must route through `apply_norm`.
        // With identity-weight norms the output is RMS-normalised
        // (variance scaled) but distinct from a non-unit-RMS input —
        // "norm never ran" is the exact bug this helper closes for
        // remote FFN paths on post-norm archs.
        let w = make_gemma3_test_weights();
        let hidden = w.hidden_size;
        let ffn_out: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let out = apply_post_ffn_norm(&w, &ffn_out, 0);
        assert_eq!(out.len(), hidden);
        assert!(
            !approx_eq(&out, &ffn_out, 1e-6),
            "post-FFN norm must transform the input on a post-norm arch"
        );
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-3,
            "identity-weight post-FFN norm should rescale to unit RMS (got {rms})"
        );
    }

    #[test]
    fn apply_post_ffn_norm_layer_keys_distinct_per_layer() {
        // Per-layer key lookup: both layers must succeed and produce
        // hidden-sized output. Regression surface is a hardcoded
        // layer index in the helper.
        let w = make_gemma3_test_weights();
        let ffn_out: Vec<f32> = vec![0.25; w.hidden_size];
        let out0 = apply_post_ffn_norm(&w, &ffn_out, 0);
        let out1 = apply_post_ffn_norm(&w, &ffn_out, 1);
        assert_eq!(out0.len(), w.hidden_size);
        assert_eq!(out1.len(), w.hidden_size);
    }

    // ── apply_norm_for_ffn (parallel helper, previously untested) ───

    #[test]
    fn apply_norm_for_ffn_uses_pre_feedforward_key_on_post_norm_arch() {
        // Post-norm archs (Gemma 3) route the pre-FFN norm through
        // `pre_feedforward_layernorm`, not `post_attention_layernorm`.
        // Local forward and remote-FFN caller must agree.
        let w = make_gemma3_test_weights();
        let h_post_attn: Vec<f32> = (0..w.hidden_size).map(|i| i as f32 * 0.3).collect();
        let out = apply_norm_for_ffn(&w, &h_post_attn, 0);
        assert_eq!(out.len(), w.hidden_size);
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / w.hidden_size as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "expected unit-RMS, got {rms}");
    }

    #[test]
    fn apply_norm_for_ffn_uses_post_attention_key_on_pre_norm_arch() {
        // Pre-norm archs (TinyModel, Llama/Mistral layout) route the
        // pre-FFN norm through `post_attention_layernorm`. The
        // identity-weight fixture means we mainly assert shape and
        // that the call doesn't panic on the key the function selected.
        let w = make_test_weights();
        let h_post_attn: Vec<f32> = (0..w.hidden_size).map(|i| (i as f32 - 5.0) * 0.4).collect();
        let out = apply_norm_for_ffn(&w, &h_post_attn, 0);
        assert_eq!(out.len(), w.hidden_size);
    }

    // ── prenorm_layers / postnorm_layers composition ────────────────

    use super::{postnorm_layers, prenorm_layers};

    #[test]
    fn prenorm_layers_applies_pre_norm_per_layer() {
        // Multi-layer pre-norm pass over a synthetic h_capture.
        // Output length must match input, and each layer's vector
        // must individually be unit-RMS (identity-weight RMS norm
        // in the Gemma 3 fixture).
        let w = make_gemma3_test_weights();
        let hidden = w.hidden_size;
        let h_capture: Vec<Vec<f32>> = (0..w.num_layers)
            .map(|l| (0..hidden).map(|i| (l + i) as f32 * 0.25 + 1.0).collect())
            .collect();
        let normed = prenorm_layers(&w, &h_capture);
        assert_eq!(normed.len(), w.num_layers);
        for (l, v) in normed.iter().enumerate() {
            assert_eq!(v.len(), hidden);
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-3,
                "layer {l} prenorm should rescale to unit RMS, got {rms}"
            );
        }
    }

    #[test]
    fn postnorm_layers_applies_post_norm_per_layer_on_post_norm_arch() {
        // Multi-layer post-norm pass: raw FFN outputs come back from
        // the remote, helper applies the per-layer post-norm so the
        // residual add sees the same input the local forward path
        // would produce.
        let w = make_gemma3_test_weights();
        let hidden = w.hidden_size;
        let raw: Vec<Vec<f32>> = (0..w.num_layers)
            .map(|l| (0..hidden).map(|i| (l + i) as f32 * 0.5 + 2.0).collect())
            .collect();
        let post = postnorm_layers(&w, raw);
        assert_eq!(post.len(), w.num_layers);
        for (l, v) in post.iter().enumerate() {
            assert_eq!(v.len(), hidden);
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-3,
                "layer {l} postnorm should rescale to unit RMS, got {rms}"
            );
        }
    }

    #[test]
    fn postnorm_layers_is_identity_on_pre_norm_arch() {
        // TinyModel reports `has_post_norms() == false`. The helper
        // must short-circuit each layer to passthrough so the remote
        // FFN's raw output flows straight into the residual add.
        // Regression surface: if the per-layer short-circuit ever
        // gets gated wrong, a pre-norm-arch remote-FFN run silently
        // gets identity-RMS normalised and diverges from local.
        let w = make_test_weights();
        let raw: Vec<Vec<f32>> = (0..w.num_layers)
            .map(|l| (0..w.hidden_size).map(|i| (l + i) as f32).collect())
            .collect();
        let raw_clone = raw.clone();
        let post = postnorm_layers(&w, raw);
        assert_eq!(post, raw_clone);
    }

    #[test]
    fn ffn_norm_round_trip_dispatches_pre_normed_input_and_post_norms_output() {
        // End-to-end shape of the decode-path composition: the
        // dispatch closure (stand-in for the remote FFN) sees a
        // pre-normed buffer, and the caller post-norms its return
        // before the residual add. This is the contract that
        // chrishayuk/larql#114 violated when the f32 fallback
        // branches sent raw residuals.
        let w = make_gemma3_test_weights();
        let hidden = w.hidden_size;
        let h_post_attn: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.3 + 1.5).collect();

        // Mirror exactly what the decode moe_fn does, with an
        // identity-mapping dispatch closure. `seen_input` captures
        // what the "remote" was handed.
        let h_normed = apply_norm_for_ffn(&w, &h_post_attn, 0);
        let seen_input = h_normed.clone();
        let raw_out = h_normed.clone(); // identity dispatch
        let result = apply_post_ffn_norm(&w, &raw_out, 0);

        // Pre-normed input: dispatch sees unit-RMS, not the raw h.
        let in_rms = (seen_input.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!(
            (in_rms - 1.0).abs() < 1e-3,
            "dispatch must see pre-normed input (unit-RMS), got {in_rms}"
        );

        // Post-normed output: result is unit-RMS too (identity
        // dispatch fed unit-RMS through, post-norm on Gemma 3 keeps
        // it unit-RMS).
        let out_rms = (result.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!(
            (out_rms - 1.0).abs() < 1e-3,
            "post-normed output should be unit-RMS, got {out_rms}"
        );

        // And the result differs from the raw h_post_attn — without
        // the round-trip norms, an identity-dispatch would return
        // exactly h_post_attn.
        assert_ne!(
            &result, &h_post_attn,
            "decode-path round-trip must transform the residual, not echo it"
        );
    }
}
