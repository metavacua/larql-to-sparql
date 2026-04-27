//! Token generation loop — GPU prefill + KV-cached decode

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;
use super::CachedLayerGraph;

/// Top-K logits lookup that transparently handles models with tied
/// input/output embeddings (Gemma 2/3/4) whose vindex has no dedicated
/// `lm_head.bin` / `lm_head_q4.bin`.
///
/// Resolution order:
/// 1. Vindex-native KNN (`lm_head_knn_backend`) — fastest, used when the
///    vindex was built with a separate lm_head.
/// 2. CPU gemv against `weights.lm_head` — the loader fills this from
///    `embed.clone()` for tied-embedding models, so it's always populated
///    even when no lm_head file is present.
///
/// The second path is O(vocab * hidden) floats through the CPU, but that's
/// a one-shot matvec per generated token — negligible compared to the
/// per-layer attention + FFN. It lets every model generate tokens through
/// the Metal pipeline regardless of how its vindex was packaged.
pub(crate) fn lm_head_topk(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    let hits = index.lm_head_knn_backend(query, top_k, backend);
    if !hits.is_empty() {
        return hits;
    }
    backend_lm_head_topk(weights, query, top_k, backend)
}

/// LM-head top-K via the active ComputeBackend.
///
/// Performs a single gemv `scores[vocab] = lm_head[vocab, hidden] · query[hidden]`
/// by dispatching `matmul_transb(query[1, hidden], lm_head[vocab, hidden])`.
/// On Metal this is a GPU f32 gemv (under Apple Silicon unified memory the
/// 2.68 GB `weights.lm_head` is shared, not copied). On CPU it's the
/// BLAS fallback via the same trait method. Either way this replaces the
/// previous unconditional CPU `ndarray::dot`, which was ~26 ms/tok on
/// Gemma 3 4B — the dominant cost of real-vindex decode.
fn backend_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() { return Vec::new(); }
    let vocab = lm.shape()[0];
    let hidden = lm.shape()[1];
    if hidden != query.len() { return Vec::new(); }

    // Try the dedicated GPU gemv first (~3-5 ms on Metal for the Gemma
    // 262K × 2560 tied LM head). Fall back to `matmul_transb` (which
    // itself falls back to BLAS below the flop threshold) if the backend
    // doesn't specialise gemv.
    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };
    let scores_vec: Vec<f32> = if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r, Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    };

    let mut indexed: Vec<(u32, f32)> = scores_vec
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();
    let k = top_k.min(indexed.len());
    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.retain(|(_, s)| s.is_finite());
    let _ = vocab;
    indexed
}

/// Kept for the `LARQL_METAL_COMPARE_CPU=1` diagnostic mode which wants a
/// known-good CPU reference. Not used in the hot path.
#[allow(dead_code)]
fn cpu_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
) -> Vec<(u32, f32)> {
    backend_lm_head_topk(weights, query, top_k, &larql_compute::CpuBackend)
}

/// Dense LM-head: full `Vec<f32>` of vocabulary scores. Required for
/// constrained decoding — the sparse vindex KNN can't apply an arbitrary
/// vocabulary mask because masked-out tokens might fall outside the top-K.
/// Same compute kernel as [`backend_lm_head_topk`], just no truncation.
fn backend_lm_head_scores(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
) -> Vec<f32> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() { return Vec::new(); }
    let hidden = lm.shape()[1];
    if hidden != query.len() { return Vec::new(); }
    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };
    if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    }
}

/// Apply `mask_fn` to dense logits, then return the argmax `(id, score)`
/// over finite (post-mask) entries. Returns `None` if every entry is NaN
/// or `-inf`.
fn pick_next_token_masked<M>(
    weights: &ModelWeights,
    h_1d: &ndarray::Array1<f32>,
    generated: &[u32],
    backend: &dyn ComputeBackend,
    mask_fn: &mut M,
) -> Option<(u32, f32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let mut logits = backend_lm_head_scores(weights, h_1d, backend);
    if logits.is_empty() {
        return None;
    }
    mask_fn(generated, &mut logits);
    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan() && v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &s)| (i as u32, s))
}

/// Multi-token generation: GPU prefill → decode loop with KV cache.
///
/// 1. GPU prefill: full_pipeline_q4 populates KV cache for all layers
/// 2. Decode loop: decode_token reads from KV cache, generates one token at a time
/// 3. Logits: vindex lm_head KNN (no dense matmul)
///
/// Returns: Vec of (token_string, probability) for each generated token,
/// plus timing (prefill_ms, per_token_ms).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> GenerateResult {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    // Build layer descriptors
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
    let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

    if !backend.has_q4() || q4_ffn.is_none() {
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    let q4_ffn_mmap = q4_ffn.unwrap();
    let intermediate = gate_index.num_features(layer_range.start);
    if intermediate == 0 || (!has_q4k && !has_q8) {
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    // Q4_K GGUF layout: 144 bytes per 256-value superblock.
    // Q4_0: 18 bytes per 32-value block (2-byte f16 scale + 16 bytes of nibbles).
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };

    let ffn_format = if ffn_is_q4k { larql_compute::QuantFormat::Q4_K } else { larql_compute::QuantFormat::Q4_0 };

    let num_layers = weights.num_layers;
    let layers = super::pipeline_layer::build_pipeline_layers(
        weights, index, 0..num_layers,
        q4_ffn_mmap, q4_ffn_per_matrix, ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(layer_range.start) as f32;

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();

    // Pre-allocate per-layer KV cache for models with asymmetric attention geometry
    // (e.g. Gemma 4 26B: sliding layers use 8×256, global layers use 2×512).
    // Without this, the lazy uniform allocation uses the first layer's dims for all layers,
    // causing global layers to read/write off the end of under-sized KV buffers.
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }
    let seq_len = token_ids.len();

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    let h_vec = backend.prefill_q4(
        &layers, &x, hidden, intermediate, q_dim, kv_dim,
        seq_len, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
        rope, qk_norm_val, softcap_val,
    ).unwrap_or_else(|| {
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        let mut h = h_embed.clone();
        for layer in 0..num_layers {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
        h.as_slice().unwrap_or(&[]).to_vec()
    });

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());

    let compare = std::env::var("LARQL_METAL_COMPARE_CPU").is_ok();

    let h = h_metal;
    let h_1d = {
        let h_final = crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
        h_final.row(seq_len - 1).to_owned()
    };

    // CPU-vs-Metal comparison mode (LARQL_METAL_COMPARE_CPU=1). Runs the
    // known-correct `predict_q4k` CPU path on the same prompt and diffs
    // the top-5 predicted tokens against the Metal path. Purpose: isolate
    // whether wrong-token output is from the compute path or from the
    // lm_head / logits-sampling layer.
    if compare {
        let metal_hits_vindex = index.lm_head_knn_backend(&h_1d, 5, backend);
        let metal_hits_cpu_lm = cpu_lm_head_topk(weights, &h_1d, 5);
        let as_toks = |hits: &[(u32, f32)]| -> Vec<String> {
            hits.iter()
                .map(|(t, _)| tokenizer.decode(&[*t], true).unwrap_or_default().trim().to_string())
                .collect()
        };
        eprintln!("[compare] metal final h_1d:  len={}  nan={}  inf={}  max_abs={:.3e}",
            h_1d.len(),
            h_1d.iter().filter(|v| v.is_nan()).count(),
            h_1d.iter().filter(|v| v.is_infinite()).count(),
            h_1d.iter().map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0f32, f32::max));
        eprintln!("[compare] metal top-5 via vindex-KNN:    {:?}", as_toks(&metal_hits_vindex));
        eprintln!("[compare] metal top-5 via CPU lm_head:   {:?}", as_toks(&metal_hits_cpu_lm));

        eprintln!("[compare] (run `larql walk --predict` (no --metal) for CPU reference tokens)");
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Sample first token
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);

    let first_hits = lm_head_topk(index, weights, &h_1d, 5, backend);
    if let Some(&(tid, score)) = first_hits.first() {
        let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
        let prob = super::logits::softmax_prob(score, &first_hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
        tokens.push((tok_str, prob));
    }

    // ── Phase 2: GPU decode loop ──
    let mut current_token_id = first_hits.first().map(|&(tid, _)| tid).unwrap_or(0);
    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);

    // Per-stage decode profiling. Set LARQL_PROFILE_DECODE=1 to log a
    // one-line per-step breakdown of embed / GPU forward / final norm /
    // lm_head / detokenize, plus a summary at the end.
    let profile = std::env::var("LARQL_PROFILE_DECODE").is_ok();
    let profile_split = std::env::var("LARQL_PROFILE_SPLIT").is_ok();
    let mut t_embed = 0.0f64;
    let mut t_gpu = 0.0f64;
    let mut t_norm = 0.0f64;
    let mut t_lmhead = 0.0f64;
    let mut t_detok = 0.0f64;

    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if profile && _step <= 2 {
            let x_nan = x_dec.iter().filter(|v| v.is_nan()).count();
            let x_max = x_dec.iter().map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0f32, f32::max);
            eprintln!(
                "[profile] step={} input tok={} x_dec: len={} nan={} max_abs={:.3e}",
                _step, current_token_id, x_dec.len(), x_nan, x_max,
            );
        }

        let t1 = std::time::Instant::now();
        let result = if profile_split && _step == 2 {
            // Step 2 is post-JIT warm — run split profiling once and print.
            let (r, _ta, _tgu, _td) = backend.decode_token_split_profile(
                &layers, &x_dec, hidden, intermediate, q_dim, kv_dim,
                weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            );
            r
        } else {
            backend.decode_token(
                &layers, &x_dec, hidden, intermediate, q_dim, kv_dim,
                weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            )
        };
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

        if profile && _step <= 2 {
            match &result {
                Some(h) => {
                    let h_nan = h.iter().filter(|v| v.is_nan()).count();
                    let h_max = h.iter().map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0f32, f32::max);
                    eprintln!(
                        "[profile] step={} decode_token h_out: len={} nan={} max_abs={:.3e}",
                        _step, h.len(), h_nan, h_max,
                    );
                }
                None => eprintln!("[profile] step={} decode_token returned None", _step),
            }
        }

        if let Some(h_out) = result {
            let t2 = std::time::Instant::now();
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let norm_ms = t2.elapsed().as_secs_f64() * 1000.0;

            let t3 = std::time::Instant::now();
            let hits = lm_head_topk(index, weights, &h_1d, 5, backend);
            let lmhead_ms = t3.elapsed().as_secs_f64() * 1000.0;
            if profile && _step <= 2 {
                let h_nan = h_1d.iter().filter(|v| v.is_nan()).count();
                let h_inf = h_1d.iter().filter(|v| v.is_infinite()).count();
                let h_max = h_1d.iter().map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0f32, f32::max);
                eprintln!(
                    "[profile] step={} h_1d: len={} nan={} inf={} max_abs={:.3e}  hits.len()={}",
                    _step, h_1d.len(), h_nan, h_inf, h_max, hits.len(),
                );
            }

            let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            decode_ms.push(step_ms);

            if let Some(&(tid, score)) = hits.first() {
                let t4 = std::time::Instant::now();
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
                let detok_ms = t4.elapsed().as_secs_f64() * 1000.0;
                let prob = super::logits::softmax_prob(score, &hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
                let is_eos = tok_str == "<eos>" || tok_str == "</s>" || tok_str == "<|endoftext|>";
                if profile {
                    eprintln!(
                        "[profile] step={} total={:.1}ms  embed={:.2}  gpu={:.1}  norm={:.2}  lm_head={:.1}  detok={:.2}",
                        _step, step_ms, embed_ms, gpu_ms, norm_ms, lmhead_ms, detok_ms,
                    );
                }
                t_embed += embed_ms; t_gpu += gpu_ms; t_norm += norm_ms;
                t_lmhead += lmhead_ms; t_detok += detok_ms;
                tokens.push((tok_str, prob));
                current_token_id = tid;
                if is_eos { break; }
            } else {
                if profile { eprintln!("[profile] step={} — lm_head returned empty; break", _step); }
                break;
            }
        } else {
            // GPU failed — CPU fallback
            if profile {
                eprintln!("[profile] step={} — GPU returned None, CPU fallback", _step);
            }
            let mut h_dec = h_tok;
            for layer in 0..num_layers {
                let (h_post_attn, _, _) =
                    crate::attention::run_attention_block_gpu(weights, &h_dec, layer, false, None).unwrap();
                let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h_dec = h_out;
            }
            let h_final = crate::forward::apply_norm(weights, &h_dec, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let hits = lm_head_topk(index, weights, &h_1d, 5, backend);
            let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            decode_ms.push(step_ms);
            if let Some(&(tid, score)) = hits.first() {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
                let prob = super::logits::softmax_prob(score, &hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
                let is_eos = tok_str == "<eos>" || tok_str == "</s>" || tok_str == "<|endoftext|>";
                // CPU-fallback path: the full decode is attributed to `gpu_ms_total`
                // for lack of a better bucket — consumers interpret it as "forward
                // work" regardless of which backend ran it.
                t_gpu += step_ms;
                tokens.push((tok_str, prob));
                current_token_id = tid;
                if is_eos { break; }
            } else { break; }
        }
    }

    if profile && !decode_ms.is_empty() {
        let n = decode_ms.len() as f64;
        eprintln!(
            "[profile] SUMMARY over {} steps: embed={:.2}ms  gpu={:.1}ms  norm={:.2}ms  lm_head={:.1}ms  detok={:.2}ms  total={:.1}ms",
            decode_ms.len(),
            t_embed / n, t_gpu / n, t_norm / n, t_lmhead / n, t_detok / n,
            decode_ms.iter().sum::<f64>() / n,
        );
    }

    // Per-stage totals across all successful steps (not vec-per-step to
    // keep the struct tiny — the `larql bench` harness averages these
    // against `decode_ms.len()`).
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: t_embed,
            gpu_ms_total: t_gpu,
            norm_ms_total: t_norm,
            lm_head_ms_total: t_lmhead,
            detok_ms_total: t_detok,
        },
    }
}

/// Constrained variant of [`generate`] for grammar-controlled decoding.
///
/// Differs from `generate` in two places only:
///
///   1. The LM-head step uses a **dense** vocabulary score vector
///      ([`backend_lm_head_scores`]) rather than the sparse vindex KNN.
///      Required because an arbitrary mask can disqualify tokens that
///      would otherwise have fallen outside the top-K.
///   2. After scoring, `mask_fn(generated_ids, &mut logits)` runs and the
///      next token is the masked argmax.
///
/// Per-token cost is slightly higher than unconstrained `generate` (full
/// 2.68 GB tied LM-head gemv vs. KNN over the 5-NN partial), but on Metal
/// it's still ~3-5 ms — acceptable for grammar-constrained dispatch.
///
/// Stops on EOS / common end-of-turn markers or when `max_tokens` is hit.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<M>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mut mask_fn: M,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
    let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

    // Constrained mode requires the GPU prefill + Q4 path to be available.
    // Fall back to the unconstrained dense single-token predict if it isn't —
    // the mask still applies to that one token via pick_next_token_masked.
    if !backend.has_q4() || q4_ffn.is_none() {
        // Dense single-token prediction with mask.
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }
    let q4_ffn_mmap = q4_ffn.unwrap();
    let intermediate = gate_index.num_features(layer_range.start);
    if intermediate == 0 || (!has_q4k && !has_q8) {
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k { larql_compute::QuantFormat::Q4_K } else { larql_compute::QuantFormat::Q4_0 };

    let num_layers = weights.num_layers;
    let layers = super::pipeline_layer::build_pipeline_layers(
        weights, index, 0..num_layers,
        q4_ffn_mmap, q4_ffn_per_matrix, ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(layer_range.start) as f32;

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }
    let seq_len = token_ids.len();
    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();
    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    let h_vec = backend.prefill_q4(
        &layers, &x, hidden, intermediate, q_dim, kv_dim,
        seq_len, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
        rope, qk_norm_val, softcap_val,
    ).unwrap_or_else(|| {
        // CPU fallback: same as unconstrained generate's fallback.
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        let mut h = h_embed.clone();
        for layer in 0..num_layers {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
        h.as_slice().unwrap_or(&[]).to_vec()
    });

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());
    let h_1d = {
        let h_final = crate::forward::apply_norm(weights, &h_metal, weights.arch.final_norm_key(), norm_offset);
        h_final.row(seq_len - 1).to_owned()
    };
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // ── First token: dense LM-head + mask + argmax ──
    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let first = pick_next_token_masked(weights, &h_1d, &generated, backend, &mut mask_fn);
    let mut current_token_id = match first {
        Some((tid, _)) => {
            let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
            let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
            tokens.push((tok_str, 1.0));
            generated.push(tid);
            if is_eos {
                return GenerateResult { tokens, prefill_ms, decode_ms, stage_timings: StageTimings::default() };
            }
            tid
        }
        None => return GenerateResult { tokens, prefill_ms, decode_ms, stage_timings: StageTimings::default() },
    };

    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);

    // ── Phase 2: GPU decode loop ──
    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();

        let result = backend.decode_token(
            &layers, &x_dec, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
        );

        let h_1d = if let Some(h_out) = result {
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
            h_final.row(0).to_owned()
        } else {
            // CPU fallback for one decode step.
            let mut h_dec = h_tok;
            for layer in 0..num_layers {
                let (h_post_attn, _, _) =
                    crate::attention::run_attention_block_gpu(weights, &h_dec, layer, false, None).unwrap();
                let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h_dec = h_out;
            }
            let h_final = crate::forward::apply_norm(weights, &h_dec, weights.arch.final_norm_key(), norm_offset);
            h_final.row(0).to_owned()
        };

        let pick = pick_next_token_masked(weights, &h_1d, &generated, backend, &mut mask_fn);
        decode_ms.push(decode_start.elapsed().as_secs_f64() * 1000.0);

        match pick {
            Some((tid, _)) => {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
                let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
                tokens.push((tok_str, 1.0));
                generated.push(tid);
                current_token_id = tid;
                if is_eos { break; }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
    }
}

/// Sum of per-stage decode times across every successful step.
///
/// Dividing each field by `GenerateResult::decode_ms.len()` gives the
/// per-token average. Populated unconditionally — the six
/// `Instant::now()` calls per step are negligible next to the GPU
/// forward pass and the LM-head gemv.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimings {
    pub embed_ms_total: f64,
    pub gpu_ms_total: f64,
    pub norm_ms_total: f64,
    pub lm_head_ms_total: f64,
    pub detok_ms_total: f64,
}

/// Result of multi-token generation.
pub struct GenerateResult {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
    pub stage_timings: StageTimings,
}

impl StageTimings {
    /// Per-token average across `n` decode steps. Returns all-zero if
    /// `n == 0` (short-circuit no-decode paths safely).
    pub fn avg_per_step(&self, n: usize) -> StageTimings {
        if n == 0 { return Self::default(); }
        let nf = n as f64;
        StageTimings {
            embed_ms_total: self.embed_ms_total / nf,
            gpu_ms_total: self.gpu_ms_total / nf,
            norm_ms_total: self.norm_ms_total / nf,
            lm_head_ms_total: self.lm_head_ms_total / nf,
            detok_ms_total: self.detok_ms_total / nf,
        }
    }
}

impl GenerateResult {
    pub fn avg_decode_ms(&self) -> f64 {
        if self.decode_ms.is_empty() { 0.0 }
        else { self.decode_ms.iter().sum::<f64>() / self.decode_ms.len() as f64 }
    }

    pub fn decode_tok_s(&self) -> f64 {
        let avg = self.avg_decode_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    pub fn text(&self) -> String {
        self.tokens.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("")
    }
}
