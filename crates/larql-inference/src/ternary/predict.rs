//! Model / layer types and the full-sequence prediction path.
//!
//! Split out of the former single-file `ternary.rs`; [`super`] shows
//! how the pieces compose.

// Siblings resolve through the parent's re-exports.
use super::*;
use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_a8_f32_into, matvec_i2s_a8_into, quantize_activation_i8, BitLinearWeight,
};
use ndarray::{Array1, Array2, ArrayView2};

/// Complete BitNet 1.58 model — every tensor needed for a forward
/// pass.  Built by `larql-vindex::extract::bitnet_loader` from a
/// `--keep-quant` vindex; feed into [`predict_bitnet`].
pub struct BitnetModel {
    /// Per-layer BitLinear projections + RMSnorm weights.
    pub layers: Vec<BitnetLayer>,
    /// Token embedding table, shape [vocab, hidden], f32.
    /// (Source GGUF has it in F16; we expand on load — the embed
    /// table is small relative to the BitLinear weights.)
    pub embed: Array2<f32>,
    /// Optional embed scale (most BitNet builds = 1.0).
    pub embed_scale: f32,
    /// Output RMSnorm weight, length = hidden_size.
    pub output_norm: Vec<f32>,
    /// LM head matrix, shape [vocab, hidden], f32.  Often tied to
    /// embed; supplied separately so the loader can decide.
    pub lm_head: Array2<f32>,
    /// RMSnorm epsilon used everywhere.
    pub eps: f32,
    /// Per-head dimension (= hidden / n_q_heads typically).
    pub head_dim: usize,
    /// Number of query heads.
    pub n_q_heads: usize,
    /// Number of key/value heads (GQA: usually < n_q_heads).
    pub n_kv_heads: usize,
    /// RoPE base (theta) — read from GGUF metadata.
    pub rope_base: f64,
}

/// One transformer block's worth of BitLinear weights + norms.
pub struct BitnetLayer {
    pub attn_norm: Vec<f32>,     // input RMSnorm, length = hidden
    pub attn_q: BitLinearWeight, // [hidden, hidden] (q heads x head_dim packed)
    pub attn_k: BitLinearWeight, // [n_kv_heads * head_dim, hidden]
    pub attn_v: BitLinearWeight, // [n_kv_heads * head_dim, hidden]
    pub attn_sub_norm: Vec<f32>, // post-attn RMSnorm, length = hidden
    pub attn_o: BitLinearWeight, // [hidden, hidden]
    pub ffn: BitNetFfn,          // self-contained FFN block
}

/// One top-K prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPrediction {
    pub token: String,
    pub probability: f64,
}

/// Run a full BitNet forward pass and return top-K next-token
/// predictions for the position immediately after `token_ids`.
///
/// Single-shot prefill, no KV cache.  Adequate for pg_infer's
/// `infer()` SQL surface (one-shot per call) and for the bug
/// report's repro path (a single `/v1/infer` from curl).
///
/// Memory profile at BitNet b1.58 2 B 4 T:
///   - weights resident:  ~1.1 GB (the I2_S bytes + scales + norms)
///   - per-call working:  ~10 MB (h, q/k/v, scratch buffers)
///
/// Compare to the dense f16 path's ~5 GB resident — that's the
/// architectural goal closed by this commit.
pub fn predict_bitnet(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> Vec<TernaryPrediction> {
    if token_ids.is_empty() {
        return Vec::new();
    }
    let seq_len = token_ids.len();
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    debug_assert!(n_q_heads >= n_kv_heads, "GQA: n_q_heads >= n_kv_heads");
    debug_assert_eq!(
        n_q_heads * head_dim,
        hidden,
        "hidden = n_q_heads * head_dim"
    );

    // 1. Embed lookup -> residual stream h: [seq_len, hidden].
    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok) in token_ids.iter().enumerate() {
        let row = model.embed.row(tok as usize % model.embed.shape()[0]);
        let mut h_row = h.row_mut(i);
        for (dst, &src) in h_row.iter_mut().zip(row.iter()) {
            *dst = src * model.embed_scale;
        }
    }

    // Scratch buffers reused across layers.
    let mut x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut q = Array2::<f32>::zeros((seq_len, n_q_heads * head_dim));
    let mut k = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut v = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut attn_pool = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_pool_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for layer in &model.layers {
        // a. attn input norm.
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.attn_norm,
                model.eps,
                x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
        }

        // b. Q/K/V projections via ternary matvec, per token. Q/K/V share
        //    x_norm.row(i), so quantise it to int8 once (A8) and reuse.
        for i in 0..seq_len {
            let (x_i8, x_scale) = quantize_activation_i8(x_norm.row(i).as_slice().unwrap());
            matvec_i2s_a8_into(
                &layer.attn_q,
                &x_i8,
                x_scale,
                q.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_q shape");
            matvec_i2s_a8_into(
                &layer.attn_k,
                &x_i8,
                x_scale,
                k.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_k shape");
            matvec_i2s_a8_into(
                &layer.attn_v,
                &x_i8,
                x_scale,
                v.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_v shape");
        }

        // c. RoPE on Q (per q-head) and K (per kv-head).
        let q_rotated =
            larql_compute::attention::rope::apply_rope(&q, n_q_heads, head_dim, model.rope_base);
        let k_rotated =
            larql_compute::attention::rope::apply_rope(&k, n_kv_heads, head_dim, model.rope_base);

        // d. Per-head causal-masked scaled-dot-product attention.
        attn_pool.fill(0.0);
        scaled_dot_product_attention_gqa(
            q_rotated.view(),
            k_rotated.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            attn_pool.view_mut(),
        );

        // e. Post-attn norm + output projection.
        for i in 0..seq_len {
            rmsnorm_into(
                attn_pool.row(i).as_slice().unwrap(),
                &layer.attn_sub_norm,
                model.eps,
                attn_pool_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            matvec_i2s_a8_f32_into(
                &layer.attn_o,
                attn_pool_norm.row(i).as_slice().unwrap(),
                attn_out.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_o shape");
        }

        // f. residual h += attn_out
        h += &attn_out;

        // g. FFN: per-token RMSnorm + BitNetFfn.forward_into +
        //    residual.  We call forward_into so the per-token gate /
        //    up / hid scratch stays out of the hot allocator.
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.ffn.ffn_norm,
                model.eps,
                ffn_x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            // BitNetFfn.forward_into expects the *un-normed* x in its
            // signature so it can run its own input norm.  Since we
            // already did that here (so the same x_norm could be
            // reused), we replicate the rest of forward_into manually
            // to skip the redundant RMSnorm step.
            ffn_forward_after_input_norm(
                &layer.ffn,
                ffn_x_norm.row(i).as_slice().unwrap(),
                model.eps,
                &mut ffn_gate,
                &mut ffn_up,
                &mut ffn_hid,
                &mut ffn_out_row,
            );
            for (dst, &src) in h.row_mut(i).iter_mut().zip(ffn_out_row.iter()) {
                *dst += src;
            }
        }
    }

    // Final norm + lm_head over the LAST token only.
    let last_h = h.row(seq_len - 1).to_owned();
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        last_h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_final_arr = Array1::from(h_final);
    let logits = model.lm_head.dot(&h_final_arr);

    // Top-K softmax.  Stable softmax: subtract max before exp.
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) as f64).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs
        .into_iter()
        .take(top_k)
        .filter_map(|(token_id, prob)| {
            tokenizer
                .id_to_token(token_id as u32)
                .map(|s| TernaryPrediction {
                    token: s,
                    probability: prob,
                })
        })
        .collect()
}

/// FFN forward pass that skips the input RMSnorm (caller already did
/// it).  Used by `predict_bitnet` to avoid double-norming when we
/// pre-compute the input norm once per layer.
pub(super) fn ffn_forward_after_input_norm(
    ffn: &BitNetFfn,
    x_norm: &[f32],
    eps: f32,
    gate: &mut [f32],
    up: &mut [f32],
    hid: &mut [f32],
    y: &mut [f32],
) {
    let inter = ffn.gate.rows;
    debug_assert_eq!(gate.len(), inter);
    debug_assert_eq!(up.len(), inter);
    debug_assert_eq!(hid.len(), inter);

    // gate / up projections. Both share x_norm — quantise to int8 once (A8).
    let (x_i8, x_scale) = quantize_activation_i8(x_norm);
    matvec_i2s_a8_into(&ffn.gate, &x_i8, x_scale, gate).expect("gate shape");
    matvec_i2s_a8_into(&ffn.up, &x_i8, x_scale, up).expect("up shape");

    // Squared-ReLU activation.
    for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
        let relu = g.max(0.0);
        *h = relu * relu * u;
    }

    // Post-gate-up norm.
    let mut hid_norm = vec![0.0f32; inter];
    rmsnorm_into(hid, &ffn.ffn_sub_norm, eps, &mut hid_norm);

    // Down projection.
    matvec_i2s_a8_f32_into(&ffn.down, &hid_norm, y).expect("down shape");
}

/// Causal-masked scaled-dot-product attention with GQA support.
///
/// `q` is `[seq_len, n_q_heads * head_dim]`, `k` and `v` are
/// `[seq_len, n_kv_heads * head_dim]`.  Each q-head maps to k/v
/// head `head_idx % n_kv_heads` (standard GQA); when `n_kv_heads
/// == n_q_heads` this is plain MHA.
///
/// Output is written to `out` (shape `[seq_len, hidden]` where
/// `hidden = n_q_heads * head_dim`).  Mask is causal: position `i`
/// only attends to positions `0..=i`.
pub(super) fn scaled_dot_product_attention_gqa(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    let seq_len = q.shape()[0];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    out.fill(0.0);

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        // For each query position, compute attention over all
        // earlier (and self) key positions.
        for i in 0..seq_len {
            // 1. scores[j] = (q[i, q_head] · k[j, kv_head]) * scale
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for j in 0..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[(i, q_off + d)] * k[(j, kv_off + d)];
                }
                scores[j] = dot * scale;
            }

            // 2. Stable softmax over the unmasked (j ≤ i) prefix.
            let max_score = scores[..=i]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for s in scores[..=i].iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores[..=i].iter_mut() {
                    *s /= sum;
                }
            }

            // 3. out[i, q_head] += sum_j weights[j] * v[j, kv_head]
            for j in 0..=i {
                let w = scores[j];
                if w == 0.0 {
                    continue;
                }
                for d in 0..head_dim {
                    out[(i, q_off + d)] += w * v[(j, kv_off + d)];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV-cached decode
// ─────────────────────────────────────────────────────────────────────────────
//
// Closes the "single-shot prefill, no KV cache" gap from the
// v0.3.0-bitnet release notes.  Three-piece API:
//
//   prefill(model, tokens) -> (BitnetKvCache, last_logits)
//      Run prefill over the full prompt, accumulating per-layer K/V
//      across all positions.  Returns the cache + the next-token
//      logits at the last prompt position.
//
//   decode_step(model, cache, new_token) -> next_logits
//      Append one token: project Q/K/V for it, append K/V to the
//      cache, run causal-masked attention against the entire cached
//      history (not against the new K/V alone — that would lose
//      context), apply the rest of the layer stack, return logits.
//
//   generate(model, tokenizer, prompt, max_new_tokens, sampler) -> String
//      Convenience wrapper that runs prefill then decodes step by
//      step until either (a) max_new_tokens reached or (b) the
//      sampler returns a stop signal.
//
// The cache is opaque to callers — they hold it across decode_step
// calls but never mutate it directly.  Per-layer storage is
// `Vec<Array2<f32>>` (one [past_len, n_kv_heads * head_dim] tensor
// per layer for K and V respectively); the cache grows by one row
// per decode_step.
