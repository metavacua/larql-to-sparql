//! CPU forward pass driven by a Q4_K / Q6_K vindex.
//!
//! The normal CPU path reads attention Q/K/V/O and FFN gate/up/down from
//! `weights.tensors` as f32 matrices. For a Q4 vindex those tensors were
//! never loaded (expanding 31B to f32 is ~127 GB and won't fit on a 96 GB
//! machine), so this module dequantises one layer's worth of weights into
//! `weights.tensors`, runs the existing `run_layer_with_ffn` against it,
//! then removes the entries before moving to the next layer. Peak f32 heap
//! stays around 1.8 GB per layer (the 31B down_proj) — the rest of the
//! model lives on disk through `VectorIndex` mmaps.
//!
//! The forward path reuses every attention / QK-norm / RoPE / GQA /
//! GEGLU routine from the f32 code, so Gemma 2/3/4 model families all
//! work. A future optimisation would call
//! `larql_compute::cpu::ops::q4k_matvec` directly to avoid the per-layer
//! dequant, but that would mean re-implementing the whole attention
//! block.
//!
//! ## Gemma 4 E2B specifics
//!
//! Getting E2B green required four fixes on top of the baseline 31B
//! path:
//!
//! - **Cross-layer KV sharing** — `num_kv_shared_layers=20` means layers
//!   15-34 reuse K/V computed by the last unshared sliding / full layer.
//!   We thread a `kv_cache: HashMap<usize, SharedKV>` through the loop
//!   (mirrors `predict_with_temperature`).
//! - **Per-Layer Embeddings (PLE)** — extraction writes the global PLE
//!   tensors (`per_layer_model_projection`, `embed_tokens_per_layer`)
//!   and the per-layer `per_layer_input_gate` / `per_layer_projection`
//!   into `ple_weights.bin` at **f16** (NOT Q4_K — the super-block
//!   calibration zeroes out embedding-style tensors). Load populates
//!   `weights.tensors` so `precompute_per_layer_inputs` and
//!   `apply_per_layer_embedding` can read them directly.
//! - **Double-wide MLP** — `use_double_wide_mlp=True` gives some layers
//!   `intermediate=12288` while the model-wide config reports 6144. Use
//!   `index.num_features(layer)` per-layer to size the FFN dequant;
//!   `weights.intermediate_size` is wrong for wide layers.
//! - **Final-logit softcap** — `final_logit_softcapping=30.0` must
//!   survive extract → vindex → load. Without it `logits_to_predictions`
//!   peaks on the wrong token; the cos-sim 0.99 uncapped distribution
//!   on E2B happened to argmax on "hyperparameters".
//!
//! Wire-in point: `walk --predict --index <q4 vindex>` in
//! `larql-cli/src/commands/extraction/walk_cmd.rs`.

use std::collections::HashMap;

use ndarray::Array2;
use tokenizers::Tokenizer;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::PredictResult;
use crate::forward::run_layer_with_ffn;

/// End-to-end predict on a Q4_K/Q6_K vindex.
///
/// `weights` must carry norms + embed + lm_head but is allowed — and
/// expected — to have empty attn / FFN tensor entries; this function
/// fills them in per layer from the vindex. Returns the top-k next-token
/// predictions in the same shape as `larql_inference::predict`.
pub fn predict_q4k(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    // NOTE: don't use `weights.intermediate_size` — Gemma 4 E2B has
    // `use_double_wide_mlp=True`, so half the layers (15-34) actually
    // ship with intermediate=12288 while `weights.intermediate_size`
    // reports the baseline 6144. Ask the index per layer instead.

    let mut h = embed_tokens_pub(weights, token_ids);

    // Per-Layer Embeddings + cross-layer KV-sharing — both used by
    // Gemma 4 E2B (PLE + last-20 layers reuse K/V from the preceding
    // unshared sliding/global layer). Mirrors `predict_with_temperature`.
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let dump_dir = std::env::var("LARQL_CPU_DUMP_LAYERS").ok();
    if let Some(ref dir) = dump_dir {
        let slice = h.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/cpu_h_embed.f32"), &bytes);
    }

    for layer in 0..num_layers {
        // ── Dequantise this layer's Q/K/V/O and gate/up/down ──
        let attn = index.attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));
        let ffn = index.interleaved_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("ffn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        // Per-layer intermediate size — 6144 on standard E2B layers,
        // 12288 on double-wide ones.
        let intermediate = index.num_features(layer);

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let gate_key = arch.ffn_gate_key(layer);
        let up_key = arch.ffn_up_key(layer);
        let down_key = arch.ffn_down_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
        let w_down = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate);

        // Insert into weights.tensors so the existing f32 forward paths
        // can find them. We own `&mut weights`, so this is direct.
        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());
        weights.tensors.insert(gate_key.clone(), w_gate.into_shared());
        weights.tensors.insert(up_key.clone(), w_up.into_shared());
        weights.tensors.insert(down_key.clone(), w_down.into_shared());

        // ── Run the layer — reuses the standard block so layer_scalar,
        //    per-layer embedding, and KV-sharing all apply identically to
        //    the float `predict_with_temperature` path.
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let ffn_backend = crate::ffn::WeightFfn { weights };
        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        // ── Drop this layer's f32 tensors before the next layer ──
        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
        weights.tensors.remove(&gate_key);
        weights.tensors.remove(&up_key);
        weights.tensors.remove(&down_key);

        // Optional per-layer residual dump matching the Metal shader's
        // LARQL_METAL_DUMP_LAYERS convention. Together these let us diff
        // CPU vs Metal layer-by-layer and bisect the first divergence.
        if let Some(ref dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = format!("{dir}/cpu_layer_{layer:02}.f32");
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("[dump] failed to write {path}: {e}");
            }
        }
    }

    crate::forward::predict::logits_to_predictions_pub(
        weights, &h, tokenizer, top_k, 1.0,
    )
}

/// Common end-of-turn / EOS markers across Gemma, Llama, Mistral, ChatML.
///
/// Used by [`generate_q4k_cpu`] to halt generation when the model emits any
/// of these. Catches a wider set than the raw EOS token id because chat
/// templates tend to use family-specific terminators.
pub fn is_end_of_turn(token: &str) -> bool {
    matches!(
        token,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|eot_id|>"
    )
}

#[cfg(test)]
mod end_of_turn_tests {
    use super::is_end_of_turn;

    #[test]
    fn recognises_known_terminators() {
        for t in [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<|eot_id|>",
        ] {
            assert!(is_end_of_turn(t), "expected {t:?} to be a terminator");
        }
    }

    #[test]
    fn does_not_match_arbitrary_tokens() {
        for t in ["", " ", "the", "<eos", "eos>", "<EOS>", "<|im_start|>"] {
            assert!(!is_end_of_turn(t), "did not expect {t:?} to be a terminator");
        }
    }
}

/// CPU autoregressive generation against a Q4_K / Q6_K vindex.
///
/// Loops [`predict_q4k`] one token at a time. Stops on `max_tokens` or when
/// the produced token text matches [`is_end_of_turn`]. Per-step cost is
/// O(N²) in context length (no KV cache) — the same trade-off
/// `larql dev walk --predict --max-tokens N` makes for the CPU path. For
/// long outputs use the Metal backend instead via
/// [`crate::layer_graph::generate`].
///
/// Returns `(token_text, token_id)` pairs in generation order.
pub fn generate_q4k_cpu(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
) -> Vec<(String, u32)> {
    let mut ids = prompt_ids.to_vec();
    let mut out: Vec<(String, u32)> = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let result = predict_q4k(weights, tokenizer, &ids, 1, index);
        let next_id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = is_end_of_turn(&tok);
        out.push((tok, next_id));
        ids.push(next_id);
        if stop {
            break;
        }
    }
    out
}

/// End-to-end predict on a Q4_K vindex with the FFN served by an external
/// [`FfnBackend`] — typically [`crate::ffn::RemoteWalkBackend`] for the
/// dense-remote demo where attention runs locally and each layer's FFN is
/// one HTTP round trip to an `larql serve --ffn-only` server.
///
/// Mirrors [`predict_q4k`] except: only attention Q/K/V/O are dequantised
/// per layer (FFN weights are never loaded client-side), and the per-layer
/// FFN step is delegated to the passed backend rather than `WeightFfn`.
/// Peak f32 heap drops from ~1.8 GB/layer to ~0.4 GB/layer on 31B.
pub fn predict_q4k_with_ffn(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    ffn_backend: &dyn crate::ffn::FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        // Attention Q/K/V/O only — FFN lives on the remote server.
        let attn = index.attn_q4k_layer_data(layer)
            .unwrap_or_else(|| panic!("attn Q4K slices missing for layer {layer}"));

        let arch = &*weights.arch;
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let head_dim = arch.head_dim_for_layer(layer);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;

        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);

        weights.tensors.insert(q_key.clone(), w_q.into_shared());
        weights.tensors.insert(k_key.clone(), w_k.into_shared());
        weights.tensors.insert(v_key.clone(), w_v.into_shared());
        weights.tensors.insert(o_key.clone(), w_o.into_shared());

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        weights.tensors.remove(&q_key);
        weights.tensors.remove(&k_key);
        weights.tensors.remove(&v_key);
        weights.tensors.remove(&o_key);
    }

    crate::forward::predict::logits_to_predictions_pub(
        weights, &h, tokenizer, top_k, 1.0,
    )
}

/// End-to-end predict on a Q4_K vindex driven by a Metal (or any Q4-capable)
/// `ComputeBackend`. Prompt tokens are fed through `backend.decode_token` one
/// position at a time — each call reads the token's embedding, appends its K/V
/// to the per-layer cache, attends causally against positions 0..=pos, and
/// returns the post-residual hidden state. Logits come from the final
/// post-prompt position via the standard final-norm + lm_head path.
///
/// Gemma 4 31B's asymmetric geometry (sliding 16×256 / global 4×512) is
/// handled by calling `backend.preallocate_kv_cache_per_layer` with the
/// exact per-layer `(num_kv_heads, head_dim)` shapes before the first decode.
/// Without that preallocation the backend would lazily size the cache from
/// the first layer's dims and the global layers would read off the end of
/// under-sized buffers.
pub fn predict_q4k_metal(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
) -> PredictResult {
    use larql_compute::QuantFormat;
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;

    // ── Build FullPipelineLayer per layer ──
    // FFN weights come from interleaved_q4k_layer_data (manifest-driven
    // per-matrix layout). Attn weights come from resolve_attn_weights which
    // prefers the Q4K manifest. Norms/layer_scalar/etc come from the arch
    // + weights.vectors map populated by load_model_weights_q4k.
    let layers: Vec<_> = (0..num_layers).map(|layer| {
        let (wq, wk, wv, wo) = resolve_attn_weights(index, layer)
            .expect("attn Q4K slices missing for layer");
        let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] =
            index.interleaved_q4k_layer_data(layer)
                .expect("ffn Q4K slices missing for layer");
        fn to_format(s: &str) -> QuantFormat {
            match s { "Q6_K" => QuantFormat::Q6_K, _ => QuantFormat::Q4_K }
        }
        let gate = larql_compute::QuantWeight { data: gate_bytes, scales: None, format: to_format(gate_fmt) };
        let up   = larql_compute::QuantWeight { data: up_bytes,   scales: None, format: to_format(up_fmt) };
        let down = larql_compute::QuantWeight { data: down_bytes, scales: None, format: to_format(down_fmt) };
        build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
    }).collect();

    // ── Preallocate KV cache with correct per-layer shapes ──
    let max_seq = token_ids.len().max(64);
    let shapes: Vec<(usize, usize)> = layers.iter()
        .map(|l| (l.num_kv_heads, l.head_dim))
        .collect();
    backend.preallocate_kv_cache_per_layer(&shapes, max_seq);
    backend.reset_kv_cache();

    // ── Run decode one token at a time, building up KV cache ──
    let hidden = weights.hidden_size;
    let embed = &weights.embed;
    let embed_scale = arch.embed_scale();

    let q_dim_first = layers[0].num_q_heads * layers[0].head_dim;
    let kv_dim_first = layers[0].num_kv_heads * layers[0].head_dim;
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    let _ = (q_dim_first, kv_dim_first, qk_norm, softcap); // reserved for a future prefill path

    // decode_token processes one token position at a time, appending its K/V
    // to the per-layer cache and attending causally against positions 0..=pos.
    // We feed the prompt tokens through it one by one to build the cache, then
    // the final residual is the prediction-time hidden state.
    //
    // Each decode_token call takes the FIRST layer's dims as the outer
    // scalar shape; the per-layer FullPipelineLayer inside drives the actual
    // geometry. This works even on Gemma 4 31B because the scratch buffers
    // inside decode_token are now sized to max(layer.q_dim) / max(layer.kv_dim).
    let dims_q = layers[0].num_q_heads * layers[0].head_dim;
    let dims_kv = layers[0].num_kv_heads * layers[0].head_dim;

    let mut h_vec: Vec<f32> = Vec::with_capacity(hidden);
    for &tok in token_ids {
        let row = embed.row(tok as usize);
        let x: Vec<f32> = row.iter().map(|v| v * embed_scale).collect();

        let out = backend
            .decode_token(
                &layers, &x,
                hidden, weights.intermediate_size,
                dims_q, dims_kv,
                layers[0].num_q_heads, layers[0].num_kv_heads, layers[0].head_dim,
                layers[0].rope_base,
            )
            .expect("backend doesn't support decode_token — need Metal with Q4 kernels");
        h_vec = out;
    }

    // ── Final norm + lm_head over the last position's residual ──
    let h_last = ndarray::Array2::from_shape_vec((1, hidden), h_vec)
        .expect("residual shape");
    crate::forward::predict::logits_to_predictions_pub(
        weights, &h_last, tokenizer, top_k, 1.0,
    )
}

/// Run one layer's FFN forward on a Q4_K vindex — dequantise gate/up/down
/// for just this layer and apply the architecture's activation gate.
///
/// Used by `larql-server`'s `/v1/walk-ffn` (full_output mode) when serving
/// a Q4_K vindex: the FFN weights aren't materialised into `ModelWeights.tensors`
/// at startup (would cost ~120 GB f32 on 31B), so we dequantise per-request
/// per-layer. Working-set is ~3 GB on 31B (one layer's gate+up+down f32).
///
/// Requires `index.load_interleaved_q4k()` to have been called; panics
/// otherwise.
pub fn q4k_ffn_forward_layer(
    arch: &dyn larql_models::ModelArchitecture,
    index: &VectorIndex,
    layer: usize,
    x: &Array2<f32>,
) -> Array2<f32> {
    use crate::forward::dot_proj;
    use crate::ffn::{silu_gate_up, gelu_tanh_gate_up};

    let hidden = x.shape()[1];
    let intermediate = index.num_features(layer);

    let ffn = index.interleaved_q4k_layer_data(layer).unwrap_or_else(|| {
        panic!(
            "interleaved_q4k layer data missing for layer {layer} — \
             server must call `load_interleaved_q4k` before serving walk-ffn"
        )
    });

    let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
    let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
    let w_down = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate);

    let gate = dot_proj(x, &w_gate);
    let up = dot_proj(x, &w_up);
    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
            gelu_tanh_gate_up(&gate, &up)
        }
        _ => silu_gate_up(&gate, &up),
    };
    dot_proj(&activation, &w_down)
}

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// The on-disk layout (`rows × cols` elements) must be stored contiguously
/// row-major and padded to a multiple of 256 elements per the k-quant
/// super-block size. Formats other than `Q4_K`/`Q6_K` panic — callers have
/// already dispatched on format so the default arm is unreachable.
fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let floats = match format {
        "Q4_K" => larql_models::quant::ggml::dequantize_q4_k(bytes, padded)
            .expect("Q4_K dequant failed"),
        "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded)
            .expect("Q6_K dequant failed"),
        other => panic!("unsupported quant format in vindex: {other}"),
    };
    let truncated = if floats.len() > n { floats[..n].to_vec() } else { floats };
    Array2::from_shape_vec((rows, cols), truncated)
        .expect("shape mismatch dequantising Q4K matrix")
}
