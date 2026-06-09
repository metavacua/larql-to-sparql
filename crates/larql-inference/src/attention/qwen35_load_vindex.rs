//! Vindex → `Qwen35Weights` adapter — Phase 2 of `vindex-qwen35moe-reader`.
//!
//! Mirrors [`super::qwen35_load::load_qwen35_weights`] (which loads
//! from a GGUF mmap via `larql-models::load_gguf`) but reads from a
//! vindex directory produced by `larql convert gguf-to-vindex
//! --quant q4k`: the writer side shipped in PR #147 + the router
//! fixes in PRs #152 / #155.
//!
//! This file lands the **module skeleton** + canonical error type +
//! public signature. The per-layer assembly bodies (DeltaNet /
//! full-attn / MoE 256-expert packing) are scoped out as tasks
//! 2b.2..2b.4 in
//! `openspec/changes/vindex-qwen35moe-reader/step-2b-design.md` and
//! left as a `todo!()` that names the design-doc breadcrumb.
//!
//! Why ship the stub now: getting the module wiring + imports +
//! signature right in isolation lets the assembly PRs be pure
//! per-layer logic, not file-skeleton churn.

use std::path::Path;
use std::time::Instant;

use larql_models::quant::ggml::{dequantize, TYPE_F32, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0};
use larql_models::quant::lazy::QuantTensor;
use larql_models::{ModelArchitecture, ModelWeights};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::deltanet_block::DeltaNetDims;
use crate::attention::qwen35_block::{DeltaNetHybridCache, Qwen35AttentionDims};
use crate::attention::qwen35_forward::{qwen35_forward_step, Qwen35Weights};
use crate::layer_graph::generate::{
    Detokenizer, EosConfig, GenerateResult, Sampler, SamplingConfig,
};

/// Errors surfaced by [`load_qwen35_weights_from_vindex`].
#[derive(Debug, thiserror::Error)]
pub enum VindexLoadError {
    #[error("vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),

    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),

    #[error(
        "vindex {0:?} reports arch `{1}`, but only `qwen35` and `qwen35moe` reach this loader"
    )]
    UnexpectedArch(std::path::PathBuf, String),

    #[error(
        "tensor `{key}` not found in vindex {dir:?} — was the vindex built with PR #147+ and \
         the router-write fixes from PRs #152/#155?"
    )]
    MissingTensor {
        dir: std::path::PathBuf,
        key: String,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Reconstruct a `Qwen35Weights` from a vindex directory.
///
/// **Status: stub** — task 2b.5 of `vindex-qwen35moe-reader`. The
/// public signature is final; the body is a `todo!()` until tasks
/// 2b.2..2b.4 (per-layer assembly) land. Callers SHOULD treat any
/// successful compile as a forward-compat check only — calling this
/// function at runtime panics with a breadcrumb to the design doc.
pub fn load_qwen35_weights_from_vindex(
    vindex_dir: &Path,
) -> Result<Qwen35Weights, VindexLoadError> {
    // Sanity-check the arch so the next session's runtime tests at
    // least see a clean refusal on non-qwen35 vindexes.
    let config = larql_vindex::load_vindex_config(vindex_dir)?;
    let arch_family = config
        .model_config
        .as_ref()
        .map(|c| c.model_type.clone())
        .unwrap_or_default();
    if arch_family != "qwen35" && arch_family != "qwen35moe" && arch_family != "qwen3next" {
        return Err(VindexLoadError::UnexpectedArch(
            vindex_dir.to_path_buf(),
            arch_family,
        ));
    }

    // Manifest-driven weights — populates `vectors` (norms +
    // DeltaNet small tensors + MoE router via PR #155) but NOT the
    // `quant_tensors` map the qwen35 forward expects. The three
    // bridges below fill that gap.
    let mut callbacks = larql_vindex::index::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_q4k_shard(vindex_dir, &mut callbacks, None)?;

    // VectorIndex for byte-level access to the Q4_K matmul tensors
    // (DeltaNet attn_qkv/gate/alpha/beta/out and full-attn Q/K/V/O).
    let mut idx = larql_vindex::VectorIndex::load_vindex(vindex_dir, &mut callbacks)?;
    idx.load_attn_q4k(vindex_dir)?;
    idx.load_deltanet_q4k(vindex_dir)?;

    // Temporarily swap the arch out of `weights` so the bridge calls
    // can simultaneously borrow `&*arch` and `&mut weights.*` without
    // tripping the borrow checker on `&weights.arch`. The placeholder
    // is a trivial `GenericArch` that's never used — `weights.arch`
    // gets restored before `load_qwen35_weights` is invoked.
    let placeholder = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "generic",
        "hidden_size": 1,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
    }));
    let arch = std::mem::replace(&mut weights.arch, placeholder);

    // `ssm_conv1d` is written into `norms.bin` as a flattened VECTOR
    // entry but `qwen35_load::load_deltanet_layer` reads it via
    // `get_tensor` (2-D from `weights.tensors`). Reshape every linear
    // layer's flat conv1d vector back to the GGUF wire shape
    // `[conv_dim, d_conv]` so the consumer's standard dim-swap +
    // transpose produces the `[d_conv, conv_dim]` per-tap-channel
    // layout the forward expects (per `qwen35_load.rs:322-340`).
    let d_conv = arch.ssm_conv_kernel();
    for layer in 0..weights.num_layers {
        if !arch.is_linear_attention_layer(layer) {
            continue;
        }
        let Some(key) = arch.ssm_conv1d_key(layer) else {
            continue;
        };
        let Some(flat) = weights.vectors.get(&key).cloned() else {
            continue;
        };
        if d_conv == 0 || flat.len() % d_conv != 0 {
            continue;
        }
        let conv_dim = flat.len() / d_conv;
        let arr = Array2::from_shape_vec((conv_dim, d_conv), flat).map_err(|e| {
            VindexLoadError::Vindex(larql_vindex::VindexError::Parse(e.to_string()))
        })?;
        weights.tensors.insert(key, arr.into_shared());
    }

    populate_deltanet_quant_tensors(&idx, &*arch, &mut weights)?;
    populate_attn_quant_tensors(&idx, &*arch, &mut weights)?;
    populate_moe_quant_tensors(vindex_dir, &*arch, &mut weights)?;
    populate_shexp_quant_tensors(vindex_dir, &mut weights)?;

    // Pad embed up to lm_head's vocab dim when the vindex writer
    // truncated embed to `logical_vocab` (PR #137) but kept lm_head
    // at the GGUF-original padded vocab. Special tokens
    // (e.g. `<|im_start|>` at id 248045 on Qwen 3.6) live in the
    // padded range; without padding the embed row lookup panics
    // with "index < dim" when the chat template renders them.
    //
    // Zero-padding is wrong-but-not-fatal: the special-token row
    // contributes a zero residual, which makes the model not see
    // its own template markers. Output quality suffers but the
    // server doesn't panic — and the lm_head logits stay valid
    // (vocabsel works because lm_head has the real rows). A proper
    // fix lands in the vindex writer (preserve the full padded
    // embed); this padding keeps the server runnable in the
    // meantime.
    let embed_rows = weights.embed.shape()[0];
    // Tokenizers ship special tokens at IDs beyond `vocab_size` (Qwen
    // 3.6 has ~26 added tokens at 248045-248069 above vocab=248044).
    // Both the vindex embed AND lm_head writers truncated to the
    // logical vocab per PR #137 (`format/weights/load.rs:770-784`),
    // so without padding the embed row lookup panics on the first
    // special token the chat template renders.
    //
    // Round up to a 256 alignment + 256 safety margin → covers up to
    // ~510 added tokens, plenty for any tokenizer in scope.
    let target_rows = embed_rows.div_ceil(256) * 256 + 256;
    if target_rows > embed_rows {
        let hidden = weights.embed.shape()[1];
        let mut new_arr = ndarray::Array2::<f32>::zeros((target_rows, hidden));
        let embed_arr = weights.embed.as_array();
        let src_rows = embed_arr.shape()[0];
        new_arr
            .slice_mut(ndarray::s![..src_rows, ..])
            .assign(&*embed_arr);
        drop(embed_arr);
        weights.embed = larql_models::embed::EmbedMatrix::from_array(new_arr.into_shared());
    }

    // Restore the real arch and delegate to the existing
    // data-source-agnostic loader. From this point the call graph is
    // identical to the GGUF path — every field of the produced
    // `Qwen35Weights` is populated by `qwen35_load::load_qwen35_weights`.
    weights.arch = arch;
    crate::attention::qwen35_load::load_qwen35_weights(&weights, &*weights.arch)
        .map_err(|e| VindexLoadError::Vindex(larql_vindex::VindexError::Parse(e.to_string())))
}

/// **Task 2b.2** — bridge DeltaNet Q4_K bytes into a `ModelWeights`.
///
/// Inserts the three lazy-quant matmul tensors (`attn_qkv`,
/// `attn_gate`, `ssm_out`) into `weights.quant_tensors` so the
/// existing GGUF-side `load_qwen35_weights` (which is data-source
/// agnostic) finds them via `weights.quant_tensors.get(&key)`. The
/// two smaller matmuls without a `*_quant` slot in
/// `DeltaNetLayerWeights` (`ssm_alpha`, `ssm_beta`) get dequantised
/// to f32 and inserted into `weights.tensors`.
///
/// No-op when the vindex carries no DeltaNet bytes
/// (`idx.has_deltanet_q4k() == false`) — every non-hybrid arch
/// falls through cleanly.
///
/// Looks up arch keys via `attn_qkv_key` / `attn_gate_key` /
/// `ssm_alpha_key` / `ssm_beta_key` / `ssm_out_key`. The Qwen35 arch
/// handler returns the **HF-normalised** form
/// (`layers.{L}.self_attn.qkv_proj.weight` etc) which matches the
/// writer-side keys recorded by PR #147's manifest. The historical
/// router-key mismatch (PR #155 / task 2b.0b) does NOT affect
/// DeltaNet matmuls — the GGUF→HF remap table covers `attn_qkv.` →
/// `self_attn.qkv_proj.`, so there's no naming-skew gotcha here.
pub fn populate_deltanet_quant_tensors(
    idx: &VectorIndex,
    arch: &dyn ModelArchitecture,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    if !idx.has_deltanet_q4k() {
        return Ok(());
    }
    let n_layers = weights.num_layers;
    for layer in 0..n_layers {
        if !arch.is_linear_attention_layer(layer) {
            continue;
        }
        let Some(slots) = idx.deltanet_q4k_layer_data(layer) else {
            continue;
        };

        // Fixed-order keys mirror the writer's tensor enumeration
        // in `write_q4k::deltanet::write_deltanet_weights_q4k`.
        let keys: [Option<String>; 5] = [
            arch.attn_qkv_key(layer),
            arch.attn_gate_key(layer),
            arch.ssm_alpha_key(layer),
            arch.ssm_beta_key(layer),
            arch.ssm_out_key(layer),
        ];

        for (i, (bytes, fmt, shape)) in slots.iter().enumerate() {
            let Some(key) = keys[i].as_ref() else {
                continue;
            };
            if shape.len() != 2 {
                continue;
            }
            let (rows, cols) = (shape[0], shape[1]);
            let tensor_type = match *fmt {
                "Q4_K" => TYPE_Q4_K,
                "Q5_K" => TYPE_Q5_K,
                "Q6_K" => TYPE_Q6_K,
                "Q8_0" => TYPE_Q8_0,
                _ => continue,
            };

            // Three tensors have a `*_quant` slot in
            // `DeltaNetLayerWeights` (attn_qkv / attn_gate / ssm_out
            // — indices 0, 1, 4). The other two (ssm_alpha, ssm_beta
            // — indices 2, 3) are dense-only on the consumer side,
            // so dequantise to f32 and feed `weights.tensors`. The
            // existing `load_qwen35_weights` looks them up via
            // `get_tensor` which only reads `weights.tensors`.
            let is_dense_only = matches!(i, 2 | 3);
            if is_dense_only {
                let floats = dequantize(bytes, tensor_type, rows * cols)?;
                let arr = Array2::from_shape_vec((rows, cols), floats).map_err(|e| {
                    VindexLoadError::Vindex(larql_vindex::VindexError::Parse(e.to_string()))
                })?;
                weights.tensors.insert(key.clone(), arr.into_shared());
            } else {
                let qt = QuantTensor::from_raw(bytes.to_vec(), tensor_type, rows, cols)?;
                weights.quant_tensors.insert(key.clone(), qt);
            }
        }
    }
    Ok(())
}

/// **Task 2b.3b** — bridge full-attn Q/K/V/O vindex bytes into a
/// `ModelWeights`. All 4 projections have `*_quant` slots in
/// `Qwen35AttentionLayerWeights`, so every tensor lands in
/// `weights.quant_tensors` — no dequant fallback needed (unlike
/// the DeltaNet bridge's `ssm_alpha`/`ssm_beta` dense-only case).
///
/// Walks every layer L where `arch.is_linear_attention_layer(L) ==
/// false` and pulls 4 (bytes, fmt, shape) tuples via the sparse-
/// aware accessor from PR #163 (`attn_q4k_sparse_layer_data`). The
/// existing dense accessor's `layer * 4` arithmetic would read
/// wrong bytes for hybrid arches whose manifest only carries
/// entries for the 10 full-attention layers.
///
/// V is typically Q6_K (writer's higher-precision choice for the
/// value projection); Q/K/O are Q4_K. The fmt tag string per
/// tensor determines the `tensor_type` constant passed to
/// `QuantTensor::from_raw`.
pub fn populate_attn_quant_tensors(
    idx: &VectorIndex,
    arch: &dyn ModelArchitecture,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    let n_layers = weights.num_layers;
    for layer in 0..n_layers {
        if arch.is_linear_attention_layer(layer) {
            continue;
        }
        let Some(slots) = idx.attn_q4k_sparse_layer_data(layer) else {
            continue;
        };

        // Fixed-order keys mirror the writer's tensor enumeration
        // in `write_q4k::attn::write_attn_weights_q4k`.
        let keys: [String; 4] = [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ];

        for (i, (bytes, fmt, shape)) in slots.iter().enumerate() {
            if shape.len() != 2 {
                continue;
            }
            let (rows, cols) = (shape[0], shape[1]);
            let tensor_type = match *fmt {
                "Q4_K" => TYPE_Q4_K,
                "Q5_K" => TYPE_Q5_K,
                "Q6_K" => TYPE_Q6_K,
                "Q8_0" => TYPE_Q8_0,
                _ => continue,
            };
            let qt = QuantTensor::from_raw(bytes.to_vec(), tensor_type, rows, cols)?;
            weights.quant_tensors.insert(keys[i].clone(), qt);
        }
    }
    Ok(())
}

/// **Task 2b.4** — bridge MoE per-layer expert files into a
/// `ModelWeights`. Reads each `layers/layer_LL.weights` file
/// produced by PR #147's writer, parses the LYRW header to learn
/// `(num_experts, inter, hidden)` + per-expert offsets, then
/// **concatenates** all 256 experts' bytes into three packed
/// QuantTensors matching the shape `Qwen35MoeFfnWeights` expects:
///
/// - `ffn_gate_exps`: `[num_experts * inter, hidden]`
/// - `ffn_up_exps`:   `[num_experts * inter, hidden]`
/// - `ffn_down_exps`: `[num_experts * hidden, padded_inter]`
///
/// The writer's `quantize_dense_entry` interleaves each expert's
/// gate + up rows into one `[2*inter, hidden]` tensor (gate rows
/// first, then up rows). The bridge splits at the row-midpoint
/// (always safe for block-quantised tensors — row boundaries
/// align to super-block boundaries by construction).
///
/// The MoE router (`ffn_gate_inp.weight`) lives in
/// `weights.vectors` (PR #155 fix), but the consumer
/// `load_qwen35_moe_ffn` looks it up in `weights.quant_tensors`.
/// The bridge therefore wraps the f32 router buffer in a
/// `QuantTensor` of `TYPE_F32` and inserts it under the arch's
/// router key.
///
/// Shared-expert tensors (`ffn_*_shexp.weight`) are not in scope
/// for this PR — `load_qwen35_moe_ffn` will return `None` for
/// them, and the qwen35moe forward currently treats `shexp` as
/// optional. A follow-up PR adds them once the writer side starts
/// emitting them.
pub fn populate_moe_quant_tensors(
    vindex_dir: &Path,
    arch: &dyn ModelArchitecture,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    if !arch.is_moe() {
        return Ok(());
    }
    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;
    if num_experts == 0 || moe_inter == 0 || hidden == 0 {
        return Ok(());
    }

    for layer in 0..weights.num_layers {
        // ── Router: f32 in `vectors`, wrap as TYPE_F32 QuantTensor ──
        if let Some(router_key) = arch.moe_router_key(layer) {
            if let Some(router_floats) = weights.vectors.get(&router_key) {
                if router_floats.len() == num_experts * hidden {
                    let bytes: Vec<u8> =
                        router_floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                    let qt = QuantTensor::from_raw(bytes, TYPE_F32, num_experts, hidden)?;
                    weights.quant_tensors.insert(router_key.clone(), qt);
                }
            }
        }

        // ── Per-layer expert file ──
        let path = vindex_dir.join(format!("layers/layer_{layer:02}.weights"));
        let Ok(file) = std::fs::File::open(&path) else {
            // qwen35moe vindexes from PR #147 always have these; skip
            // gracefully if absent (test fixtures, partial vindexes).
            continue;
        };
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let (fmt, n_entries, inter, _h, offsets) =
            larql_vindex::format::weights::write_layers::parse_layer_weights_header(&mmap)
                .ok_or_else(|| {
                    VindexLoadError::Vindex(larql_vindex::VindexError::Parse(format!(
                        "layers/layer_{layer:02}.weights: malformed LYRW header"
                    )))
                })?;
        if n_entries != num_experts || inter != moe_inter {
            return Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                format!(
                    "layers/layer_{layer:02}.weights: header says \
                     num_entries={n_entries}, inter={inter}; arch expects \
                     num_experts={num_experts}, moe_intermediate={moe_inter}"
                ),
            )));
        }
        // Per-projection format support (LYRW v2): all entries in a
        // layer file share the same gate_up/down formats (the writer
        // makes the decision per-source-format, not per-expert), but
        // gate_up and down themselves may differ — Coder-Next ships
        // Q4_K gate/up + Q6_K down across all experts. Read the first
        // entry's overrides (or fall back to the file-level `fmt` for
        // LYRW v1 files where overrides are None).
        let layer_fmt_to_ttype =
            |lf: larql_vindex::format::weights::write_layers::LayerWeightFormat| match lf {
                larql_vindex::format::weights::write_layers::LayerWeightFormat::Q4_K => {
                    Ok(TYPE_Q4_K)
                }
                larql_vindex::format::weights::write_layers::LayerWeightFormat::Q5_K => {
                    Ok(TYPE_Q5_K)
                }
                larql_vindex::format::weights::write_layers::LayerWeightFormat::Q6_K => {
                    Ok(TYPE_Q6_K)
                }
                other => Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    format!("layers/layer_{layer:02}.weights: format {other:?} unsupported"),
                ))),
            };
        let first = offsets.first().ok_or_else(|| {
            VindexLoadError::Vindex(larql_vindex::VindexError::Parse(format!(
                "layers/layer_{layer:02}.weights: no entries"
            )))
        })?;
        let gate_up_fmt = first.gate_up_format;
        let down_fmt = first.down_format;
        let gate_up_ttype = layer_fmt_to_ttype(gate_up_fmt)?;
        let down_ttype = layer_fmt_to_ttype(down_fmt)?;
        // Sanity: assert all entries agree (the writer guarantees this,
        // but a future caller that doesn't follow the convention should
        // fail loudly).
        for row in &offsets {
            if row.gate_up_format != gate_up_fmt || row.down_format != down_fmt {
                return Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    format!(
                        "layers/layer_{layer:02}.weights: per-entry format drift not yet supported"
                    ),
                )));
            }
        }
        let _ = fmt; // file-level default — informational only when v2 carries overrides

        // Concat all `num_experts` experts' bytes. The writer guarantees
        // `gate_up_bytes` is `[2 * inter, hidden]` row-major
        // (gate rows then up rows), so a midpoint byte cut is safe —
        // row boundaries align to super-block boundaries for any
        // block-quantised format we support.
        let mut packed_gate: Vec<u8> = Vec::new();
        let mut packed_up: Vec<u8> = Vec::new();
        let mut packed_down: Vec<u8> = Vec::new();
        let mut down_len_per_expert: usize = 0;
        for row in &offsets {
            let gu_off = row.gate_up_offset;
            let gu_len = row.gate_up_len;
            let dn_off = row.down_offset;
            let dn_len = row.down_len;
            let gu_end = gu_off.checked_add(gu_len).ok_or_else(|| {
                VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    "expert gate_up byte range overflows usize".into(),
                ))
            })?;
            let dn_end = dn_off.checked_add(dn_len).ok_or_else(|| {
                VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    "expert down byte range overflows usize".into(),
                ))
            })?;
            if gu_end > mmap.len() || dn_end > mmap.len() {
                return Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    format!("layers/layer_{layer:02}.weights: expert byte range past EOF"),
                )));
            }
            let half = gu_len / 2;
            packed_gate.extend_from_slice(&mmap[gu_off..gu_off + half]);
            packed_up.extend_from_slice(&mmap[gu_off + half..gu_end]);
            packed_down.extend_from_slice(&mmap[dn_off..dn_end]);
            down_len_per_expert = dn_len;
        }

        // `down` per expert is `[hidden, padded_inter]`. Derive
        // padded_inter from byte count: bytes_per_block × blocks_per_row × hidden.
        // Use down's tensor_type (not gate_up's) — they may differ.
        let block_elems = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let bytes_per_block = match down_fmt {
            larql_vindex::format::weights::write_layers::LayerWeightFormat::Q4_K => {
                larql_models::quant::ggml::Q4_K_BLOCK_BYTES
            }
            larql_vindex::format::weights::write_layers::LayerWeightFormat::Q5_K => {
                larql_models::quant::ggml::Q5_K_BLOCK_BYTES
            }
            larql_vindex::format::weights::write_layers::LayerWeightFormat::Q6_K => {
                larql_models::quant::ggml::Q6_K_BLOCK_BYTES
            }
            _ => unreachable!("format checked above"),
        };
        let blocks_per_row = down_len_per_expert / hidden / bytes_per_block;
        let padded_inter = blocks_per_row * block_elems;

        let prefix = format!("layers.{layer}.");
        let gate_qt =
            QuantTensor::from_raw(packed_gate, gate_up_ttype, num_experts * inter, hidden)?;
        let up_qt = QuantTensor::from_raw(packed_up, gate_up_ttype, num_experts * inter, hidden)?;
        let down_qt =
            QuantTensor::from_raw(packed_down, down_ttype, num_experts * hidden, padded_inter)?;
        weights
            .quant_tensors
            .insert(format!("{prefix}ffn_gate_exps.weight"), gate_qt);
        weights
            .quant_tensors
            .insert(format!("{prefix}ffn_up_exps.weight"), up_qt);
        weights
            .quant_tensors
            .insert(format!("{prefix}ffn_down_exps.weight"), down_qt);
    }
    Ok(())
}

/// Populate `weights.quant_tensors` with per-layer shared-expert
/// projections from `shexp_weights_q4k.bin` + manifest. No-op when
/// the file is absent (dense / non-shexp arch).
///
/// The downstream `load_qwen35_moe_ffn` reads keys exactly as
/// `layers.{L}.ffn_{gate,up,down}_shexp.weight` — the writer records
/// these keys, the loader inserts under the same names.
pub fn populate_shexp_quant_tensors(
    vindex_dir: &Path,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    let bin_path = vindex_dir.join(larql_vindex::format::filenames::SHEXP_WEIGHTS_Q4K_BIN);
    let manifest_path =
        vindex_dir.join(larql_vindex::format::filenames::SHEXP_WEIGHTS_Q4K_MANIFEST_JSON);
    if !bin_path.exists() || !manifest_path.exists() {
        return Ok(()); // no shexp — file-absent is a valid "no shared expert" signal
    }

    let bytes = std::fs::read(&bin_path)?;
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let manifest: Vec<larql_vindex::format::weights::Q4kManifestEntry> =
        serde_json::from_str(&manifest_text).map_err(|e| {
            VindexLoadError::Vindex(larql_vindex::VindexError::Parse(e.to_string()))
        })?;

    for entry in &manifest {
        let off = entry.offset as usize;
        let len = entry.length as usize;
        let end = off.checked_add(len).ok_or_else(|| {
            VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                "shexp manifest entry length overflows usize".into(),
            ))
        })?;
        if end > bytes.len() {
            return Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                format!(
                    "shexp entry {:?} byte range past EOF (off={off} len={len} file={})",
                    entry.key,
                    bytes.len(),
                ),
            )));
        }
        if entry.shape.len() != 2 {
            continue;
        }
        let (rows, cols) = (entry.shape[0], entry.shape[1]);
        let tensor_type = match entry.format_tag() {
            "Q4_K" => TYPE_Q4_K,
            "Q5_K" => TYPE_Q5_K,
            "Q6_K" => TYPE_Q6_K,
            "Q8_0" => TYPE_Q8_0,
            "MXFP4" => larql_models::quant::ggml::TYPE_MXFP4,
            other => {
                return Err(VindexLoadError::Vindex(larql_vindex::VindexError::Parse(
                    format!("shexp entry {:?}: unsupported format {other}", entry.key),
                )));
            }
        };
        let slice = bytes[off..end].to_vec();
        let qt = QuantTensor::from_raw(slice, tensor_type, rows, cols)?;
        weights.quant_tensors.insert(entry.key.clone(), qt);
    }

    Ok(())
}

/// **Task 2c.a** — multi-token decode driver for qwen35 vindexes.
///
/// Mirrors [`crate::layer_graph::generate_with_sampling`]'s API but
/// drives the qwen35 hybrid forward (`qwen35_forward_step` +
/// `DeltaNetHybridCache`) instead of the standard transformer
/// `predict_q4k_hidden_with_cache` + `KvCache` path.
///
/// ## Flow
///
/// 1. Build dims (`DeltaNetDims`, `Qwen35AttentionDims`) from the arch
///    via the `from_arch` helpers shipped in PR #172.
/// 2. Allocate a fresh `DeltaNetHybridCache` for the sequence.
/// 3. **Prefill** — feed every prompt token through
///    `qwen35_forward_step`; the final token's logits are the first
///    decode-step output.
/// 4. **Decode loop** — sample → push to detokeniser → check EOS →
///    advance state. Stop on `eos.is_eos(id, decoded_text)` match or
///    `max_tokens` exhaustion.
///
/// Returns a `GenerateResult` with `tokens: Vec<(text, prob)>` —
/// `prob` is 1.0 for greedy decodes (no Sampler-internal posterior
/// reconstruction); set by the upstream sampler when temperature
/// sampling is configured.
pub fn qwen35_generate_with_sampling(
    weights: &Qwen35Weights,
    arch: &dyn ModelArchitecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult {
    let eps = 1e-6_f32;
    let dn_dims = DeltaNetDims::from_arch(arch, eps);
    let attn_dims = Qwen35AttentionDims::from_arch(arch, eps);

    if std::env::var("LARQL_QWEN35_MOE_DUMP").is_ok() {
        eprintln!(
            "qwen35_generate: arch family={} is_moe={} n_experts={} top_k={} moe_inter={} weights.layers[0].moe.is_some={}",
            arch.family(),
            arch.is_moe(),
            arch.num_experts(),
            arch.num_experts_per_token(),
            arch.moe_intermediate_size(),
            weights.layers.first().map(|l| l.moe.is_some()).unwrap_or(false),
        );
    }

    if prompt_ids.is_empty() {
        return GenerateResult::empty_error(crate::layer_graph::generate::GenerateError::Other {
            reason: "prompt_ids is empty".into(),
        });
    }

    let num_layers = weights.layers.len();
    let layer_kinds: Vec<bool> = (0..num_layers)
        .map(|l| arch.is_linear_attention_layer(l))
        .collect();
    let mut cache = DeltaNetHybridCache::allocate(
        &layer_kinds,
        attn_dims.kv_dim(),
        dn_dims.d_conv,
        dn_dims.conv_dim(),
        dn_dims.head_v_dim,
        dn_dims.n_v_heads,
    );

    // Prefill — only the last token's logits matter for the first
    // decode step. Earlier tokens advance the cache (KV slabs +
    // DeltaNet recurrent state).
    // Phase 4a (long-context arc, batched-prefill sub-arc): the
    // prefill loop is now routed through `qwen35_forward_prefill`,
    // a single insertion point that future PRs will replace with
    // batched-kernel implementations. Today it's bit-equivalent to
    // the prior inline per-token loop.
    let prefill_start = Instant::now();
    let mut last_logits = crate::attention::qwen35_forward::qwen35_forward_prefill(
        prompt_ids, weights, &dn_dims, &attn_dims, &mut cache, eps,
    );
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut sampler = Sampler::new(sampling);
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(prompt_ids);

    let mut out: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms: Vec<f64> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    for _ in 0..max_tokens {
        let step_start = Instant::now();
        let Some(logits) = last_logits.take() else {
            break;
        };
        let Some(next_id) =
            sampler.sample_with_history(logits.as_slice().unwrap_or(&[]), &generated)
        else {
            // Sampler returned None — all logits non-finite or empty
            // distribution after filters. Surface as a clean stop.
            break;
        };
        let text = detok.push(next_id);
        // EOS may fire on either the token id or the decoded text
        // (e.g. `<|im_end|>` decodes to a stop-string match even when
        // the id isn't in `eos_token_ids`).
        if eos.is_eos(next_id, &text) {
            break;
        }
        out.push((text, 1.0));
        generated.push(next_id);
        last_logits = Some(qwen35_forward_step(
            next_id, weights, &dn_dims, &attn_dims, &mut cache, eps,
        ));
        decode_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
    }

    GenerateResult {
        tokens: out,
        prefill_ms,
        decode_ms,
        stage_timings: crate::layer_graph::generate::StageTimings::default(),
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-qwen35 vindex must be refused with `UnexpectedArch` —
    /// guards against the loader being mis-called from
    /// `larql-server`'s dispatch (Step 2c).
    ///
    /// Disabled until a tiny mock vindex fixture lands; the body
    /// references the right error variant so the import + signature
    /// stay verified at compile time.
    #[test]
    #[ignore = "needs a synthetic non-qwen35 vindex fixture; revisit when 2b.3 lands"]
    fn rejects_non_qwen35_arch() {
        let _ = load_qwen35_weights_from_vindex(Path::new("/dev/null"));
    }

    /// Compile-time check that the error type carries the expected
    /// variants. Catches drift if a future PR drops or renames any
    /// of the breadcrumb error cases the design doc references.
    #[test]
    fn error_variants_compile() {
        let dir = std::path::PathBuf::from("/tmp/no-such-vindex");
        let _e1 = VindexLoadError::UnexpectedArch(dir.clone(), "llama".into());
        let _e2 = VindexLoadError::MissingTensor {
            dir,
            key: "layers.0.ssm_norm.weight".into(),
        };
    }
}
