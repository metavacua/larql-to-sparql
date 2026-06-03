//! Q4_K / Q6_K streaming writer — separate from `write_f32` because
//! the Q4_K pipeline owns its own QuantBlockFormat manifest, padding
//! helpers, and per-tensor quantisation policy.
//!
//! Carved out of the monolithic `write.rs` in the 2026-04-25 reorg,
//! and re-decomposed in 2026-05-09 round-5 into one sibling per
//! emitted artefact:
//!
//! - [`attn`] — `attn_weights_q4k.bin` (+ manifest)
//! - [`ffn`] — `interleaved_q4k.bin` (+ opt `down_features_q4k.bin`)
//! - [`moe_layers`] — `layers/layer_{L:02}.weights` (hybrid MoE)
//! - [`norms`] — `norms.bin` (norms + MoE router/scales)
//! - [`ple`] — `ple_weights.bin` (Gemma 4 E2B PLE, f16)
//! - [`lm_head`] — `lm_head_q4.bin`
//!
//! The orchestrator below threads the running `Vec<WeightEntry>`
//! manifest through the norms → ple → lm_head trio, then emits a
//! single `weight_manifest.json` and patches `index.json`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{FfnLayout, VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::capabilities::{ensure_standard_attention_supported, SURFACE_Q4K_WEIGHT_WRITER};
use super::write_f32::WeightSource;

mod attn;
mod ffn;
mod lm_head;
mod moe_layers;
mod norms;
mod ple;

pub mod feature_major_down;

/// Per-block quantisation format for a single tensor in the Q4_K pipeline.
/// Serde writes / reads the literal strings `"Q4_K"` and `"Q6_K"` to match
/// llama.cpp / Ollama on-disk conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantBlockFormat {
    #[serde(rename = "Q4_K")]
    Q4K,
    #[serde(rename = "Q6_K")]
    Q6K,
}

/// Pad a row-major f32 buffer to the next multiple of 256 with zeros
/// (Q4_K/Q6_K super-blocks require length % 256 == 0).
///
/// Kept only for unit-test coverage of the flat-padding helper pattern;
/// production paths now use [`pad_rows_to_block`] since the shader reads
/// each row as a fixed number of super-blocks.
#[cfg(test)]
fn pad_to_block(data: &[f32]) -> Vec<f32> {
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded_len = data.len().div_ceil(block) * block;
    if padded_len == data.len() {
        data.to_vec()
    } else {
        let mut v = Vec::with_capacity(padded_len);
        v.extend_from_slice(data);
        v.resize(padded_len, 0.0);
        v
    }
}

/// Pad each row of a 2-D row-major matrix to the next multiple of 256 with
/// zeros. Returns `(padded_flat, padded_cols)`.
///
/// Why this exists: Q4_K/Q6_K super-blocks hold exactly 256 values, so the
/// Metal matvec shader computes `bytes_per_row = (cols / 256) * block_size`.
/// When `cols % 256 != 0` (e.g. Gemma 4 26B A4B's `intermediate_size=2112`),
/// flat-padding the whole tensor leaves row boundaries misaligned with
/// super-block boundaries and every row past row 0 reads wrong bytes. Per-row
/// padding realigns each row onto a super-block boundary at the cost of a
/// small storage overhead (the padding columns are zero and contribute
/// nothing to the dot product at dispatch time, provided the caller also
/// zero-pads the input vector to `padded_cols`).
pub(super) fn pad_rows_to_block(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    debug_assert_eq!(data.len(), rows * cols);
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded_cols = cols.div_ceil(block) * block;
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }
    let mut out = Vec::with_capacity(rows * padded_cols);
    let pad = padded_cols - cols;
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        out.extend_from_slice(row);
        out.extend(std::iter::repeat_n(0.0f32, pad));
    }
    (out, padded_cols)
}

/// Resolve the V tensor for a layer in the Q4_K writer.
///
/// When `v_proj` is absent from the source (e.g. Gemma 4 31B global
/// layers ship without one), fall back to K's tensor if the
/// architecture advertises `v_shares_k(layer) == true`. This keeps
/// the 4-per-layer attn manifest contiguous: each layer emits exactly
/// Q / K / V / O even when V physically reuses K's bytes.
pub(super) fn resolve_v_tensor<T: Clone>(
    v: Option<T>,
    k: &Option<T>,
    v_shares_k: bool,
) -> Option<T> {
    v.or_else(|| if v_shares_k { k.clone() } else { None })
}

/// Options for [`write_model_weights_q4k_with_opts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4kWriteOptions {
    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default `false`
    /// preserves the Ollama-compatible "Q4_K_M" mix (Q4_K for gate/up,
    /// Q6_K for down). Setting `true` uses Q4_K uniformly — saves ~30MB
    /// per layer on 31B (1.8GB total) and drops down matmul cost ~1.5-1.7×
    /// to match up-proj timings. Quantisation noise on the scatter-sum
    /// averages across the intermediate dimension; empirically close.
    pub down_q4k: bool,

    /// Emit `down_features_q4k.bin` alongside `interleaved_q4k.bin`.
    /// When set, the down weights are also stored in feature-major
    /// `[intermediate, hidden]` orientation (Q4_K/Q6_K matching
    /// `down_q4k`), so per-feature decode can skip the
    /// `q4k_ffn_layer` whole-layer dequant + transpose cache. Adds
    /// roughly the same disk footprint as the down portion of
    /// `interleaved_q4k.bin` (~14 MB / layer at Gemma 4B dims).
    /// Recommended for CPU sparse walk and grid/MoE workloads where
    /// the ~840 MB heap cache ceiling is the binding constraint.
    /// Default `false` so existing extracts don't grow on disk.
    pub feature_major_down: bool,
}

/// Write model weights in Q4_K/Q6_K format, zero f32 intermediate on disk.
///
/// Emits:
///   attn_weights_q4k.bin + attn_weights_q4k_manifest.json
///     — Q/K/O → Q4_K, V → Q6_K
///     — On layers where V reuses K (Gemma 4 31B global layers), the K
///       bytes are written into the V slot so 4-per-layer indexing stays
///       valid and downstream kernels reading V get K.
///   interleaved_q4k.bin
///     — [gate Q4_K | up Q4_K | down Q6_K] per layer, regular stride.
///     — With `down_q4k=true`: [gate | up | down] all Q4_K.
///   lm_head_q4.bin
///     — Q4_K of the output projection (falls back to embed_tokens when tied).
///   norms.bin (f32, unchanged from non-Q4 path).
///
/// The source's per-tensor f32 materialisation is transient — one tensor's
/// worth of heap (~350 MB peak on 31B global layer Q) quantised then dropped.
pub fn write_model_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_q4k_with_opts(source, dir, callbacks, Q4kWriteOptions::default())
}

/// Like [`write_model_weights_q4k`] but accepts a [`Q4kWriteOptions`] knob
/// to toggle the FFN down-proj quantisation format and the
/// feature-major-down emit.
pub fn write_model_weights_q4k_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: Q4kWriteOptions,
) -> Result<(), VindexError> {
    callbacks.on_stage(STAGE_MODEL_WEIGHTS_Q4K);
    let start = std::time::Instant::now();

    let arch = source.arch();
    ensure_standard_attention_supported(arch, SURFACE_Q4K_WEIGHT_WRITER)?;
    let num_layers = source.num_layers();

    attn::write_attn_weights_q4k(source, dir, num_layers, callbacks)?;
    ffn::write_interleaved_ffn_q4k(source, dir, num_layers, opts, callbacks)?;
    moe_layers::write_per_layer_moe_q4k(source, dir, num_layers)?;
    let mut entries = norms::write_norms_and_router(source, dir, num_layers)?;
    ple::write_ple_weights(source, dir, num_layers, &mut entries)?;
    lm_head::write_lm_head_q4k(source, dir, &mut entries)?;

    let manifest_json =
        serde_json::to_string_pretty(&entries).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(WEIGHT_MANIFEST_JSON), manifest_json)?;

    update_index_json(dir, source.arch())?;

    callbacks.on_stage_done(
        STAGE_MODEL_WEIGHTS_Q4K,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

/// Patch `index.json` after all weight artefacts have landed:
/// `has_model_weights=true`, `quant=Q4K`, optional `ffn_layout` for
/// hybrid MoE, and a refreshed `model_config` from the architecture.
fn update_index_json(
    dir: &Path,
    arch: &dyn larql_models::ModelArchitecture,
) -> Result<(), VindexError> {
    let config_path = dir.join(INDEX_JSON);
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.quant = crate::QuantFormat::Q4K;
    if arch.is_hybrid_moe() {
        config.ffn_layout = Some(FfnLayout::PerLayer);
    }
    config.model_config = Some(VindexModelConfig::from_arch(arch));

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;
    Ok(())
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    // ── resolve_v_tensor ──

    #[test]
    fn resolve_v_returns_v_when_present() {
        let k = Some(2);
        assert_eq!(resolve_v_tensor(Some(1), &k, false), Some(1));
        assert_eq!(
            resolve_v_tensor(Some(1), &k, true),
            Some(1),
            "v_shares_k must not override a present v"
        );
    }

    #[test]
    fn resolve_v_falls_back_to_k_when_v_shared() {
        let k = Some(42);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, true),
            Some(42),
            "Gemma 4 31B global-layer fallback"
        );
    }

    #[test]
    fn resolve_v_none_when_missing_and_not_shared() {
        let k = Some(7);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, false),
            None,
            "no v_proj + v_shares_k=false → tensor is genuinely absent"
        );
    }

    #[test]
    fn resolve_v_none_when_v_missing_and_k_missing() {
        let k: Option<i32> = None;
        assert_eq!(resolve_v_tensor(None, &k, true), None);
        assert_eq!(resolve_v_tensor(None, &k, false), None);
    }

    // ── pad_to_block ──

    #[test]
    fn pad_to_block_noop_when_exact_multiple() {
        let v = vec![1.0_f32; 256];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256, "exact multiple must not grow");
        assert_eq!(padded, v);

        let v = vec![1.0_f32; 512];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 512);
    }

    #[test]
    fn pad_to_block_zero_fills_to_next_block() {
        let v = vec![1.0_f32; 200];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256, "padded to next super-block");
        // First 200 preserved, last 56 zeroed.
        assert!(padded[..200].iter().all(|&x| x == 1.0));
        assert!(padded[200..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_block_handles_one_below_multiple() {
        let v = vec![1.0_f32; 255];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256);
        assert_eq!(padded[255], 0.0);
    }

    #[test]
    fn pad_to_block_handles_one_above_multiple() {
        let v = vec![1.0_f32; 257];
        let padded = pad_to_block(&v);
        assert_eq!(
            padded.len(),
            512,
            "one above block boundary → next full block"
        );
        assert!(padded[..257].iter().all(|&x| x == 1.0));
        assert!(padded[257..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_block_empty_input_stays_empty() {
        let v: Vec<f32> = Vec::new();
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 0);
    }
}
