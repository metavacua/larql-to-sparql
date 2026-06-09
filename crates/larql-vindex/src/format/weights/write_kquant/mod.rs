//! Q4_K / Q6_K streaming writer — separate from `write_f32` because
//! the Q4_K pipeline owns its own QuantBlockFormat manifest, padding
//! helpers, and per-tensor quantisation policy.
//!
//! Carved out of the monolithic `write.rs` in the 2026-04-25 reorg,
//! and re-decomposed in 2026-05-09 round-5 into one sibling per
//! emitted artefact:
//!
//! - [`attn`] — `attn_weights_q4k.bin` (+ manifest)
//! - [`ffn`] — `interleaved_kquant.bin` (+ opt `down_features_q4k.bin`)
//! - [`moe_layers`] — `layers/layer_{L:02}.weights` (hybrid MoE)
//! - [`norms`] — `norms.bin` (norms + MoE router/scales)
//! - `super::ple_sidecar` — `ple_weights.bin` (Gemma 4 PLE, f16,
//!   shared with the `write_f32` writer so non-Q4 extracts capture
//!   the same sidecars; see chrishayuk/larql#49)
//! - [`lm_head`] — `lm_head_q4.bin`
//!
//! The orchestrator below threads the running `Vec<WeightEntry>`
//! manifest through the norms → ple → lm_head trio, then emits a
//! single `weight_manifest.json` and patches `index.json`.

use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::config::{FfnLayout, VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::capabilities::{ensure_standard_attention_supported, SURFACE_Q4K_WEIGHT_WRITER};
use super::write_f32::WeightSource;

mod attn;
mod deltanet;
mod ffn;
mod lm_head;
mod moe_layers;
mod norms;
<<<<<<< HEAD:crates/larql-vindex/src/format/weights/write_kquant/mod.rs

pub mod feature_major_down;

/// Per-block quantisation format tag carried by Q4_K pipeline manifests.
///
/// Serialises / deserialises as the literal on-disk tag string
/// (`"Q4_K"`, `"Q6_K"`, …) to match llama.cpp / Ollama conventions. The
/// `Other` variant accepts tags that future binaries can decode but
/// this one can't — readers see the format string and route through
/// [`crate::quant::registry`]; if the registry returns `None` the
/// caller surfaces a clear "unknown format" error rather than the
/// previous serde panic on an unknown variant.
///
/// Adding a new format the registry can decode (e.g., Q5_K) is a
/// single entry in `QUANT_FORMATS` — no edit to this enum is required.
/// Add an explicit variant here only when the writer pipeline also
/// supports emitting the format (the writer dispatches typed because
/// emitting a new format is a deliberate act that needs an encode
/// function + user-config option).
#[derive(Debug, Clone, PartialEq, Eq)]
=======
mod ple;
mod shexp;

pub mod feature_major_down;

/// Per-block quantisation format for a single tensor in the Q4_K pipeline.
/// Serde writes / reads the on-disk strings (`"Q4_K"`, `"Q5_K"`, `"Q6_K"`,
/// `"Q8_0"`) to match llama.cpp / Ollama conventions. New formats land
/// here + as a row in [`crate::quant::registry::QUANT_FORMATS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
>>>>>>> ianblenke/main:crates/larql-vindex/src/format/weights/write_q4k/mod.rs
pub enum QuantBlockFormat {
    Q4K,
<<<<<<< HEAD:crates/larql-vindex/src/format/weights/write_kquant/mod.rs
    Q6K,
    /// Tag the writer pipeline cannot emit but the reader can identify.
    /// Carries the on-disk string so dispatch can consult the registry.
    Other(String),
}

impl QuantBlockFormat {
    /// On-disk tag string. Routes through [`crate::quant::registry::lookup`].
    pub fn tag(&self) -> &str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Construct from a tag string, succeeding only when the format is
    /// known to [`crate::quant::registry`]. Use at vindex-load seams to
    /// reject unknown formats once, instead of letting the dispatch
    /// kernels report `None` per-row.
    pub fn from_registry_tag(tag: &str) -> Option<Self> {
        crate::quant::registry::lookup(tag)?;
        Some(match tag {
            "Q4_K" => Self::Q4K,
            "Q6_K" => Self::Q6K,
            other => Self::Other(other.to_string()),
        })
    }
}

impl Serialize for QuantBlockFormat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.tag())
    }
}

impl<'de> Deserialize<'de> for QuantBlockFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "Q4_K" => Self::Q4K,
            "Q6_K" => Self::Q6K,
            _ => Self::Other(s),
        })
    }
=======
    #[serde(rename = "Q5_K")]
    Q5K,
    #[serde(rename = "Q6_K")]
    Q6K,
    /// Legacy GGUF 32-element-block 8-bit format. Higher precision
    /// than Q4_K (2.7 bpw less compression). Used by Unsloth-style
    /// "Q4_K_M" GGUFs to keep high-importance tensors (attn / lm_head)
    /// at near-f16 precision while compressing FFN experts.
    #[serde(rename = "Q8_0")]
    Q8_0,
    /// GGUF interleaved MXFP4 — 17 B per 32-element block. Used by
    /// Qwen3-Coder-Next's `ffn_*_shexp.weight` (shared-expert gate/up).
    /// Block size 32 — alignment constraint differs from K-quants.
    #[serde(rename = "MXFP4")]
    Mxfp4,
>>>>>>> ianblenke/main:crates/larql-vindex/src/format/weights/write_q4k/mod.rs
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

/// Try a bit-exact passthrough of GGUF Q4_K (or Q6_K) bytes from
/// `source` into the writer when:
///
/// - the source carries the tensor as raw quant bytes (GGUF-backed)
/// - the source's tensor_type matches the writer's target format
/// - `cols` is already a multiple of the super-block size (256) so
///   no row-padding is needed
///
/// Returns `Some((bytes, rows, cols))` on a successful passthrough.
/// `None` falls back to the legacy dequant → re-quantize path
/// (`quantize_q4_k(pad_rows_to_block(get_tensor(...)))`).
///
/// Imatrix-aware quantizers (Unsloth, TheBloke 2026, Qwen team)
/// pick per-block scales informed by a calibration dataset's
/// activation distribution. Our naïve `quantize_q4_k` doesn't have
/// that signal — it minimizes per-block reconstruction error in
/// f32 space. The two produce ~0.1% per-element divergence, which
/// compounds through 40 layers × ~5 matmuls/layer into the residual
/// stream and breaks chat-completion coherence by L31 on Qwen 3.6
/// 35B-A3B (per PR #194's bisection).
pub(super) fn try_q4k_passthrough(
    source: &dyn WeightSource,
    key: &str,
    target: QuantBlockFormat,
) -> Option<(Vec<u8>, usize, usize)> {
    let (bytes, ttype, rows, cols) = source.get_quant_raw(key)?;
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    if !cols.is_multiple_of(block) {
        return None;
    }
    let expected = match target {
        QuantBlockFormat::Q4K => larql_models::quant::ggml::TYPE_Q4_K,
        QuantBlockFormat::Q5K => larql_models::quant::ggml::TYPE_Q5_K,
        QuantBlockFormat::Q6K => larql_models::quant::ggml::TYPE_Q6_K,
        QuantBlockFormat::Q8_0 => larql_models::quant::ggml::TYPE_Q8_0,
        QuantBlockFormat::Mxfp4 => larql_models::quant::ggml::TYPE_MXFP4,
    };
    if ttype != expected {
        return None;
    }
    // Sanity-check the byte count matches the (rows, cols) at target format.
    let block_bytes = match target {
        QuantBlockFormat::Q4K => larql_models::quant::ggml::Q4_K_BLOCK_BYTES,
        QuantBlockFormat::Q5K => larql_models::quant::ggml::Q5_K_BLOCK_BYTES,
        QuantBlockFormat::Q6K => larql_models::quant::ggml::Q6_K_BLOCK_BYTES,
        QuantBlockFormat::Q8_0 => larql_models::quant::ggml::Q8_0_BLOCK_BYTES,
        QuantBlockFormat::Mxfp4 => larql_models::quant::ggml::MXFP4_BLOCK_BYTES,
    };
    let expected_bytes = rows * (cols / block) * block_bytes;
    if bytes.len() != expected_bytes {
        return None;
    }
    Some((bytes, rows, cols))
}

/// Source-preserving passthrough: pick the storage format from
/// whatever the source carries (Q4_K / Q5_K / Q6_K / Q8_0), rather
/// than requiring the caller to specify a target. Returns
/// `(bytes, source_format, rows, cols)` when:
///
/// - the source carries the tensor as raw quant bytes (GGUF-backed)
/// - the source tensor_type is one we know how to store
/// - alignment + byte count check out for that format's block size
///
/// Use this in writer sites where the policy is "store at whatever
/// precision the source provides" — i.e. preserve Q5_K / Q8_0
/// instead of downquantizing to Q4_K. Falls back to `None` (caller
/// dequant + requantize) for anything that doesn't match.
///
/// Q4_K / Q5_K / Q6_K share the K-quant 256-element super-block
/// (144 / 176 / 210 B respectively). **Q8_0 uses 32-element blocks
/// (34 B/block)** — different alignment constraint, so the check
/// is per-format.
pub(super) fn try_preserve_quant_passthrough(
    source: &dyn WeightSource,
    key: &str,
) -> Option<(Vec<u8>, QuantBlockFormat, usize, usize)> {
    use larql_models::quant::ggml::{
        K_QUANT_BLOCK_ELEMS, LEGACY_BLOCK_ELEMS, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
        Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, TYPE_MXFP4,
        TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0,
    };
    let (bytes, ttype, rows, cols) = source.get_quant_raw(key)?;
    let (format, block_elems, block_bytes) = match ttype {
        x if x == TYPE_Q4_K => (QuantBlockFormat::Q4K, K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES),
        x if x == TYPE_Q5_K => (QuantBlockFormat::Q5K, K_QUANT_BLOCK_ELEMS, Q5_K_BLOCK_BYTES),
        x if x == TYPE_Q6_K => (QuantBlockFormat::Q6K, K_QUANT_BLOCK_ELEMS, Q6_K_BLOCK_BYTES),
        x if x == TYPE_Q8_0 => (QuantBlockFormat::Q8_0, LEGACY_BLOCK_ELEMS, Q8_0_BLOCK_BYTES),
        x if x == TYPE_MXFP4 => (
            QuantBlockFormat::Mxfp4,
            MXFP4_BLOCK_ELEMS,
            MXFP4_BLOCK_BYTES,
        ),
        _ => return None,
    };
    if !cols.is_multiple_of(block_elems) {
        return None;
    }
    let expected_bytes = rows * (cols / block_elems) * block_bytes;
    if bytes.len() != expected_bytes {
        return None;
    }
    Some((bytes, format, rows, cols))
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

/// Quantisation format for the FFN down-projection. Gate / up are
/// always Q4_K in the k-quant writer; only down varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DownProjFormat {
    /// Q6_K for down (Ollama "Q4_K_M" default mix). Higher precision,
    /// larger files, slower down matmul.
    #[default]
    Q6K,
    /// Q4_K for down (uniform Q4_K across gate/up/down). Saves ~30 MB
    /// per layer on 31B (~1.8 GB total) and drops down matmul cost
    /// ~1.5-1.7× to match up-proj timings. Quantisation noise on the
    /// scatter-sum averages across the intermediate dimension; empirically
    /// close.
    Q4K,
}

/// Options for [`write_model_weights_kquant_with_opts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct KquantWriteOptions {
    /// Format for the FFN down-projection. See [`DownProjFormat`].
    pub down_proj: DownProjFormat,

    /// Emit `down_features_q4k.bin` alongside `interleaved_kquant.bin`.
    /// When set, the down weights are also stored in feature-major
    /// `[intermediate, hidden]` orientation (Q4_K/Q6_K matching
    /// `down_proj`), so per-feature decode can skip the
    /// `kquant_ffn_layer` whole-layer dequant + transpose cache. Adds
    /// roughly the same disk footprint as the down portion of
    /// `interleaved_kquant.bin` (~14 MB / layer at Gemma 4B dims).
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
///   interleaved_kquant.bin
///     — [gate Q4_K | up Q4_K | down Q6_K] per layer, regular stride.
///     — With `down_proj=DownProjFormat::Q4K`: [gate | up | down] all Q4_K.
///   lm_head_q4.bin
///     — Q4_K of the output projection (falls back to embed_tokens when tied).
///   norms.bin (f32, unchanged from non-Q4 path).
///
/// The source's per-tensor f32 materialisation is transient — one tensor's
/// worth of heap (~350 MB peak on 31B global layer Q) quantised then dropped.
pub fn write_model_weights_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_kquant_with_opts(source, dir, callbacks, KquantWriteOptions::default())
}

/// Like [`write_model_weights_kquant`] but accepts a [`KquantWriteOptions`]
/// knob to toggle the FFN down-proj quantisation format and the
/// feature-major-down emit.
pub fn write_model_weights_kquant_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: KquantWriteOptions,
) -> Result<(), VindexError> {
    callbacks.on_stage(STAGE_MODEL_WEIGHTS_KQUANT);
    let start = std::time::Instant::now();

    let arch = source.arch();
    if arch.uses_mla() {
        return Err(VindexError::UnsupportedArchitecture {
            family: arch.family().to_string(),
            feature: "multi-head latent attention (MLA)".into(),
            surface: SURFACE_Q4K_WEIGHT_WRITER.into(),
        });
    }
    ensure_standard_attention_supported(arch, SURFACE_Q4K_WEIGHT_WRITER)?;
    let num_layers = source.num_layers();

<<<<<<< HEAD:crates/larql-vindex/src/format/weights/write_kquant/mod.rs
    attn::write_attn_weights_kquant(source, dir, num_layers, callbacks)?;
    ffn::write_interleaved_ffn_kquant(source, dir, num_layers, opts, callbacks)?;
    moe_layers::write_per_layer_moe_kquant(source, dir, num_layers)?;
=======
    attn::write_attn_weights_q4k(source, dir, num_layers, callbacks)?;
    deltanet::write_deltanet_weights_q4k(source, dir, num_layers, callbacks)?;
    ffn::write_interleaved_ffn_q4k(source, dir, num_layers, opts, callbacks)?;
    moe_layers::write_per_layer_moe_q4k(source, dir, num_layers)?;
    shexp::write_shexp_weights_q4k(source, dir, num_layers)?;
>>>>>>> ianblenke/main:crates/larql-vindex/src/format/weights/write_q4k/mod.rs
    let mut entries = norms::write_norms_and_router(source, dir, num_layers)?;
    super::ple_sidecar::write_ple_weights(source, dir, num_layers, &mut entries)?;
    lm_head::write_lm_head_kquant(source, dir, &mut entries)?;

    let manifest_json =
        serde_json::to_string_pretty(&entries).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(WEIGHT_MANIFEST_JSON), manifest_json)?;

    update_index_json(dir, source.arch())?;

    callbacks.on_stage_done(
        STAGE_MODEL_WEIGHTS_KQUANT,
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

    // ── QuantBlockFormat: future-format extension ──

    #[test]
    fn quant_block_format_round_trip_known_variants() {
        // Existing typed variants serialise to their canonical tags
        // and round-trip back as the same variant.
        let q4 = serde_json::to_string(&QuantBlockFormat::Q4K).unwrap();
        let q6 = serde_json::to_string(&QuantBlockFormat::Q6K).unwrap();
        assert_eq!(q4, "\"Q4_K\"");
        assert_eq!(q6, "\"Q6_K\"");
        let back: QuantBlockFormat = serde_json::from_str("\"Q4_K\"").unwrap();
        assert_eq!(back, QuantBlockFormat::Q4K);
        let back: QuantBlockFormat = serde_json::from_str("\"Q6_K\"").unwrap();
        assert_eq!(back, QuantBlockFormat::Q6K);
    }

    #[test]
    fn quant_block_format_unknown_tag_round_trips_as_other() {
        // A future format the reader binary doesn't recognise must
        // round-trip through the manifest without panicking. This is
        // the whole point of the open-enum redesign: a manifest emitted
        // by a future writer (with Q5_K) is identifiable to a current
        // binary as "format `Q5_K`, dispatch unknown", not a serde
        // panic at deserialize time.
        let parsed: QuantBlockFormat = serde_json::from_str("\"Q5_K\"").unwrap();
        assert_eq!(parsed, QuantBlockFormat::Other("Q5_K".into()));
        assert_eq!(parsed.tag(), "Q5_K");
        let re_serialized = serde_json::to_string(&parsed).unwrap();
        assert_eq!(re_serialized, "\"Q5_K\"");
    }

    #[test]
    fn quant_block_format_from_registry_tag_validates() {
        // `from_registry_tag` is the construction seam — only tags the
        // registry recognises become a `QuantBlockFormat`. This is the
        // load-time gate that prevents a corrupt manifest from
        // surfacing as silent zero-row matmul results.
        assert_eq!(
            QuantBlockFormat::from_registry_tag("Q4_K"),
            Some(QuantBlockFormat::Q4K)
        );
        assert_eq!(
            QuantBlockFormat::from_registry_tag("Q6_K"),
            Some(QuantBlockFormat::Q6K)
        );
        // Q5_K isn't in the registry yet — gated.
        assert!(QuantBlockFormat::from_registry_tag("Q5_K").is_none());
        // Typos and lowercase are gated too — manifests on disk always
        // use the canonical upper-case form.
        assert!(QuantBlockFormat::from_registry_tag("q4_k").is_none());
        assert!(QuantBlockFormat::from_registry_tag("").is_none());
    }

    #[test]
    fn quant_block_format_tag_borrows_from_other_variant() {
        // The `Other` variant's tag must round-trip the runtime string
        // exactly — no canonicalisation, no case-folding. Stale or
        // future manifest entries are echoed back unchanged so callers
        // can produce a clear "unknown format `X`" diagnostic.
        let f = QuantBlockFormat::Other("Q5_K_M".into());
        assert_eq!(f.tag(), "Q5_K_M");
    }

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
