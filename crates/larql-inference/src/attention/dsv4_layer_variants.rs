//! Detect per-layer variant + compress_ratio from GGUF tensor presence
//! and shapes.
//!
//! The real DSv4-Flash GGUF doesn't carry a per-layer `compress_ratio`
//! metadata field — it has to be inferred from the layer's
//! `attn_compress_ape` tensor shape:
//!
//! ```text
//! ape shape (ggml ne order) = [coff * head_dim, compress_ratio]
//! → coff           = dims[0] / head_dim
//! → compress_ratio = dims[1]
//! ```
//!
//! Layers without an `attn_compress_*` tensor set are "no-compress"
//! (use Stage 8a). Layers with both compressor + indexer tensors use
//! Stage 8g (compress_ratio=4 path). Layers with compressor only use
//! Stage 8f. Hash routing for the FFN side is signalled by the
//! presence of `ffn_gate_tid2eid` (first `n_hash_layers` layers; DSv4
//! spec uses 3).
//!
//! ## Real DSv4-Flash pattern (43 layers)
//! - layers 0, 1: hash routing only
//! - layer 2: hash routing + compressor + indexer
//! - odd 3-41: regular routing + compressor (no indexer)
//! - even 4-42: regular routing + compressor + indexer

use larql_models::loading::gguf::GgufFile;

/// Per-layer variant record. Drives both attention dispatch and FFN
/// routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsV4LayerVariant {
    /// True iff the layer's FFN uses hash routing (looking up
    /// `ffn_gate_tid2eid`). False → sqrt(softplus) router.
    pub uses_hash_routing: bool,
    /// `Some(ratio)` iff the layer has an `attn_compress_*` tensor set
    /// (HCA path). `None` for no-compress (Stage 8a) layers.
    pub compress_ratio: Option<usize>,
    /// True iff the layer has indexer tensors (`indexer.compress_*`,
    /// `indexer.attn_q_b`, `indexer.proj`). Only meaningful when
    /// `compress_ratio == Some(4)` per the DSv4 spec, but the detector
    /// reports the raw presence — callers can validate the combination.
    pub has_indexer: bool,
}

impl DsV4LayerVariant {
    /// Pick which attention block this layer uses.
    pub fn attention_variant_tag(self) -> AttnVariantTag {
        match (self.compress_ratio, self.has_indexer) {
            (None, _) => AttnVariantTag::NoCompress,
            (Some(_), false) => AttnVariantTag::Compress,
            (Some(_), true) => AttnVariantTag::Indexer,
        }
    }
}

/// Tag matching the three Stage 8a/8f/8g attention paths.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AttnVariantTag {
    NoCompress,
    Compress,
    Indexer,
}

/// Detect a single layer's variant from its GGUF tensors.
///
/// `head_dim` is needed to interpret the `attn_compress_ape` shape
/// (to recover `coff` and `compress_ratio`).
pub fn detect_layer_variant(
    gguf: &GgufFile,
    layer_index: usize,
    head_dim: usize,
) -> DsV4LayerVariant {
    let prefix = format!("blk.{layer_index}.");
    let has = |stem_with_suffix: &str| -> bool {
        let name = format!("{prefix}{stem_with_suffix}");
        gguf.tensor_infos.iter().any(|i| i.name() == name)
    };

    let uses_hash_routing = has("ffn_gate_tid2eid");
    let has_indexer = has("indexer.compress_kv.weight")
        || has("indexer.attn_q_b.weight")
        || has("indexer.proj.weight");

    let compress_ratio = compress_ratio_from_ape(gguf, layer_index, head_dim);

    DsV4LayerVariant {
        uses_hash_routing,
        compress_ratio,
        has_indexer,
    }
}

/// Detect the variant for every layer in `0..n_layer`.
pub fn detect_all_layer_variants(
    gguf: &GgufFile,
    n_layer: usize,
    head_dim: usize,
) -> Vec<DsV4LayerVariant> {
    (0..n_layer)
        .map(|l| detect_layer_variant(gguf, l, head_dim))
        .collect()
}

/// Extract `compress_ratio` for a single layer by looking up
/// `blk.N.attn_compress_ape` and interpreting its shape.
///
/// The GGUF stores the ape as ggml `[coff * head_dim, compress_ratio]`,
/// so `compress_ratio = dims[1]`. Returns `None` if the tensor isn't
/// present (no-compress layer) or the shape is unexpected.
pub fn compress_ratio_from_ape(
    gguf: &GgufFile,
    layer_index: usize,
    head_dim: usize,
) -> Option<usize> {
    let name = format!("blk.{layer_index}.attn_compress_ape");
    let info = gguf.tensor_infos.iter().find(|i| i.name() == name)?;
    let dims = info.dims();
    if dims.len() != 2 {
        return None;
    }
    let coff_times_head_dim = dims[0] as usize;
    if head_dim == 0 || coff_times_head_dim % head_dim != 0 {
        return None;
    }
    Some(dims[1] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_variant_tag_dispatches_correctly() {
        let no_comp = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: None,
            has_indexer: false,
        };
        assert_eq!(no_comp.attention_variant_tag(), AttnVariantTag::NoCompress);

        let comp_only = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(128),
            has_indexer: false,
        };
        assert_eq!(comp_only.attention_variant_tag(), AttnVariantTag::Compress);

        let with_indexer = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(4),
            has_indexer: true,
        };
        assert_eq!(
            with_indexer.attention_variant_tag(),
            AttnVariantTag::Indexer
        );
    }

    /// Real-GGUF detection: 43 layers must match the calibration baseline:
    /// - layers 0, 1: hash routing, no compressor, no indexer
    /// - layer 2: hash routing + compressor + indexer (ratio=4)
    /// - odd 3..=41: regular + compressor (no indexer)
    /// - even 4..=42: regular + compressor + indexer (ratio=4)
    /// Plus odd compress_ratio==128 (sliding-window-sized) and even==4.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_layer_variants_match_calibration_baseline() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let head_dim = 512;
        let n_layer = 43;
        let variants = detect_all_layer_variants(&gguf, n_layer, head_dim);
        assert_eq!(variants.len(), n_layer);

        // Layers 0, 1: hash routing, no compress, no indexer.
        for l in [0, 1] {
            assert!(variants[l].uses_hash_routing, "layer {l} hash routing");
            assert_eq!(variants[l].compress_ratio, None, "layer {l} compress_ratio");
            assert!(!variants[l].has_indexer, "layer {l} indexer");
            assert_eq!(
                variants[l].attention_variant_tag(),
                AttnVariantTag::NoCompress
            );
        }

        // Layer 2: hash routing + compressor + indexer (ratio=4).
        assert!(variants[2].uses_hash_routing);
        assert_eq!(variants[2].compress_ratio, Some(4));
        assert!(variants[2].has_indexer);
        assert_eq!(variants[2].attention_variant_tag(), AttnVariantTag::Indexer);

        // Layer 3..=42:
        for l in 3..n_layer {
            assert!(!variants[l].uses_hash_routing, "layer {l} regular routing");
            let want_indexer = l % 2 == 0; // even ≥ 4 have indexer
            assert_eq!(
                variants[l].has_indexer, want_indexer,
                "layer {l} indexer={want_indexer}"
            );
            let want_ratio = if want_indexer { 4 } else { 128 };
            assert_eq!(
                variants[l].compress_ratio,
                Some(want_ratio),
                "layer {l} compress_ratio={want_ratio}"
            );
            let want_tag = if want_indexer {
                AttnVariantTag::Indexer
            } else {
                AttnVariantTag::Compress
            };
            assert_eq!(variants[l].attention_variant_tag(), want_tag);
        }
    }
}
