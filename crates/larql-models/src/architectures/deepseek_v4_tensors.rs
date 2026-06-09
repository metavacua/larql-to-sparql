//! DeepSeek V4 GGUF tensor name schema.
//!
//! Maps the per-layer and global tensors expected by the DSv4 attention
//! and FFN blocks (Stages 8a, 8c, 8d, 8f, 8g) to GGUF tensor name
//! strings. Used by the per-layer GGUF loader to populate the typed
//! weight structs.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:88-160`
//! (`llama_model_deepseek4::llama_model_deepseek4` — the create_tensor
//! calls per layer).
//!
//! Naming convention follows llama.cpp's `LLM_TENSOR_FOO_BAR →
//! "blk.%d.foo_bar"` (or `"foo_bar"` for non-layered tensors).
//!
//! ## Calibration (2026-05-23)
//! Schema corrected against the real `DeepSeek-V4-Flash-Q4_K_M.gguf`
//! artifact (172 GB, 43 layers). Notable deviations from the speculative
//! initial draft:
//! - `attn_compress_*` (not `attn_compressor_*`)
//! - `indexer.compress_*` (dotted, not `indexer_compressor_*`)
//! - `attn_kv_latent` (not `attn_kv`)
//! - `attn_output_a/b` (not `attn_out_a/b`)
//! - `hc_head_*` for the output bookend (not `output_hc_*`)
//! - Several tensors carry **no `.weight` suffix**: `attn_sinks`,
//!   `attn_compress_ape`, `indexer.compress_ape`, `hc_attn_*`,
//!   `hc_ffn_*`, `hc_head_*`, `exp_probs_b`, `ffn_gate_tid2eid`.
//! - `exp_probs_b` (drop `ffn_` prefix and `.bias` suffix).
//!
//! See `crates/larql-models/examples/dsv4_inspect.rs` for the inspector
//! that produced these corrections.
//!
//! ## Tensors per layer
//! - Attention:
//!   - `attn_norm`              — input RMSNorm gain
//!   - `attn_q_a`               — Q low-rank "down"
//!   - `attn_q_a_norm`          — RMSNorm gain on q_a output
//!   - `attn_q_b`               — Q low-rank "up"
//!   - `attn_kv`                — single-head KV projection
//!   - `attn_kv_a_norm`         — RMSNorm gain on kv output
//!   - `attn_sinks`             — per-head attention sink logits
//!   - `attn_out_a` / `attn_out_b` — grouped low-rank output projection
//! - mHC (per-layer residual bookends):
//!   - `hc_attn_base`, `hc_attn_fn`, `hc_attn_scale`
//!   - `hc_ffn_base`,  `hc_ffn_fn`,  `hc_ffn_scale`
//! - Compressor (HCA, when compress_ratio > 0):
//!   - `attn_compressor_kv`, `attn_compressor_gate`,
//!     `attn_compressor_ape`, `attn_compressor_norm`
//! - Indexer (compress_ratio == 4):
//!   - `indexer_compressor_kv` / `_gate` / `_ape` / `_norm`
//!   - `indexer_attn_q_b`, `indexer_proj`
//! - MoE FFN:
//!   - `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b` (bias),
//!     `ffn_gate_tid2eid` (hash routing, first 3 layers only)
//!   - `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`
//!   - `ffn_gate_shexp`, `ffn_down_shexp`, `ffn_up_shexp` (shared experts)
//!
//! ## Global tensors
//! - `token_embd`, `output_norm`, `output`
//! - `output_hc_base`, `output_hc_fn`, `output_hc_scale` (mHC bookend
//!   at the head)

use std::fmt;

/// Suffix carried by a GGUF tensor name. Most weights end in `.weight`,
/// a few end in `.bias`, and some DSv4 tensors carry **no suffix at
/// all** (e.g. `blk.0.attn_sinks`, `blk.0.hc_attn_fn`). The schema
/// tracks the suffix per kind so the formatter / parser produce and
/// accept the right strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TensorSuffix {
    Weight,
    Bias,
    /// No suffix — the name is just `blk.N.stem` (per-layer) or
    /// `stem` (global).
    None,
}

/// Kind of DSv4 tensor — covers every weight the reference forward
/// path consumes. Variants are split into per-layer (require a layer
/// index) and global (don't).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DsV4TensorKind {
    // ── Global ──
    TokenEmbd,
    OutputNorm,
    Output,
    OutputHcBase,
    OutputHcFn,
    OutputHcScale,
    // ── Per-layer: attention ──
    AttnNorm,
    AttnQA,
    AttnQANorm,
    AttnQB,
    AttnKv,
    AttnKvANorm,
    AttnSinks,
    AttnOutA,
    AttnOutB,
    // ── Per-layer: mHC ──
    HcAttnBase,
    HcAttnFn,
    HcAttnScale,
    HcFfnBase,
    HcFfnFn,
    HcFfnScale,
    // ── Per-layer: compressor ──
    AttnCompressorKv,
    AttnCompressorGate,
    AttnCompressorApe,
    AttnCompressorNorm,
    // ── Per-layer: indexer ──
    IndexerCompressorKv,
    IndexerCompressorGate,
    IndexerCompressorApe,
    IndexerCompressorNorm,
    IndexerAttnQB,
    IndexerProj,
    // ── Per-layer: FFN / MoE ──
    FfnNorm,
    FfnGateInp,
    FfnExpProbsB,
    FfnGateTid2Eid,
    FfnGateExps,
    FfnDownExps,
    FfnUpExps,
    FfnGateShexp,
    FfnDownShexp,
    FfnUpShexp,
}

impl DsV4TensorKind {
    /// True iff this kind requires a layer index to form a name.
    pub fn is_per_layer(self) -> bool {
        !matches!(
            self,
            DsV4TensorKind::TokenEmbd
                | DsV4TensorKind::OutputNorm
                | DsV4TensorKind::Output
                | DsV4TensorKind::OutputHcBase
                | DsV4TensorKind::OutputHcFn
                | DsV4TensorKind::OutputHcScale
        )
    }

    /// Short stem matching the llama.cpp `LLM_TENSOR_FOO_BAR` → "foo_bar"
    /// convention (lower snake case with no "blk.N." or ".weight"
    /// decorators).
    pub fn stem(self) -> &'static str {
        match self {
            DsV4TensorKind::TokenEmbd => "token_embd",
            DsV4TensorKind::OutputNorm => "output_norm",
            DsV4TensorKind::Output => "output",
            // Output mHC bookend: actual GGUF uses "hc_head_*" (not "output_hc_*").
            DsV4TensorKind::OutputHcBase => "hc_head_base",
            DsV4TensorKind::OutputHcFn => "hc_head_fn",
            DsV4TensorKind::OutputHcScale => "hc_head_scale",
            DsV4TensorKind::AttnNorm => "attn_norm",
            DsV4TensorKind::AttnQA => "attn_q_a",
            DsV4TensorKind::AttnQANorm => "attn_q_a_norm",
            DsV4TensorKind::AttnQB => "attn_q_b",
            // Real GGUF name: "attn_kv_latent" (not "attn_kv").
            DsV4TensorKind::AttnKv => "attn_kv_latent",
            DsV4TensorKind::AttnKvANorm => "attn_kv_a_norm",
            DsV4TensorKind::AttnSinks => "attn_sinks",
            // Real GGUF names: "attn_output_a/b" (not "attn_out_a/b").
            DsV4TensorKind::AttnOutA => "attn_output_a",
            DsV4TensorKind::AttnOutB => "attn_output_b",
            DsV4TensorKind::HcAttnBase => "hc_attn_base",
            DsV4TensorKind::HcAttnFn => "hc_attn_fn",
            DsV4TensorKind::HcAttnScale => "hc_attn_scale",
            DsV4TensorKind::HcFfnBase => "hc_ffn_base",
            DsV4TensorKind::HcFfnFn => "hc_ffn_fn",
            DsV4TensorKind::HcFfnScale => "hc_ffn_scale",
            // Real GGUF names: "attn_compress_*" (not "attn_compressor_*").
            DsV4TensorKind::AttnCompressorKv => "attn_compress_kv",
            DsV4TensorKind::AttnCompressorGate => "attn_compress_gate",
            DsV4TensorKind::AttnCompressorApe => "attn_compress_ape",
            DsV4TensorKind::AttnCompressorNorm => "attn_compress_norm",
            // Real GGUF names: "indexer.compress_*" (dotted, no "or").
            DsV4TensorKind::IndexerCompressorKv => "indexer.compress_kv",
            DsV4TensorKind::IndexerCompressorGate => "indexer.compress_gate",
            DsV4TensorKind::IndexerCompressorApe => "indexer.compress_ape",
            DsV4TensorKind::IndexerCompressorNorm => "indexer.compress_norm",
            DsV4TensorKind::IndexerAttnQB => "indexer.attn_q_b",
            DsV4TensorKind::IndexerProj => "indexer.proj",
            DsV4TensorKind::FfnNorm => "ffn_norm",
            DsV4TensorKind::FfnGateInp => "ffn_gate_inp",
            // Real GGUF name: "exp_probs_b" (no "ffn_" prefix).
            DsV4TensorKind::FfnExpProbsB => "exp_probs_b",
            DsV4TensorKind::FfnGateTid2Eid => "ffn_gate_tid2eid",
            DsV4TensorKind::FfnGateExps => "ffn_gate_exps",
            DsV4TensorKind::FfnDownExps => "ffn_down_exps",
            DsV4TensorKind::FfnUpExps => "ffn_up_exps",
            DsV4TensorKind::FfnGateShexp => "ffn_gate_shexp",
            DsV4TensorKind::FfnDownShexp => "ffn_down_shexp",
            DsV4TensorKind::FfnUpShexp => "ffn_up_shexp",
        }
    }

    /// GGUF tensor-name suffix carried by this kind. Most tensors are
    /// `.weight`; a few DSv4-specific ones carry **no suffix** (see
    /// [`TensorSuffix::None`]). The DSv4 GGUF format doesn't currently
    /// produce any `.bias` tensors — `exp_probs_b` was speculative;
    /// the real artifact stores it as just `blk.N.exp_probs_b`.
    pub fn tensor_suffix(self) -> TensorSuffix {
        match self {
            // No-suffix variants observed in DeepSeek-V4-Flash GGUF.
            DsV4TensorKind::AttnSinks
            | DsV4TensorKind::AttnCompressorApe
            | DsV4TensorKind::IndexerCompressorApe
            | DsV4TensorKind::HcAttnBase
            | DsV4TensorKind::HcAttnFn
            | DsV4TensorKind::HcAttnScale
            | DsV4TensorKind::HcFfnBase
            | DsV4TensorKind::HcFfnFn
            | DsV4TensorKind::HcFfnScale
            | DsV4TensorKind::OutputHcBase
            | DsV4TensorKind::OutputHcFn
            | DsV4TensorKind::OutputHcScale
            | DsV4TensorKind::FfnExpProbsB
            | DsV4TensorKind::FfnGateTid2Eid => TensorSuffix::None,
            _ => TensorSuffix::Weight,
        }
    }
}

impl fmt::Display for DsV4TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stem())
    }
}

/// Build the GGUF tensor name for a `(kind, layer_index)` pair.
///
/// For global tensors `layer_index` is ignored. For per-layer tensors,
/// the form is `"blk.{layer_index}.{stem}[.{suffix}]"`. For global, it's
/// `"{stem}[.{suffix}]"`. Suffix is omitted when
/// `tensor_suffix() == TensorSuffix::None`.
pub fn tensor_name_of(kind: DsV4TensorKind, layer_index: usize) -> String {
    let stem = kind.stem();
    let suffix_str = match kind.tensor_suffix() {
        TensorSuffix::Weight => ".weight",
        TensorSuffix::Bias => ".bias",
        TensorSuffix::None => "",
    };
    if kind.is_per_layer() {
        format!("blk.{layer_index}.{stem}{suffix_str}")
    } else {
        format!("{stem}{suffix_str}")
    }
}

/// Reverse lookup: given a GGUF tensor name, recover the
/// `(kind, layer_index)` pair if it matches the DSv4 schema.
///
/// Returns `None` for names that don't fit the DSv4 pattern (e.g.
/// tensors from a different architecture).
pub fn try_parse_name(name: &str) -> Option<(DsV4TensorKind, Option<usize>)> {
    // Try with a .weight / .bias suffix first; if that fails, accept
    // a no-suffix form (for the DSv4 tensors that omit it).
    let parse = |head: &str, expect_kind: Option<DsV4TensorKind>| {
        if let Some(rest) = head.strip_prefix("blk.") {
            let (idx_str, stem) = rest.split_once('.')?;
            let layer: usize = idx_str.parse().ok()?;
            let kind = kind_from_stem(stem)?;
            if !kind.is_per_layer() {
                return None;
            }
            if let Some(ek) = expect_kind {
                if kind != ek {
                    return None;
                }
            }
            return Some((kind, Some(layer)));
        }
        let kind = kind_from_stem(head)?;
        if kind.is_per_layer() {
            return None;
        }
        if let Some(ek) = expect_kind {
            if kind != ek {
                return None;
            }
        }
        Some((kind, None))
    };

    if let Some((head, suffix)) = name.rsplit_once('.') {
        if suffix == "weight" || suffix == "bias" {
            if let Some((kind, layer)) = parse(head, None) {
                let want = match kind.tensor_suffix() {
                    TensorSuffix::Weight => suffix == "weight",
                    TensorSuffix::Bias => suffix == "bias",
                    TensorSuffix::None => false,
                };
                if want {
                    return Some((kind, layer));
                }
            }
        }
    }
    // Fall through: no-suffix form.
    parse(name, None).filter(|(k, _)| k.tensor_suffix() == TensorSuffix::None)
}

fn kind_from_stem(stem: &str) -> Option<DsV4TensorKind> {
    use DsV4TensorKind::*;
    let k = match stem {
        "token_embd" => TokenEmbd,
        "output_norm" => OutputNorm,
        "output" => Output,
        "hc_head_base" => OutputHcBase,
        "hc_head_fn" => OutputHcFn,
        "hc_head_scale" => OutputHcScale,
        "attn_norm" => AttnNorm,
        "attn_q_a" => AttnQA,
        "attn_q_a_norm" => AttnQANorm,
        "attn_q_b" => AttnQB,
        "attn_kv_latent" => AttnKv,
        "attn_kv_a_norm" => AttnKvANorm,
        "attn_sinks" => AttnSinks,
        "attn_output_a" => AttnOutA,
        "attn_output_b" => AttnOutB,
        "hc_attn_base" => HcAttnBase,
        "hc_attn_fn" => HcAttnFn,
        "hc_attn_scale" => HcAttnScale,
        "hc_ffn_base" => HcFfnBase,
        "hc_ffn_fn" => HcFfnFn,
        "hc_ffn_scale" => HcFfnScale,
        "attn_compress_kv" => AttnCompressorKv,
        "attn_compress_gate" => AttnCompressorGate,
        "attn_compress_ape" => AttnCompressorApe,
        "attn_compress_norm" => AttnCompressorNorm,
        "indexer.compress_kv" => IndexerCompressorKv,
        "indexer.compress_gate" => IndexerCompressorGate,
        "indexer.compress_ape" => IndexerCompressorApe,
        "indexer.compress_norm" => IndexerCompressorNorm,
        "indexer.attn_q_b" => IndexerAttnQB,
        "indexer.proj" => IndexerProj,
        "ffn_norm" => FfnNorm,
        "ffn_gate_inp" => FfnGateInp,
        "exp_probs_b" => FfnExpProbsB,
        "ffn_gate_tid2eid" => FfnGateTid2Eid,
        "ffn_gate_exps" => FfnGateExps,
        "ffn_down_exps" => FfnDownExps,
        "ffn_up_exps" => FfnUpExps,
        "ffn_gate_shexp" => FfnGateShexp,
        "ffn_down_shexp" => FfnDownShexp,
        "ffn_up_shexp" => FfnUpShexp,
        _ => return None,
    };
    Some(k)
}

/// Enumerate every DSv4 tensor kind — useful for cataloguing a model
/// (e.g. "for each layer, expect these N tensors").
pub fn all_kinds() -> &'static [DsV4TensorKind] {
    use DsV4TensorKind::*;
    &[
        TokenEmbd,
        OutputNorm,
        Output,
        OutputHcBase,
        OutputHcFn,
        OutputHcScale,
        AttnNorm,
        AttnQA,
        AttnQANorm,
        AttnQB,
        AttnKv,
        AttnKvANorm,
        AttnSinks,
        AttnOutA,
        AttnOutB,
        HcAttnBase,
        HcAttnFn,
        HcAttnScale,
        HcFfnBase,
        HcFfnFn,
        HcFfnScale,
        AttnCompressorKv,
        AttnCompressorGate,
        AttnCompressorApe,
        AttnCompressorNorm,
        IndexerCompressorKv,
        IndexerCompressorGate,
        IndexerCompressorApe,
        IndexerCompressorNorm,
        IndexerAttnQB,
        IndexerProj,
        FfnNorm,
        FfnGateInp,
        FfnExpProbsB,
        FfnGateTid2Eid,
        FfnGateExps,
        FfnDownExps,
        FfnUpExps,
        FfnGateShexp,
        FfnDownShexp,
        FfnUpShexp,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_name_format() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnNorm, 7),
            "blk.7.attn_norm.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnQA, 0),
            "blk.0.attn_q_a.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::FfnGateInp, 12),
            "blk.12.ffn_gate_inp.weight"
        );
    }

    #[test]
    fn global_name_format() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::TokenEmbd, 0),
            "token_embd.weight"
        );
        // OutputHcFn has TensorSuffix::None and uses the "hc_head_*" stem.
        assert_eq!(
            tensor_name_of(DsV4TensorKind::OutputHcFn, 999),
            "hc_head_fn"
        );
    }

    /// `exp_probs_b` carries no suffix in the real GGUF.
    #[test]
    fn exp_probs_b_has_no_suffix() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::FfnExpProbsB, 3),
            "blk.3.exp_probs_b"
        );
        assert_eq!(
            DsV4TensorKind::FfnExpProbsB.tensor_suffix(),
            TensorSuffix::None
        );
    }

    /// Several mHC/sinks/ape tensors carry no `.weight` suffix.
    #[test]
    fn no_suffix_variants_format_without_dot_weight() {
        for (kind, expected) in [
            (DsV4TensorKind::AttnSinks, "blk.5.attn_sinks"),
            (DsV4TensorKind::HcAttnFn, "blk.5.hc_attn_fn"),
            (DsV4TensorKind::HcFfnScale, "blk.5.hc_ffn_scale"),
            (DsV4TensorKind::AttnCompressorApe, "blk.5.attn_compress_ape"),
            (
                DsV4TensorKind::IndexerCompressorApe,
                "blk.5.indexer.compress_ape",
            ),
            (DsV4TensorKind::FfnGateTid2Eid, "blk.5.ffn_gate_tid2eid"),
        ] {
            assert_eq!(tensor_name_of(kind, 5), expected, "kind={kind:?}");
            assert_eq!(kind.tensor_suffix(), TensorSuffix::None);
        }
    }

    /// Renamed kinds use the calibrated names from the real DSv4 GGUF.
    #[test]
    fn renamed_kinds_match_real_gguf() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnKv, 7),
            "blk.7.attn_kv_latent.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnOutA, 7),
            "blk.7.attn_output_a.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnCompressorKv, 7),
            "blk.7.attn_compress_kv.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::IndexerCompressorGate, 7),
            "blk.7.indexer.compress_gate.weight"
        );
    }

    #[test]
    fn indexer_uses_dotted_stem() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::IndexerAttnQB, 5),
            "blk.5.indexer.attn_q_b.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::IndexerProj, 5),
            "blk.5.indexer.proj.weight"
        );
    }

    #[test]
    fn round_trip_per_layer() {
        // Every per-layer kind should round-trip through name → parse.
        for &kind in all_kinds() {
            if !kind.is_per_layer() {
                continue;
            }
            let name = tensor_name_of(kind, 42);
            let parsed = try_parse_name(&name)
                .unwrap_or_else(|| panic!("failed to parse {name} for {kind:?}"));
            assert_eq!(parsed, (kind, Some(42)), "round-trip mismatch on {name}");
        }
    }

    #[test]
    fn round_trip_global() {
        for &kind in all_kinds() {
            if kind.is_per_layer() {
                continue;
            }
            let name = tensor_name_of(kind, 0);
            let parsed = try_parse_name(&name)
                .unwrap_or_else(|| panic!("failed to parse {name} for {kind:?}"));
            assert_eq!(parsed, (kind, None), "round-trip mismatch on {name}");
        }
    }

    #[test]
    fn parse_rejects_unrelated_names() {
        assert_eq!(try_parse_name("blk.0.attn_q.weight"), None);
        assert_eq!(try_parse_name("ssm.in.weight"), None);
        assert_eq!(try_parse_name("garbage"), None);
        // No `.weight` / `.bias` suffix.
        assert_eq!(try_parse_name("blk.0.attn_norm"), None);
        // Per-layer kind used without a layer index.
        assert_eq!(try_parse_name("attn_norm.weight"), None);
        // Global kind used with a layer index.
        assert_eq!(try_parse_name("blk.0.token_embd.weight"), None);
    }

    #[test]
    fn is_per_layer_classification() {
        assert!(!DsV4TensorKind::TokenEmbd.is_per_layer());
        assert!(!DsV4TensorKind::OutputHcFn.is_per_layer());
        assert!(DsV4TensorKind::AttnNorm.is_per_layer());
        assert!(DsV4TensorKind::IndexerProj.is_per_layer());
        assert!(DsV4TensorKind::HcAttnFn.is_per_layer());
    }

    #[test]
    fn all_kinds_are_unique() {
        let kinds = all_kinds();
        let mut seen: std::collections::HashSet<DsV4TensorKind> = std::collections::HashSet::new();
        for &k in kinds {
            assert!(seen.insert(k), "duplicate kind {k:?}");
        }
    }

    #[test]
    fn all_stems_are_unique() {
        let kinds = all_kinds();
        let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        for &k in kinds {
            assert!(seen.insert(k.stem()), "duplicate stem {}", k.stem());
        }
    }
}
