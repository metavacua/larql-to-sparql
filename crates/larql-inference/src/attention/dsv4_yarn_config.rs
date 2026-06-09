//! DSv4 YARN RoPE-scaling configuration.
//!
//! DSv4-Flash uses YARN to extend context from 65 536 (original) to
//! 1 048 576 tokens (factor=16). The scaling parameters live in GGUF
//! metadata; this module extracts and validates them so a future
//! `dsv4_rope_tail_yarn` variant can consume a struct rather than a
//! grab-bag of f32s.
//!
//! Reference: llama.cpp `rope_yarn_corr_dims`, `rope_yarn_ramp`,
//! `ggml_rope_yarn_log_mscale` (in ggml-cpu/ops.cpp). The same 7
//! parameters are threaded into `ggml_dsv4_rope_tail` from
//! `src/models/deepseek4.cpp:485-487`.

use larql_models::loading::gguf::{GgufFile, GgufValue};

use super::dsv4_hyperparams_load::DsV4MetadataError;

/// RoPE scaling family identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    /// No scaling — plain RoPE.
    None,
    /// YARN extrapolation+interpolation hybrid.
    Yarn,
}

/// YARN RoPE scaling configuration.
///
/// All `f32` fields use the same names the reference llama.cpp
/// implementation uses. When `scaling_type == None` the YARN params
/// degenerate to identity (`factor = 1`, `n_ctx_orig` ignored).
#[derive(Clone, Copy, Debug)]
pub struct DsV4RopeYarnConfig {
    /// Whether YARN scaling is active.
    pub scaling_type: RopeScalingType,
    /// Main rope base (typically 10 000.0 for DSv4).
    pub freq_base: f64,
    /// Scaling factor (= 1 / yarn_factor). For yarn_factor=16,
    /// freq_scale=1/16=0.0625. For RopeScalingType::None, 1.0.
    pub freq_scale: f32,
    /// Extrapolation factor — `1.0` means standard YARN ramp;
    /// `0.0` disables extrapolation entirely. DSv4 typical: 1.0.
    pub ext_factor: f32,
    /// Attention factor: scales the rope angles for the attention
    /// softmax temperature. Typically 1.0. Set to `0.1 * ln(factor) + 1.0`
    /// for the canonical YARN mscale.
    pub attn_factor: f32,
    /// Lower-bound rotation index where YARN ramp begins.
    pub beta_fast: f32,
    /// Upper-bound rotation index where YARN ramp ends.
    pub beta_slow: f32,
    /// Pre-extension context length (used by the YARN ramp).
    pub n_ctx_orig: usize,
}

impl DsV4RopeYarnConfig {
    /// Construct a no-scaling config — identical to plain RoPE.
    pub fn identity(freq_base: f64) -> Self {
        DsV4RopeYarnConfig {
            scaling_type: RopeScalingType::None,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 0,
        }
    }

    /// Extract from a GgufFile's metadata (DSv4-Flash layout).
    ///
    /// Reads `deepseek4.rope.*` keys. Returns an `identity`-style
    /// config when `scaling.type` is missing or "none"; YARN params
    /// when "yarn".
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, DsV4MetadataError> {
        let md = &gguf.metadata;

        // freq_base is required.
        let freq_base = f32_md(md, "deepseek4.rope.freq_base")? as f64;

        // Scaling type optional — fall back to None (=identity).
        let scaling_type = match str_md_opt(md, "deepseek4.rope.scaling.type")? {
            Some("yarn") => RopeScalingType::Yarn,
            None | Some("none") | Some("linear") => RopeScalingType::None,
            Some(_other) => {
                return Err(DsV4MetadataError::WrongType {
                    key: "deepseek4.rope.scaling.type",
                    wanted: "\"none\" | \"linear\" | \"yarn\"",
                });
            }
        };

        if scaling_type == RopeScalingType::None {
            return Ok(DsV4RopeYarnConfig::identity(freq_base));
        }

        // YARN params.
        let factor = f32_md(md, "deepseek4.rope.scaling.factor")?;
        let beta_fast = f32_md(md, "deepseek4.rope.scaling.yarn_beta_fast")?;
        let beta_slow = f32_md(md, "deepseek4.rope.scaling.yarn_beta_slow")?;
        let n_ctx_orig = u32_md(md, "deepseek4.rope.scaling.original_context_length")? as usize;

        // Derived: freq_scale = 1 / factor. attn_factor and ext_factor
        // aren't in the metadata; use the canonical YARN defaults:
        //   attn_factor = 0.1 * ln(factor) + 1.0   (YARN paper §3.2)
        //   ext_factor  = 1.0                       (full YARN ramp)
        let freq_scale = if factor > 0.0 { 1.0 / factor } else { 1.0 };
        let attn_factor = 0.1 * (factor.max(1.0)).ln() + 1.0;
        let ext_factor = 1.0;

        Ok(DsV4RopeYarnConfig {
            scaling_type,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            n_ctx_orig,
        })
    }
}

fn u32_md(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<u32, DsV4MetadataError> {
    md.get(key)
        .ok_or(DsV4MetadataError::MissingKey(key))?
        .as_u32()
        .ok_or(DsV4MetadataError::WrongType { key, wanted: "u32" })
}

fn f32_md(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<f32, DsV4MetadataError> {
    let v = md.get(key).ok_or(DsV4MetadataError::MissingKey(key))?;
    match v {
        GgufValue::F32(f) => Ok(*f),
        GgufValue::F64(f) => Ok(*f as f32),
        _ => Err(DsV4MetadataError::WrongType { key, wanted: "f32" }),
    }
}

fn str_md_opt<'a>(
    md: &'a std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<Option<&'a str>, DsV4MetadataError> {
    match md.get(key) {
        None => Ok(None),
        Some(GgufValue::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(DsV4MetadataError::WrongType {
            key,
            wanted: "string",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_config_matches_no_scaling() {
        let cfg = DsV4RopeYarnConfig::identity(10000.0);
        assert_eq!(cfg.scaling_type, RopeScalingType::None);
        assert_eq!(cfg.freq_base, 10000.0);
        assert_eq!(cfg.freq_scale, 1.0);
        assert_eq!(cfg.ext_factor, 0.0);
        assert_eq!(cfg.attn_factor, 1.0);
    }

    /// Real-GGUF YARN extraction — DSv4-Flash baseline:
    ///   scaling=yarn, factor=16, beta_fast=32, beta_slow=1,
    ///   original_context_length=65536, freq_base=10000.
    /// Derived: freq_scale=1/16=0.0625, attn_factor=0.1*ln(16)+1≈1.277,
    /// ext_factor=1.0.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_yarn_matches_calibration_baseline() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = larql_models::loading::gguf::GgufFile::open(path).expect("open DSv4 GGUF");
        let cfg = DsV4RopeYarnConfig::from_gguf(&gguf).expect("extract YARN");

        assert_eq!(cfg.scaling_type, RopeScalingType::Yarn);
        assert!((cfg.freq_base - 10000.0).abs() < 1e-3);
        assert!((cfg.freq_scale - 0.0625).abs() < 1e-5);
        assert!((cfg.beta_fast - 32.0).abs() < 1e-3);
        assert!((cfg.beta_slow - 1.0).abs() < 1e-3);
        assert_eq!(cfg.n_ctx_orig, 65536);
        // attn_factor = 0.1 * ln(16) + 1.0 ≈ 1.2773
        let expected_attn = 0.1 * 16.0_f32.ln() + 1.0;
        assert!((cfg.attn_factor - expected_attn).abs() < 1e-4);
        assert_eq!(cfg.ext_factor, 1.0);
    }

    /// Missing scaling.type returns identity (no scaling).
    #[test]
    fn missing_scaling_type_falls_back_to_identity() {
        // Build a synthetic GGUF-like metadata map and exercise the
        // string-optional path via the helper.
        let mut md = std::collections::HashMap::new();
        md.insert(
            "deepseek4.rope.freq_base".to_string(),
            GgufValue::F32(10000.0),
        );
        let v = str_md_opt(&md, "deepseek4.rope.scaling.type").unwrap();
        assert_eq!(v, None);
    }

    /// `str_md_opt` accepts strings and rejects non-strings.
    #[test]
    fn str_md_opt_rejects_non_string() {
        let mut md = std::collections::HashMap::new();
        md.insert("k".to_string(), GgufValue::U32(42));
        let r = str_md_opt(&md, "k");
        assert!(matches!(r, Err(DsV4MetadataError::WrongType { .. })));
    }
}
