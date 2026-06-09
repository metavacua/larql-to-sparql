//! Extract [`DsV4Hyperparams`] from a real GGUF file's metadata +
//! layer-0 tensor shapes.
//!
//! GGUF metadata keys used (DSv4-Flash, calibrated 2026-05-23):
//! - `deepseek4.embedding_length`               → `n_embd`
//! - `deepseek4.attention.head_count`           → `n_head`
//! - `deepseek4.attention.key_length`           → `head_dim`
//! - `deepseek4.attention.q_lora_rank`          → `q_lora_rank`
//! - `deepseek4.attention.sliding_window`       → `window_size`
//! - `deepseek4.attention.layer_norm_rms_epsilon` → `norm_eps`
//! - `deepseek4.rope.dimension_count`           → `n_rot`
//! - `deepseek4.rope.freq_base`                 → `rope_base`
//! - `deepseek4.attention.indexer.head_count`   → `n_index_head`
//! - `deepseek4.attention.indexer.key_length`   → `indexer_head_size`
//! - `deepseek4.attention.indexer.top_k`        → `top_k`
//!
//! The schema doesn't carry `n_groups` and `o_lora_rank` directly — we
//! infer them from the shape of the layer-0 `attn_output_a` tensor:
//! `[group_dim, n_groups, o_lora_rank]` (ggml ne order). With
//! `n_head, head_dim, n_groups` known, `group_dim = head_dim *
//! (n_head / n_groups)`.

use larql_models::loading::gguf::{GgufFile, GgufValue};

use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage_build::DsV4Hyperparams;

/// Error returned by [`DsV4Hyperparams::from_gguf`].
#[derive(Debug)]
pub enum DsV4MetadataError {
    /// A required metadata key was missing.
    MissingKey(&'static str),
    /// A metadata key was present but the wrong type for its slot.
    WrongType {
        key: &'static str,
        wanted: &'static str,
    },
    /// `attn_output_a` for layer 0 wasn't present — needed to infer
    /// `n_groups` and `o_lora_rank`.
    MissingAttnOutputATensor,
    /// `attn_output_a` shape was unexpected (not 3D, dim mismatch, etc.).
    UnexpectedAttnOutputAShape {
        got: Vec<u64>,
        n_head: usize,
        head_dim: usize,
    },
}

impl std::fmt::Display for DsV4MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4MetadataError::MissingKey(k) => write!(f, "missing GGUF metadata key: {k}"),
            DsV4MetadataError::WrongType { key, wanted } => {
                write!(f, "GGUF metadata {key} has wrong type (wanted {wanted})")
            }
            DsV4MetadataError::MissingAttnOutputATensor => write!(
                f,
                "no `blk.0.attn_output_a.weight` tensor in GGUF — required to infer n_groups + o_lora_rank"
            ),
            DsV4MetadataError::UnexpectedAttnOutputAShape {
                got,
                n_head,
                head_dim,
            } => write!(
                f,
                "attn_output_a shape {got:?} not compatible with n_head={n_head}, head_dim={head_dim}; expected 3D [group_dim, n_groups, o_lora_rank]"
            ),
        }
    }
}

impl std::error::Error for DsV4MetadataError {}

impl DsV4Hyperparams {
    /// Extract [`DsV4Hyperparams`] from a real DSv4 GGUF.
    ///
    /// Reads metadata + the layer-0 `attn_output_a` tensor shape.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, DsV4MetadataError> {
        let md = &gguf.metadata;
        let n_embd = u32_md(md, "deepseek4.embedding_length")? as usize;
        let n_head = u32_md(md, "deepseek4.attention.head_count")? as usize;
        let head_dim = u32_md(md, "deepseek4.attention.key_length")? as usize;
        let q_lora_rank = u32_md(md, "deepseek4.attention.q_lora_rank")? as usize;
        let window_size = u32_md(md, "deepseek4.attention.sliding_window")? as usize;
        let norm_eps = f32_md(md, "deepseek4.attention.layer_norm_rms_epsilon")?;
        let n_rot = u32_md(md, "deepseek4.rope.dimension_count")? as usize;
        let rope_base = f32_md(md, "deepseek4.rope.freq_base")? as f64;

        // Indexer hyperparams: present in the DSv4-Flash GGUF; ignore
        // gracefully if absent (degenerate model without indexer).
        let indexer_head_size =
            u32_md_opt(md, "deepseek4.attention.indexer.key_length")?.map(|v| v as usize);
        let n_index_head =
            u32_md_opt(md, "deepseek4.attention.indexer.head_count")?.map(|v| v as usize);
        let top_k = u32_md_opt(md, "deepseek4.attention.indexer.top_k")?.map(|v| v as usize);

        // Infer n_groups + o_lora_rank from layer-0 attn_output_a shape.
        // Real GGUF stores it as 2D `[group_dim, n_groups * o_lora_rank]`
        // (ggml ne order). Derive via:
        //   n_groups   = head_dim * n_head / group_dim
        //   o_lora_rank = hidden / n_groups
        let info = gguf
            .tensor_infos
            .iter()
            .find(|i| i.name() == "blk.0.attn_output_a.weight")
            .ok_or(DsV4MetadataError::MissingAttnOutputATensor)?;
        let dims = info.dims();
        if dims.len() != 2 {
            return Err(DsV4MetadataError::UnexpectedAttnOutputAShape {
                got: dims.to_vec(),
                n_head,
                head_dim,
            });
        }
        let group_dim = dims[0] as usize;
        let hidden = dims[1] as usize;
        let total = head_dim * n_head;
        if group_dim == 0 || total % group_dim != 0 {
            return Err(DsV4MetadataError::UnexpectedAttnOutputAShape {
                got: dims.to_vec(),
                n_head,
                head_dim,
            });
        }
        let n_groups = total / group_dim;
        if n_groups == 0 || hidden % n_groups != 0 {
            return Err(DsV4MetadataError::UnexpectedAttnOutputAShape {
                got: dims.to_vec(),
                n_head,
                head_dim,
            });
        }
        let o_lora_rank = hidden / n_groups;

        // YARN scaling: extract if present, otherwise None.
        let yarn = match super::dsv4_yarn_config::DsV4RopeYarnConfig::from_gguf(gguf) {
            Ok(cfg) if cfg.scaling_type == super::dsv4_yarn_config::RopeScalingType::Yarn => {
                Some(cfg)
            }
            Ok(_) | Err(_) => None,
        };

        // FFN / MoE hyperparams.
        let n_expert = u32_md(md, "deepseek4.expert_count")? as usize;
        let n_expert_used = u32_md(md, "deepseek4.expert_used_count")? as usize;
        let n_ff_exp = u32_md(md, "deepseek4.expert_feed_forward_length")? as usize;
        let n_expert_shared = u32_md(md, "deepseek4.expert_shared_count")? as usize;
        let expert_weights_norm = bool_md_opt(md, "deepseek4.expert_weights_norm")?.unwrap_or(true);
        let expert_weights_scale = f32_md_opt(md, "deepseek4.expert_weights_scale")?.unwrap_or(1.0);

        // n_hc: number of hyper-connection streams. Derived from the
        // global `hc_head_base` tensor length (4 for DSv4-Flash). If
        // the tensor isn't present we default to 4 — every DSv4 model
        // observed so far has n_hc=4.
        let n_hc = gguf
            .tensor_infos
            .iter()
            .find(|i| i.name() == "hc_head_base")
            .and_then(|i| i.dims().first().copied())
            .map(|d| d as usize)
            .unwrap_or(4);

        // SWA-specific rope base (DSv4 separates the sliding-window
        // RoPE from the main one). Optional — when absent the SWA path
        // reuses `rope_base`.
        let rope_base_swa = match md.get("deepseek4.rope.freq_base_swa") {
            None => None,
            Some(GgufValue::F32(v)) => Some(*v as f64),
            Some(GgufValue::F64(v)) => Some(*v),
            Some(_) => {
                return Err(DsV4MetadataError::WrongType {
                    key: "deepseek4.rope.freq_base_swa",
                    wanted: "f32",
                });
            }
        };

        Ok(DsV4Hyperparams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot,
            rope_base,
            rope_mode: DsV4RopeMode::Neox,
            window_size,
            norm_eps,
            indexer_head_size,
            n_index_head,
            top_k,
            n_hc,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_expert_shared,
            expert_weights_norm,
            expert_weights_scale,
            yarn,
            rope_base_swa,
        })
    }
}

// ── GGUF metadata accessors ───────────────────────────────────────────

fn u32_md(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<u32, DsV4MetadataError> {
    md.get(key)
        .ok_or(DsV4MetadataError::MissingKey(key))?
        .as_u32()
        .ok_or(DsV4MetadataError::WrongType { key, wanted: "u32" })
}

fn u32_md_opt(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<Option<u32>, DsV4MetadataError> {
    match md.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_u32()
            .map(Some)
            .ok_or(DsV4MetadataError::WrongType { key, wanted: "u32" }),
    }
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

fn f32_md_opt(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<Option<f32>, DsV4MetadataError> {
    match md.get(key) {
        None => Ok(None),
        Some(GgufValue::F32(f)) => Ok(Some(*f)),
        Some(GgufValue::F64(f)) => Ok(Some(*f as f32)),
        Some(_) => Err(DsV4MetadataError::WrongType { key, wanted: "f32" }),
    }
}

fn bool_md_opt(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &'static str,
) -> Result<Option<bool>, DsV4MetadataError> {
    match md.get(key) {
        None => Ok(None),
        Some(GgufValue::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(DsV4MetadataError::WrongType {
            key,
            wanted: "bool",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke against the real DSv4-Flash GGUF. Skipped unless the file
    /// exists at the canonical path. Calibration baseline:
    /// n_embd=4096, n_head=64, head_dim=512, q_lora_rank=1024,
    /// n_groups=8, o_lora_rank=1024, n_rot=64, window_size=128,
    /// rope_base=10000, indexer_head_size=128, n_index_head=64, top_k=512.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_hyperparams_match_calibration_baseline() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("extract hyperparams");
        assert_eq!(hp.n_embd, 4096);
        assert_eq!(hp.n_head, 64);
        assert_eq!(hp.head_dim, 512);
        assert_eq!(hp.q_lora_rank, 1024);
        assert_eq!(hp.n_groups, 8);
        assert_eq!(hp.o_lora_rank, 1024);
        assert_eq!(hp.n_rot, 64);
        assert_eq!(hp.window_size, 128);
        assert!((hp.rope_base - 10000.0).abs() < 1e-3);
        assert_eq!(hp.indexer_head_size, Some(128));
        assert_eq!(hp.n_index_head, Some(64));
        assert_eq!(hp.top_k, Some(512));

        // YARN: DSv4-Flash has factor=16, beta_fast=32, beta_slow=1,
        // n_ctx_orig=65536. hp.yarn must be Some.
        let yarn = hp.yarn.expect("yarn config present");
        assert!((yarn.freq_scale - 0.0625).abs() < 1e-5);
        assert_eq!(yarn.n_ctx_orig, 65536);
        assert!((yarn.beta_fast - 32.0).abs() < 1e-3);
        assert!((yarn.beta_slow - 1.0).abs() < 1e-3);

        // SWA-specific rope base: 160 000.0 vs 10 000.0 main.
        let rope_swa = hp.rope_base_swa.expect("rope_base_swa present");
        assert!((rope_swa - 160_000.0).abs() < 1e-3);

        // FFN / MoE hyperparams: DSv4-Flash uses 256 experts, top-6
        // routing, 2048 hidden per expert, 1 shared expert,
        // expert_weights_norm=true, expert_weights_scale=1.5.
        assert_eq!(hp.n_expert, 256);
        assert_eq!(hp.n_expert_used, 6);
        assert_eq!(hp.n_ff_exp, 2048);
        assert_eq!(hp.n_expert_shared, 1);
        assert!(hp.expert_weights_norm);
        assert!((hp.expert_weights_scale - 1.5).abs() < 1e-3);

        // n_hc derived from hc_head_base.
        assert_eq!(hp.n_hc, 4);
    }

    /// Missing-key errors carry the key name so callers can pin down
    /// which field of the GGUF wasn't present.
    #[test]
    fn missing_key_error_carries_name() {
        let err = DsV4MetadataError::MissingKey("deepseek4.embedding_length");
        let msg = format!("{err}");
        assert!(msg.contains("deepseek4.embedding_length"), "got: {msg}");
    }

    /// Wrong-type error names the type wanted.
    #[test]
    fn wrong_type_error_carries_wanted() {
        let err = DsV4MetadataError::WrongType {
            key: "deepseek4.rope.freq_base",
            wanted: "f32",
        };
        let msg = format!("{err}");
        assert!(msg.contains("freq_base"), "got: {msg}");
        assert!(msg.contains("f32"), "got: {msg}");
    }

    /// `u32_md_opt` returns Ok(None) for missing keys (not an error),
    /// preserving the indexer-optional path.
    #[test]
    fn u32_md_opt_handles_missing() {
        let md: std::collections::HashMap<String, GgufValue> = std::collections::HashMap::new();
        let v = u32_md_opt(&md, "some.missing.key").unwrap();
        assert_eq!(v, None);
    }

    /// `f32_md` accepts both F32 and F64 GGUF values (some converters
    /// emit f64 for floats; we silently downcast).
    #[test]
    fn f32_md_accepts_f64() {
        let mut md = std::collections::HashMap::new();
        md.insert("k".to_string(), GgufValue::F64(1.25));
        let v = f32_md(&md, "k").unwrap();
        assert!((v - 1.25).abs() < 1e-6);
    }
}
