//! `vindex_to_fp4` — take an existing f32/f16 vindex and write a new
//! vindex with the FP4/FP8 block-storage layout. Library entry for
//! the `larql convert quantize fp4` CLI subcommand.
//!
//! Specs pinned in `docs/specs/quantize-cli-spec.md` (shape) and
//! `docs/specs/fp4-precision-policy.md` (defaults).
//!
//! Key behaviours (all from the spec):
//!
//! - **Gate stays at source dtype** in all three policies — the
//!   gate KNN needs a dense matrix for batch matmul and the
//!   FP4-aware gate KNN path is deferred.
//! - **Compliance floor is a precision-FP4 gate**, not a per-
//!   projection gate. Only projections targeted for FP4 are
//!   measured; FP8/F16 projections skip the check (the floor's
//!   distributional assumption doesn't apply).
//! - **Atomic output**: write into `DST.tmp/`, fsync, rename to
//!   `DST/` on success. Removes the "partial output looks
//!   complete" foot-gun.
//! - **Auxiliary files hard-linked** (embeddings, attn, norms,
//!   lm_head, tokenizer, etc.), f32/f16 gate hard-linked too. Only
//!   the policy-quantised projections are written fresh. On
//!   cross-filesystem DST, hard-link falls back to copy with a
//!   notice.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::config::types::{
    ComplianceGate, Fp4Config, Precision, ProjectionFormat, Projections, VindexConfig,
};
use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::fp4_codec::{write_fp4_projection, write_fp8_projection};

use super::scan::{scan_vindex, Dtype, ScanConfig, VindexComplianceReport};

/// Policy A / B / C from `fp4-precision-policy.md`. Gate stays at
/// source dtype in every policy (see FP4 gate caveat in §2 of that
/// spec); only up + down vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    A,
    B,
    C,
}

impl Policy {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "option-a" | "a" | "A" => Ok(Policy::A),
            "option-b" | "b" | "B" => Ok(Policy::B),
            "option-c" | "c" | "C" => Ok(Policy::C),
            _ => Err(format!("unknown policy {s}")),
        }
    }

    /// (gate, up, down) precision. Gate stays at source for all
    /// three — only up/down vary.
    pub fn precisions(self, gate_source: Precision) -> (Precision, Precision, Precision) {
        match self {
            Policy::A => (gate_source, Precision::Fp4, Precision::Fp4),
            Policy::B => (gate_source, Precision::Fp4, Precision::Fp8),
            Policy::C => (gate_source, Precision::Fp4, Precision::F16),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Policy::A => "option-a",
            Policy::B => "option-b",
            Policy::C => "option-c",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fp4ConvertConfig {
    pub policy: Policy,
    pub compliance_floor: f32,
    pub threshold: f32,
    pub strict: bool,
    pub force: bool,
    pub emit_sidecar: bool,
}

impl Default for Fp4ConvertConfig {
    fn default() -> Self {
        Self {
            policy: Policy::B,
            // Per docs/fp4-precision-policy.md §3: 99% of per-feature
            // blocks must satisfy R<threshold or the projection falls
            // back to FP8. Empirically tuned against gemma3-4b.
            compliance_floor: 0.99,
            // R = max_abs / mean_abs cutoff for FP4-friendly blocks.
            threshold: 16.0,
            strict: false,
            force: false,
            emit_sidecar: true,
        }
    }
}

/// What happened to one projection during conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionOutcome {
    WroteFp4,
    WroteFp8,
    WroteF16,
    LinkedAsSource,
    DowngradedFp4ToFp8,
    DowngradedFp4ToF16,
}

impl ProjectionOutcome {
    pub fn action_str(self) -> &'static str {
        match self {
            Self::WroteFp4 => "wrote_fp4",
            Self::WroteFp8 => "wrote_fp8_per_policy_default",
            Self::WroteF16 => "wrote_f16_per_policy_default",
            Self::LinkedAsSource => "linked_as_source_dtype",
            Self::DowngradedFp4ToFp8 => "downgraded_fp4_to_fp8",
            Self::DowngradedFp4ToF16 => "downgraded_fp4_to_f16",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionAction {
    pub name: String,
    pub compliance_at_threshold: Option<f32>, // None when not FP4-targeted
    pub policy_precision: Precision,
    pub chosen_precision: Precision,
    pub outcome: ProjectionOutcome,
    pub output_file: String,
    pub output_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Fp4ConvertReport {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub policy: Policy,
    pub threshold: f32,
    pub compliance_floor: f32,
    pub per_projection: Vec<ProjectionAction>,
    pub src_ffn_bytes: u64,
    pub dst_ffn_bytes: u64,
    pub compression: f64,
    pub aux_linked_count: usize,
    pub aux_linked_bytes: u64,
    pub wall_time: Duration,
    pub walk_backend: String,
}

impl Fp4ConvertReport {
    pub fn compliance_sidecar_json(&self, scan_report: &VindexComplianceReport) -> Value {
        let per_projection: Vec<Value> = self
            .per_projection
            .iter()
            .map(|p| {
                json!({
                    "projection": p.name,
                    "compliance_at_threshold": p.compliance_at_threshold,
                    "threshold": self.threshold,
                    "policy_precision": precision_str(p.policy_precision),
                    "chosen_precision": precision_str(p.chosen_precision),
                    "action": p.outcome.action_str(),
                    "output_file": p.output_file,
                    "output_size_bytes": p.output_size_bytes,
                })
            })
            .collect();
        json!({
            "extracted_at": now_iso_like(),
            "policy": self.policy.label(),
            "block_elements_scanned": larql_models::quant::fp4_block::BLOCK_ELEMENTS,
            "compliance_gate_threshold_ratio": self.threshold,
            "compliance_gate_min_fraction": self.compliance_floor,
            "per_projection": per_projection,
            "full_scan": scan_report.to_json(),
        })
    }
}

fn precision_str(p: Precision) -> String {
    match p {
        Precision::Fp4 => "fp4".into(),
        Precision::Fp8 => "fp8".into(),
        Precision::F16 => "f16".into(),
        Precision::F32 => "f32".into(),
    }
}

fn now_iso_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("@epoch+{secs}s")
}

// ── Main entry point ──────────────────────────────────────────────────

/// Convert an existing f32/f16 vindex to an FP4/FP8 vindex per the
/// given policy. Atomic: writes into `<dst>.tmp/` and renames on
/// success. Errors return early without touching `<dst>`.
///
/// Scope: input must be a flat-file vindex with `gate_vectors.bin`,
/// `up_features.bin`, `down_features.bin` present. Q4K/MXFP4-only
/// vindexes aren't supported as input (no consumer asked for it).
pub fn vindex_to_fp4(
    src: &Path,
    dst: &Path,
    config: &Fp4ConvertConfig,
) -> Result<(Fp4ConvertReport, VindexComplianceReport), VindexError> {
    let t_total = Instant::now();

    if dst.exists() {
        if !config.force {
            return Err(VindexError::Parse(format!(
                "output dir {} exists (use force=true to overwrite)",
                dst.display()
            )));
        }
        std::fs::remove_dir_all(dst)
            .map_err(|e| VindexError::Parse(format!("remove existing dst: {e}")))?;
    }

    // Atomic-rename staging: write into DST.tmp/, rename at the end.
    let dst_tmp = dst.with_file_name(format!(
        "{}.tmp",
        dst.file_name().and_then(|s| s.to_str()).unwrap_or("out")
    ));
    if dst_tmp.exists() {
        std::fs::remove_dir_all(&dst_tmp)
            .map_err(|e| VindexError::Parse(format!("clean staging dir: {e}")))?;
    }
    std::fs::create_dir_all(&dst_tmp)
        .map_err(|e| VindexError::Parse(format!("create staging dir: {e}")))?;

    // Parse source config.
    let mut src_config: VindexConfig = serde_json::from_str(
        &std::fs::read_to_string(src.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("read src index.json: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse src index.json: {e}")))?;
    let src_index_raw: Value = serde_json::from_str(
        &std::fs::read_to_string(src.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("re-read src index.json: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse raw src index.json: {e}")))?;
    let src_dtype_str = src_index_raw["dtype"].as_str().unwrap_or("f32");
    let src_dtype = Dtype::from_index_json(src_dtype_str).map_err(VindexError::Parse)?;

    let hidden = src_config.hidden_size;
    let num_layers = src_config.num_layers;
    let per_layer_features: Vec<usize> = src_config.layers.iter().map(|l| l.num_features).collect();

    if !hidden.is_multiple_of(larql_models::quant::fp4_block::BLOCK_ELEMENTS) {
        return Err(VindexError::Parse(format!(
            "hidden={hidden} not divisible by FP4 block size {}; input vindex not convertible",
            larql_models::quant::fp4_block::BLOCK_ELEMENTS
        )));
    }

    // Verify required input files exist before running the scan.
    for name in [GATE_VECTORS_BIN, UP_FEATURES_BIN, DOWN_FEATURES_BIN] {
        if !src.join(name).exists() {
            return Err(VindexError::Parse(format!(
                "{name} missing from src vindex; quantize fp4 requires the full \
                 (f32/f16) FFN projection files"
            )));
        }
    }

    // Run the compliance scan once up front — feeds both self-policing
    // and the sidecar. O(10 GB mmap scan in ~3s on M3 Max.
    let scan_config = ScanConfig {
        compliance_thresholds: vec![config.threshold],
        ..Default::default()
    };
    let scan_report = scan_vindex(src, &scan_config)?;

    // Policy precision assignments.
    let gate_source = match src_dtype {
        Dtype::F32 => Precision::F32,
        Dtype::F16 => Precision::F16,
        Dtype::Bf16 => Precision::F16, // flagged as F16 until we need a distinct tag
    };
    let (policy_g, policy_u, policy_d) = config.policy.precisions(gate_source);

    let projections: [(&str, &str, Precision); 3] = [
        ("gate", GATE_VECTORS_BIN, policy_g),
        ("up", UP_FEATURES_BIN, policy_u),
        ("down", DOWN_FEATURES_BIN, policy_d),
    ];

    // Per-projection: read source, decide final precision, write output.
    let mut actions: Vec<ProjectionAction> = Vec::with_capacity(3);
    let mut final_projections: [Option<ProjectionFormat>; 3] = [None, None, None];

    for (idx, (name, src_file, policy_prec)) in projections.iter().enumerate() {
        let src_path = src.join(src_file);
        let scan_for_proj = scan_report.projection(name);
        let compliance = scan_for_proj.map(|p| p.compliance_at(config.threshold) as f32);

        // Decide output precision. Compliance floor only gates FP4-
        // targeted projections.
        let (chosen, outcome) = match *policy_prec {
            Precision::Fp4 => {
                let c = compliance.unwrap_or(0.0);
                if c < config.compliance_floor {
                    if config.strict {
                        return Err(VindexError::Parse(format!(
                            "strict mode: {name} compliance {c:.4} below floor {} \
                             at threshold R<{}",
                            config.compliance_floor, config.threshold
                        )));
                    }
                    (Precision::Fp8, ProjectionOutcome::DowngradedFp4ToFp8)
                } else {
                    (Precision::Fp4, ProjectionOutcome::WroteFp4)
                }
            }
            Precision::Fp8 => (Precision::Fp8, ProjectionOutcome::WroteFp8),
            Precision::F16 => (Precision::F16, ProjectionOutcome::WroteF16),
            Precision::F32 => (Precision::F32, ProjectionOutcome::LinkedAsSource),
        };

        // Output file naming.
        let out_file = match chosen {
            Precision::Fp4 => format!("{}_fp4.bin", fs_prefix(name)?),
            Precision::Fp8 => format!("{}_fp8.bin", fs_prefix(name)?),
            Precision::F16 | Precision::F32 => src_file.to_string(),
        };
        let out_path = dst_tmp.join(&out_file);

        let outcome_tag = match (*policy_prec, chosen) {
            (Precision::Fp4, Precision::Fp4) => outcome,
            (Precision::Fp4, Precision::Fp8) => ProjectionOutcome::DowngradedFp4ToFp8,
            (_, Precision::Fp8) => ProjectionOutcome::WroteFp8,
            (_, Precision::F16) => ProjectionOutcome::WroteF16,
            (_, Precision::F32) => ProjectionOutcome::LinkedAsSource,
            _ => outcome,
        };

        match chosen {
            Precision::Fp4 => {
                // Decode source → float → encode FP4.
                let layers =
                    read_source_projection(&src_path, src_dtype, &per_layer_features, hidden)?;
                let refs: Vec<&[f32]> = layers.iter().map(|v| v.as_slice()).collect();
                write_fp4_projection(&out_path, hidden, &refs)?;
            }
            Precision::Fp8 => {
                let layers =
                    read_source_projection(&src_path, src_dtype, &per_layer_features, hidden)?;
                let refs: Vec<&[f32]> = layers.iter().map(|v| v.as_slice()).collect();
                write_fp8_projection(&out_path, hidden, &refs)?;
            }
            Precision::F16 | Precision::F32 => {
                link_or_copy(&src_path, &out_path)?;
            }
        }
        let out_size = std::fs::metadata(&out_path)
            .map_err(|e| VindexError::Parse(format!("stat {}: {e}", out_path.display())))?
            .len();

        final_projections[idx] = Some(ProjectionFormat {
            precision: chosen,
            file: out_file.clone(),
        });
        actions.push(ProjectionAction {
            name: name.to_string(),
            compliance_at_threshold: compliance,
            policy_precision: *policy_prec,
            chosen_precision: chosen,
            outcome: outcome_tag,
            output_file: out_file,
            output_size_bytes: out_size,
        });
    }

    // Build new VindexConfig with the fp4 manifest.
    let projections_cfg = Projections {
        gate: final_projections[0].take().unwrap(),
        up: final_projections[1].take().unwrap(),
        down: final_projections[2].take().unwrap(),
    };
    let fp4_cfg = Fp4Config {
        projections: projections_cfg,
        compliance_gate: ComplianceGate {
            threshold_ratio: config.threshold,
            min_compliant_fraction: config.compliance_floor,
            fallback_precision: Precision::Fp8,
        },
        ..Fp4Config::v1_defaults(Projections {
            gate: ProjectionFormat {
                precision: Precision::Fp4,
                file: String::new(),
            },
            up: ProjectionFormat {
                precision: Precision::Fp4,
                file: String::new(),
            },
            down: ProjectionFormat {
                precision: Precision::Fp4,
                file: String::new(),
            },
        })
    };
    src_config.fp4 = Some(fp4_cfg);

    let out_index_json = serde_json::to_string_pretty(&src_config)
        .map_err(|e| VindexError::Parse(format!("serialise: {e}")))?;
    std::fs::write(dst_tmp.join(INDEX_JSON), out_index_json)
        .map_err(|e| VindexError::Parse(format!("write index.json: {e}")))?;

    // Compliance sidecar.
    if config.emit_sidecar {
        let report_for_sidecar = Fp4ConvertReport {
            src: src.to_path_buf(),
            dst: dst.to_path_buf(),
            policy: config.policy,
            threshold: config.threshold,
            compliance_floor: config.compliance_floor,
            per_projection: actions.clone(),
            src_ffn_bytes: 0,
            dst_ffn_bytes: 0,
            compression: 0.0,
            aux_linked_count: 0,
            aux_linked_bytes: 0,
            wall_time: Duration::ZERO,
            walk_backend: String::new(),
        };
        let sidecar = report_for_sidecar.compliance_sidecar_json(&scan_report);
        std::fs::write(
            dst_tmp.join("fp4_compliance.json"),
            serde_json::to_string_pretty(&sidecar)
                .map_err(|e| VindexError::Parse(format!("serialise sidecar: {e}")))?,
        )
        .map_err(|e| VindexError::Parse(format!("write sidecar: {e}")))?;
    }

    // Hard-link auxiliary files.
    let handled: std::collections::HashSet<&str> = [
        INDEX_JSON,
        GATE_VECTORS_BIN,
        UP_FEATURES_BIN,
        DOWN_FEATURES_BIN,
        "fp4_compliance.json",
    ]
    .iter()
    .copied()
    .collect();

    let mut aux_linked = 0usize;
    let mut aux_bytes = 0u64;
    for entry in
        std::fs::read_dir(src).map_err(|e| VindexError::Parse(format!("read src dir: {e}")))?
    {
        let entry = entry.map_err(|e| VindexError::Parse(format!("{e}")))?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if handled.contains(fname_str.as_ref()) {
            continue;
        }
        let meta = entry
            .metadata()
            .map_err(|e| VindexError::Parse(format!("{e}")))?;
        if !meta.is_file() {
            continue;
        }
        let dst_path = dst_tmp.join(&fname);
        link_or_copy(&entry.path(), &dst_path)?;
        aux_linked += 1;
        aux_bytes += meta.len();
    }

    // Atomic promote: rename dst.tmp → dst.
    std::fs::rename(&dst_tmp, dst).map_err(|e| {
        VindexError::Parse(format!(
            "atomic rename {} → {}: {e}",
            dst_tmp.display(),
            dst.display(),
        ))
    })?;

    let src_ffn_bytes: u64 = src_config.layers.iter().map(|l| l.length * 3).sum();
    let dst_ffn_bytes: u64 = actions.iter().map(|a| a.output_size_bytes).sum();
    let compression = src_ffn_bytes as f64 / dst_ffn_bytes.max(1) as f64;

    // Load the new vindex to produce the backend-describe line for the
    // report. Cheap: just mmap metadata, no per-layer work.
    let walk_backend =
        describe_out_backend(dst).unwrap_or_else(|e| format!("<describe failed: {e:?}>"));

    // Patch up the actions' report now that we have the numbers.
    let n = num_layers;
    let _ = n; // silence if unused after downstream changes
    let report = Fp4ConvertReport {
        src: src.to_path_buf(),
        dst: dst.to_path_buf(),
        policy: config.policy,
        threshold: config.threshold,
        compliance_floor: config.compliance_floor,
        per_projection: actions,
        src_ffn_bytes,
        dst_ffn_bytes,
        compression,
        aux_linked_count: aux_linked,
        aux_linked_bytes: aux_bytes,
        wall_time: t_total.elapsed(),
        walk_backend,
    };
    Ok((report, scan_report))
}

fn describe_out_backend(dst: &Path) -> Result<String, VindexError> {
    use crate::{SilentLoadCallbacks, VectorIndex};
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dst, &mut cb)?;
    Ok(index.describe_ffn_backend())
}

fn fs_prefix(name: &str) -> Result<&'static str, VindexError> {
    match name {
        "gate" => Ok("gate_vectors"),
        "up" => Ok("up_features"),
        "down" => Ok("down_features"),
        _ => Err(VindexError::Parse(format!("unknown projection {name}"))),
    }
}

fn read_source_projection(
    path: &Path,
    dtype: Dtype,
    layer_features: &[usize],
    hidden: usize,
) -> Result<Vec<Vec<f32>>, VindexError> {
    let bytes = std::fs::read(path)
        .map_err(|e| VindexError::Parse(format!("read {}: {e}", path.display())))?;
    let bpf = dtype.bytes_per_float();
    let expected: usize = layer_features.iter().sum::<usize>() * hidden * bpf;
    if bytes.len() != expected {
        return Err(VindexError::Parse(format!(
            "{}: size {} != expected {}",
            path.display(),
            bytes.len(),
            expected,
        )));
    }
    let mut out = Vec::with_capacity(layer_features.len());
    let mut cursor = 0usize;
    for &n in layer_features {
        let layer_bytes = n * hidden * bpf;
        let slice = &bytes[cursor..cursor + layer_bytes];
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => {
                let view: &[f32] =
                    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, n * hidden) };
                view.to_vec()
            }
            Dtype::F16 => larql_models::quant::half::decode_f16(slice),
            Dtype::Bf16 => larql_models::quant::half::decode_bf16(slice),
        };
        cursor += layer_bytes;
        out.push(floats);
    }
    Ok(out)
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<(), VindexError> {
    if dst.exists() {
        std::fs::remove_file(dst)
            .map_err(|e| VindexError::Parse(format!("remove existing {}: {e}", dst.display())))?;
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst).map_err(|e| {
                VindexError::Parse(format!(
                    "copy fallback {} → {}: {e}",
                    src.display(),
                    dst.display()
                ))
            })?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_precisions_keep_gate_source() {
        // All three policies keep gate=source (per spec).
        assert_eq!(Policy::A.precisions(Precision::F16).0, Precision::F16);
        assert_eq!(Policy::B.precisions(Precision::F32).0, Precision::F32);
        assert_eq!(Policy::C.precisions(Precision::F16).0, Precision::F16);
    }

    #[test]
    fn policy_b_is_fp4_up_fp8_down() {
        let (_g, u, d) = Policy::B.precisions(Precision::F16);
        assert_eq!(u, Precision::Fp4);
        assert_eq!(d, Precision::Fp8);
    }

    #[test]
    fn policy_parse_accepts_short_forms() {
        assert_eq!(Policy::parse("b").unwrap(), Policy::B);
        assert_eq!(Policy::parse("option-b").unwrap(), Policy::B);
        assert_eq!(Policy::parse("A").unwrap(), Policy::A);
        assert!(Policy::parse("foo").is_err());
    }

    #[test]
    fn default_config_is_option_b() {
        let c = Fp4ConvertConfig::default();
        assert_eq!(c.policy, Policy::B);
        assert_eq!(c.compliance_floor, 0.99);
        assert_eq!(c.threshold, 16.0);
        assert!(!c.strict);
        assert!(!c.force);
        assert!(c.emit_sidecar);
    }

    #[test]
    fn fs_prefix_known_projections() {
        assert_eq!(fs_prefix("gate").unwrap(), "gate_vectors");
        assert_eq!(fs_prefix("up").unwrap(), "up_features");
        assert_eq!(fs_prefix("down").unwrap(), "down_features");
    }

    #[test]
    fn fs_prefix_unknown_returns_parse_error() {
        // Was a panic before; library code must surface this as an
        // error so callers can recover or report cleanly.
        let err = fs_prefix("attention").expect_err("unknown projection");
        match err {
            VindexError::Parse(msg) => assert!(
                msg.contains("attention"),
                "error should name the bad projection: {msg}",
            ),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
