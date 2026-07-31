//! Convert an existing f32/f16 vindex into an FP4/FP8 vindex.
//!
//! - Reads source gate/up/down projection files, decodes to f32.
//! - Runs the Q1 compliance scan per projection.
//! - Applies the policy (Option B default: gate/up FP4, down FP8) with
//!   the self-policing compliance gate: any projection whose compliance
//!   falls below `--compliance-floor` at `--threshold` is downgraded to
//!   the fallback precision rather than committed as-is.
//! - Writes a new vindex directory with:
//!     - `index.json` carrying the `fp4` manifest
//!     - `gate_vectors_fp4.bin` / `up_features_fp4.bin` / `down_features_fp8.bin`
//!     - `fp4_compliance.json` sidecar (full scan + per-projection actions)
//! - Hard-links (or copies on failure) all non-FFN files (embeddings,
//!   attention, norms, tokenizer, etc.) so the output is self-contained.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p larql-vindex --example fp4_convert -- \
//!   --in  output/gemma3-4b-f16.vindex \
//!   --out output/gemma3-4b-fp4.vindex \
//!   --policy option-b
//! ```
//!
//! Flags:
//!   --policy option-a | option-b | option-c  (default: option-b)
//!   --compliance-floor 0.99                  (default; 0.0 disables the gate)
//!   --threshold 16.0                         (ratio threshold; see policy spec §2)
//!   --force                                  (overwrite existing output dir)

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_models::quant::fp4_block::BLOCK_ELEMENTS;
use larql_vindex::{
    ComplianceGate, Fp4Config, Precision, ProjectionFormat, Projections, VindexConfig,
};
use serde_json::{json, Value};

// ── Args ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Policy {
    A,
    B,
    C,
}

impl Policy {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "option-a" | "a" => Ok(Policy::A),
            "option-b" | "b" => Ok(Policy::B),
            "option-c" | "c" => Ok(Policy::C),
            _ => Err(format!("unknown policy {s}")),
        }
    }

    /// (gate, up, down) precision under this policy.
    ///
    /// **Architectural note (exp 26 Q2 finding):** gate is always kept
    /// at source dtype (f32/f16) rather than FP4. The walk kernel's
    /// gate KNN (`gate_scores_batch`, `gate_walk`) requires a dense
    /// gate matrix for batch matmul — per-feature FP4 gate access
    /// would bypass this entirely. FP4-storing gate saves ~25% of FFN
    /// storage in theory but has no consumer in the current walk
    /// kernel; the savings would stay on disk and never translate to
    /// bandwidth gains in memory-bound inference.
    ///
    /// Options labelled A/B/C in the policy spec now apply only to
    /// the up/down projections. Gate stays at whatever dtype the
    /// source vindex used, hard-linked by the converter.
    fn precisions(self, gate_source: Precision) -> (Precision, Precision, Precision) {
        match self {
            Policy::A => (gate_source, Precision::Fp4, Precision::Fp4),
            Policy::B => (gate_source, Precision::Fp4, Precision::Fp8),
            Policy::C => (gate_source, Precision::Fp4, Precision::F16),
        }
    }
}

struct Args {
    in_path: PathBuf,
    out_path: PathBuf,
    policy: Policy,
    compliance_floor: f32,
    threshold: f32,
    force: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut in_path = None;
    let mut out_path = None;
    let mut policy = Policy::B;
    let mut compliance_floor = 0.99f32;
    let mut threshold = 16.0f32;
    let mut force = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                in_path = Some(PathBuf::from(&args[i]));
            }
            "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(&args[i]));
            }
            "--policy" => {
                i += 1;
                policy = Policy::parse(&args[i]).expect("policy");
            }
            "--compliance-floor" => {
                i += 1;
                compliance_floor = args[i].parse().expect("float");
            }
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().expect("float");
            }
            "--force" => {
                force = true;
            }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }
    let in_path = in_path.unwrap_or_else(|| {
        eprintln!("usage: fp4_convert --in SRC --out DST [--policy option-b] [--force]");
        std::process::exit(1);
    });
    let out_path = out_path.unwrap_or_else(|| {
        eprintln!("usage: fp4_convert --in SRC --out DST [--policy option-b] [--force]");
        std::process::exit(1);
    });
    Args {
        in_path,
        out_path,
        policy,
        compliance_floor,
        threshold,
        force,
    }
}

// ── Source reader (f32 or f16) ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum SrcDtype {
    F32,
    F16,
    Bf16,
}

impl SrcDtype {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            _ => Err(format!("unsupported source dtype: {s}")),
        }
    }
    fn bytes_per_float(self) -> usize {
        match self {
            Self::F32 => 4,
            _ => 2,
        }
    }
}

/// Read a whole projection file (layer-concatenated, feature-major) and
/// return per-layer flat f32 data.
fn read_source_projection(
    path: &Path,
    dtype: SrcDtype,
    per_layer_features: &[usize],
    hidden: usize,
) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("read source projection");
    let bpf = dtype.bytes_per_float();
    let expected: usize = per_layer_features.iter().sum::<usize>() * hidden * bpf;
    assert_eq!(
        bytes.len(),
        expected,
        "{}: size {} != expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut out = Vec::with_capacity(per_layer_features.len());
    let mut cursor = 0usize;
    for &n in per_layer_features {
        let layer_bytes = n * hidden * bpf;
        let slice = &bytes[cursor..cursor + layer_bytes];
        let floats: Vec<f32> = match dtype {
            SrcDtype::F32 => {
                // SAFETY: in-memory Vec, u8→f32 reinterpret is safe because
                // f32 has no alignment requirement above u8 for read.
                let view: &[f32] =
                    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, n * hidden) };
                view.to_vec()
            }
            SrcDtype::F16 => larql_models::quant::half::decode_f16(slice),
            SrcDtype::Bf16 => larql_models::quant::half::decode_bf16(slice),
        };
        cursor += layer_bytes;
        out.push(floats);
    }
    out
}

// ── Compliance scan ───────────────────────────────────────────────────────────

/// Fraction of per-feature blocks whose max/min non-zero sub-block
/// scale ratio is below `threshold`. Matches the scanner's "per-feature
/// block" granularity at 256-element sub-feature tiles.
fn compliance_fraction(layers: &[Vec<f32>], hidden: usize, threshold: f32) -> f64 {
    let mut total: u64 = 0;
    let mut compliant: u64 = 0;
    const SB: usize = 32;
    for layer in layers {
        assert!(layer.len() % hidden == 0);
        let n_features = layer.len() / hidden;
        for f in 0..n_features {
            let feat = &layer[f * hidden..(f + 1) * hidden];
            // Scales per sub-block, then treat one whole feature as one
            // "block" for the per-feature granularity. Matches scanner §5.1.
            let mut mx = 0.0f32;
            let mut mn = f32::INFINITY;
            let mut any_nonzero = false;
            for sb in feat.chunks_exact(SB) {
                let s = sb.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                if s > 0.0 {
                    any_nonzero = true;
                    if s > mx {
                        mx = s;
                    }
                    if s < mn {
                        mn = s;
                    }
                }
            }
            total += 1;
            if !any_nonzero {
                compliant += 1; // all-zero block: trivially lossless.
            } else if mx / mn < threshold {
                compliant += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        compliant as f64 / total as f64
    }
}

// ── File copy/link ────────────────────────────────────────────────────────────

fn link_or_copy(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_file(dst)?;
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst)?;
            Ok(())
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    if args.out_path.exists() {
        if !args.force {
            return Err(format!(
                "output dir {} exists (use --force to overwrite)",
                args.out_path.display()
            )
            .into());
        }
        std::fs::remove_dir_all(&args.out_path)?;
    }
    std::fs::create_dir_all(&args.out_path)?;

    // ── Read source index.json ───────────────────────────────────────────────
    let src_index: Value =
        serde_json::from_str(&std::fs::read_to_string(args.in_path.join("index.json"))?)?;
    let mut src_config: VindexConfig =
        serde_json::from_str(&std::fs::read_to_string(args.in_path.join("index.json"))?)?;

    let num_layers = src_config.num_layers;
    let hidden = src_config.hidden_size;
    let per_layer_features: Vec<usize> = src_config.layers.iter().map(|l| l.num_features).collect();
    let src_dtype = SrcDtype::from_str(src_index["dtype"].as_str().unwrap_or("f32"))?;

    if !hidden.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(format!(
            "hidden={hidden} not divisible by block size {BLOCK_ELEMENTS}; FP4 format unsupported for this model"
        ).into());
    }

    let gate_src = args.in_path.join("gate_vectors.bin");
    let up_src = args.in_path.join("up_features.bin");
    let down_src = args.in_path.join("down_features.bin");
    for (name, p) in [("gate", &gate_src), ("up", &up_src), ("down", &down_src)] {
        if !p.exists() {
            return Err(format!(
                "{name}: {} not present — fp4_convert requires an unquantised vindex with gate_vectors.bin, up_features.bin, down_features.bin",
                p.display()
            ).into());
        }
    }

    println!("== fp4_convert ==");
    println!("  src   : {}", args.in_path.display());
    println!("  dst   : {}", args.out_path.display());
    println!("  model : {}", src_config.model);
    println!("  layers: {num_layers}  hidden: {hidden}  dtype: {src_dtype:?}");
    println!(
        "  policy: {:?}  floor: {}  threshold: {}",
        args.policy, args.compliance_floor, args.threshold
    );
    println!();

    // ── Read + quantise each projection ──────────────────────────────────────
    let t_total = Instant::now();
    let mut compliance_entries: Vec<Value> = Vec::new();
    let gate_source_precision = match src_dtype {
        SrcDtype::F32 => Precision::F32,
        SrcDtype::F16 => Precision::F16,
        SrcDtype::Bf16 => Precision::F16, // stored as bf16 but flagged as F16 for now
    };
    let (policy_g, policy_u, policy_d) = args.policy.precisions(gate_source_precision);

    let projections = [
        ("gate", "gate_vectors.bin", policy_g),
        ("up", "up_features.bin", policy_u),
        ("down", "down_features.bin", policy_d),
    ];

    let mut final_projections: [Option<ProjectionFormat>; 3] = [None, None, None];

    for (idx, (name, src_file, policy_prec)) in projections.iter().enumerate() {
        let t_proj = Instant::now();
        let src_path = args.in_path.join(src_file);
        println!("→ {name}: reading {}", src_path.display());
        let layers = read_source_projection(&src_path, src_dtype, &per_layer_features, hidden);
        println!("  decoded in {:.1}s", t_proj.elapsed().as_secs_f64());

        let t_scan = Instant::now();
        let compliance = compliance_fraction(&layers, hidden, args.threshold) as f32;
        println!(
            "  compliance @ R<{}: {:.4}% (scan {:.1}s)",
            args.threshold,
            compliance * 100.0,
            t_scan.elapsed().as_secs_f64()
        );

        // Decide final precision for this projection.
        let (chosen_prec, action) = match policy_prec {
            Precision::Fp4 => {
                if compliance < args.compliance_floor {
                    // Downgrade per self-policing gate.
                    println!(
                        "  compliance {} < floor {} → downgrading to FP8",
                        compliance, args.compliance_floor
                    );
                    (Precision::Fp8, "downgraded_fp4_to_fp8")
                } else {
                    (Precision::Fp4, "wrote_fp4")
                }
            }
            Precision::Fp8 => (Precision::Fp8, "wrote_fp8_per_policy_default"),
            Precision::F16 => (Precision::F16, "wrote_f16_per_policy_default"),
            Precision::F32 => (Precision::F32, "wrote_f32_per_policy_default"),
        };

        // Emit the file.
        let out_file = match chosen_prec {
            Precision::Fp4 => format!("{}_fp4.bin", fs_prefix(name)),
            Precision::Fp8 => format!("{}_fp8.bin", fs_prefix(name)),
            Precision::F16 | Precision::F32 => src_file.to_string(),
        };
        let out_path = args.out_path.join(&out_file);
        let layer_refs: Vec<&[f32]> = layers.iter().map(|v| v.as_slice()).collect();

        let t_write = Instant::now();
        match chosen_prec {
            Precision::Fp4 => {
                larql_vindex::format::fp4_storage::write_fp4_projection(
                    &out_path,
                    hidden,
                    &layer_refs,
                )?;
            }
            Precision::Fp8 => {
                larql_vindex::format::fp4_storage::write_fp8_projection(
                    &out_path,
                    hidden,
                    &layer_refs,
                )?;
            }
            Precision::F16 | Precision::F32 => {
                // Just copy the source file — no quantisation change.
                link_or_copy(&src_path, &out_path)?;
            }
        }
        let out_size = std::fs::metadata(&out_path)?.len();
        println!(
            "  wrote {} ({:?}, {:.2} GB, {:.1}s)",
            out_path.display(),
            chosen_prec,
            out_size as f64 / 1_073_741_824.0,
            t_write.elapsed().as_secs_f64()
        );

        final_projections[idx] = Some(ProjectionFormat {
            precision: chosen_prec,
            file: out_file.clone(),
        });
        compliance_entries.push(json!({
            "projection": name,
            "compliance_at_threshold": compliance,
            "threshold": args.threshold,
            "policy_precision": format!("{:?}", policy_prec).to_lowercase(),
            "chosen_precision": format!("{:?}", chosen_prec).to_lowercase(),
            "action": action,
            "output_file": out_file,
            "output_size_bytes": out_size,
        }));
    }

    // ── Build new VindexConfig with fp4 manifest ─────────────────────────────
    let projections_cfg = Projections {
        gate: final_projections[0].take().unwrap(),
        up: final_projections[1].take().unwrap(),
        down: final_projections[2].take().unwrap(),
    };
    let fp4_cfg = Fp4Config {
        projections: projections_cfg,
        compliance_gate: ComplianceGate {
            threshold_ratio: args.threshold,
            min_compliant_fraction: args.compliance_floor,
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

    // Re-serialise with fp4 included.
    let out_index_json = serde_json::to_string_pretty(&src_config)?;
    std::fs::write(args.out_path.join("index.json"), out_index_json)?;

    // ── Write fp4_compliance.json sidecar ────────────────────────────────────
    let compliance_doc = json!({
        "extracted_at": chrono_now_fallback(),
        "scanner_version": env!("CARGO_PKG_VERSION"),
        "policy": format!("{:?}", args.policy),
        "block_elements_scanned": 256,
        "compliance_gate_threshold_ratio": args.threshold,
        "compliance_gate_min_fraction": args.compliance_floor,
        "per_projection": compliance_entries,
    });
    std::fs::write(
        args.out_path.join("fp4_compliance.json"),
        serde_json::to_string_pretty(&compliance_doc)?,
    )?;

    // ── Hard-link (or copy) all other files ──────────────────────────────────
    let handled: std::collections::HashSet<&str> = [
        "index.json",
        "gate_vectors.bin",
        "up_features.bin",
        "down_features.bin",
        "fp4_compliance.json",
    ]
    .iter()
    .copied()
    .collect();

    let mut linked = 0;
    let mut linked_bytes: u64 = 0;
    for entry in std::fs::read_dir(&args.in_path)? {
        let entry = entry?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if handled.contains(fname_str.as_ref()) {
            continue;
        }
        let meta = entry.metadata()?;
        if !meta.is_file() {
            continue;
        }
        let dst = args.out_path.join(&fname);
        link_or_copy(&entry.path(), &dst)?;
        linked += 1;
        linked_bytes += meta.len();
    }
    println!();
    println!(
        "linked/copied {linked} auxiliary files ({:.2} GB)",
        linked_bytes as f64 / 1_073_741_824.0
    );
    println!("total wall time: {:.1}s", t_total.elapsed().as_secs_f64());

    // ── Final summary ────────────────────────────────────────────────────────
    println!();
    println!("== summary ==");
    let src_ffn_bytes = src_config.layers.iter().map(|l| l.length * 3).sum::<u64>();
    let out_ffn_bytes: u64 = [
        src_config
            .fp4
            .as_ref()
            .unwrap()
            .projections
            .gate
            .file
            .clone(),
        src_config.fp4.as_ref().unwrap().projections.up.file.clone(),
        src_config
            .fp4
            .as_ref()
            .unwrap()
            .projections
            .down
            .file
            .clone(),
    ]
    .iter()
    .map(|f| {
        std::fs::metadata(args.out_path.join(f))
            .map(|m| m.len())
            .unwrap_or(0)
    })
    .sum();
    let ratio = src_ffn_bytes as f64 / out_ffn_bytes.max(1) as f64;
    println!(
        "  FFN storage src : {:.2} GB",
        src_ffn_bytes as f64 / 1_073_741_824.0
    );
    println!(
        "  FFN storage dst : {:.2} GB",
        out_ffn_bytes as f64 / 1_073_741_824.0
    );
    println!("  compression    : {ratio:.2}×");

    Ok(())
}

fn fs_prefix(proj_name: &str) -> &'static str {
    match proj_name {
        "gate" => "gate_vectors",
        "up" => "up_features",
        "down" => "down_features",
        _ => panic!("unknown projection {proj_name}"),
    }
}

/// ISO 8601 timestamp without bringing in chrono as a dep. Uses UNIX
/// epoch + a crude breakdown; good enough for log lines.
fn chrono_now_fallback() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("@epoch+{secs}s")
}
