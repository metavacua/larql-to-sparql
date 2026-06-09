use larql_vindex::format::filenames::*;
use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct ConvertArgs {
    #[command(subcommand)]
    command: ConvertCommand,
}

#[derive(Subcommand)]
enum ConvertCommand {
    /// Convert a GGUF model to a vindex.
    GgufToVindex {
        /// Path to the .gguf file.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16 (half precision). Default if `--quant` not set.
        #[arg(long)]
        f16: bool,

        /// Storage quant format for inference weights. `f16` (default) emits
        /// the dequantised half-precision weight files (`attn_weights.bin`,
        /// `up_weights.bin`, `down_weights.bin`, `lm_head.bin`). `q4k` adds
        /// the Q4_K_M fast-decode artefacts (`interleaved_q4k.bin`,
        /// `attn_weights_q4k.bin`, `lm_head_q4.bin`) and strips the
        /// redundant f16 weight files so the production decode path reads
        /// the Q4_K layer directly. Requires `--level inference` or `--level
        /// all` so the writer has weights to quantise. Internally still
        /// roundtrips through f32 — true bitwise passthrough of GGUF Q4_K
        /// blocks is a follow-up perf win.
        #[arg(long, default_value = "f16", value_parser = ["f16", "q4k"])]
        quant: String,
    },

    /// Convert a DeepSeek-V4-Flash GGUF to a (server-conforming) DSv4
    /// vindex via the dedicated extraction path (`build_dsv4_vindex`).
    /// Distinct from `gguf-to-vindex` — DSv4's low-rank/latent/grouped
    /// attention can't go through the generic Q/K/V/O writer. Emits the
    /// per-blob weight files + `index.json` (VindexConfig) + `embeddings.bin`,
    /// and copies `--tokenizer` to `tokenizer.json` so larql-server can
    /// load + serve it.
    GgufToDsv4Vindex {
        /// Path to the DeepSeek-V4-Flash `.gguf` file.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Source HuggingFace `tokenizer.json` to copy into the vindex
        /// (the GGUF stores tokenizer data as metadata KVs, not an HF
        /// file, and there's no in-repo converter). Without it the vindex
        /// has no `tokenizer.json` and the server can't tokenise.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },

    /// Convert a safetensors model to a vindex (alias for extract-index).
    SafetensorsToVindex {
        /// Path to the model directory.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16.
        #[arg(long)]
        f16: bool,
    },

    /// Show GGUF file metadata and tensor info.
    GgufInfo {
        /// Path to the .gguf file.
        input: PathBuf,
    },

    /// Quantize an existing vindex into a different storage format.
    /// Each sub-format has its own flag surface — see
    /// `docs/specs/quantize-cli-spec.md` for the shape and how new
    /// formats slot in. FP4 is the only format wired as of exp 26;
    /// Q4K and future formats land as additional subcommands.
    #[command(subcommand)]
    Quantize(QuantizeCommand),

    /// Retrofit `down_features_q4k.bin` (W2 feature-major down) into
    /// an existing Q4K vindex without re-quantising. Reads the down
    /// portion of `interleaved_kquant.bin` per layer, transposes to
    /// `[intermediate, hidden]`, re-quantises at the same precision
    /// the source used, and writes the W2 file + manifest in place.
    /// Idempotent — silent no-op when the file is already present.
    /// See ADR-009 for the architectural rationale.
    AddFeatureMajorDown {
        /// Vindex directory to retrofit. Must already have
        /// `interleaved_kquant.bin` + manifest (i.e. `quant: q4k` in
        /// `index.json`).
        #[arg(long)]
        input: PathBuf,

        /// Suppress the per-layer progress line printed during write.
        #[arg(long)]
        quiet: bool,
    },
}

#[derive(Subcommand)]
enum QuantizeCommand {
    /// Convert an f32/f16 vindex into a Q4_K/Q6_K vindex (the Ollama-
    /// compatible "Q4_K_M" mix: attention Q/K/O + FFN gate/up at
    /// Q4_K, attention V + FFN down at Q6_K). `--down-q4k` switches
    /// FFN down to Q4_K uniformly — saves ~30 MB/layer on 31B at
    /// modest precision cost.
    ///
    /// Source must be extracted with `--level inference` or `--level all`
    /// (needs the full f32/f16 weights to quantise).
    Q4K {
        /// Existing vindex directory (the source).
        #[arg(long)]
        input: PathBuf,

        /// Output vindex directory. Written atomically (to `<out>.tmp/`
        /// then renamed on success).
        #[arg(long)]
        output: PathBuf,

        /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default off
        /// preserves the Ollama Q4_K_M mix (Q4_K gate/up + Q6_K down).
        #[arg(long)]
        down_q4k: bool,

        /// Emit `down_features_q4k.bin` (W2 feature-major down) so per-feature
        /// row decode can skip the `kquant_ffn_layer` cache. Adds ~14 MB / layer
        /// at Gemma 4B dims; eliminates the ~840 MB heap cache ceiling.
        /// Recommended for CPU sparse walk and grid/MoE workloads.
        #[arg(long)]
        feature_major_down: bool,

        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,

        /// Suppress the backend-describe summary printed after write.
        #[arg(long)]
        quiet: bool,
    },

    /// Convert an f32/f16 vindex into an FP4/FP8 vindex per the
    /// chosen policy. Exp 26. Policy spec: `docs/specs/fp4-precision-policy.md`.
    Fp4 {
        /// Existing vindex directory (the source).
        #[arg(long)]
        input: PathBuf,

        /// Output vindex directory. Written atomically (to `<out>.tmp/`
        /// then renamed on success).
        #[arg(long)]
        output: PathBuf,

        /// Precision policy for up / down (gate stays at source dtype
        /// in all three policies — FP4 gate is blocked on an FP4-aware
        /// gate KNN path, see policy spec §2).
        #[arg(long, default_value = "option-b", value_parser = ["option-a", "option-b", "option-c"])]
        policy: String,

        /// Min compliance fraction for an FP4-targeted projection at
        /// the given threshold. Projections below this are downgraded
        /// to the manifest's fallback precision (FP8). Doesn't apply
        /// to FP8 / F16 projections — those don't use the
        /// distributional assumption.
        #[arg(long, default_value_t = 0.99)]
        compliance_floor: f32,

        /// max(sub-block scale)/min(sub-block scale) threshold for
        /// the FP4 compliance gate. 16.0 is the E4M3/E2M1 exponent
        /// budget (the format's derived default); lower = stricter,
        /// higher = more permissive.
        #[arg(long, default_value_t = 16.0)]
        threshold: f32,

        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,

        /// Fail (non-zero exit) if any FP4-targeted projection misses
        /// the compliance floor, instead of downgrading it.
        #[arg(long)]
        strict: bool,

        /// Skip emitting `fp4_compliance.json` in the output directory.
        #[arg(long)]
        no_sidecar: bool,

        /// Suppress the backend-describe summary printed after write.
        #[arg(long)]
        quiet: bool,
    },
}

pub fn run(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ConvertCommand::GgufToVindex {
            input,
            output,
            level,
            f16,
            quant,
        } => run_gguf_to_vindex(&input, &output, &level, f16, &quant),
        ConvertCommand::GgufToDsv4Vindex {
            input,
            output,
            tokenizer,
        } => run_gguf_to_dsv4_vindex(&input, &output, tokenizer.as_deref()),
        ConvertCommand::SafetensorsToVindex {
            input,
            output,
            level,
            f16,
        } => run_safetensors_to_vindex(&input, &output, &level, f16),
        ConvertCommand::GgufInfo { input } => run_gguf_info(&input),
        ConvertCommand::Quantize(cmd) => run_quantize(cmd),
        ConvertCommand::AddFeatureMajorDown { input, quiet } => {
            run_add_feature_major_down(&input, quiet)
        }
    }
}

fn run_gguf_to_dsv4_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    tokenizer: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::attention::dsv4_storage_build::DsV4Hyperparams;
    use larql_inference::attention::dsv4_vindex_build::build_dsv4_vindex;

    eprintln!("Loading GGUF: {}", input.display());
    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    if let Some(arch) = gguf
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
    {
        eprintln!("  Architecture: {arch}");
        if arch != "deepseek4" && arch != "deepseek_v4" {
            return Err(format!(
                "expected a DeepSeek-V4 GGUF (general.architecture deepseek4), got {arch:?}"
            )
            .into());
        }
    }
    let model_id = gguf
        .metadata
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("deepseek-v4-flash")
        .to_string();

    let hp = DsV4Hyperparams::from_gguf(&gguf)
        .map_err(|e| format!("DsV4Hyperparams::from_gguf: {e}"))?;

    // Derive the layer count from the tensor names (`blk.<N>.*`) — robust
    // against metadata-key drift.
    let n_layer = gguf
        .tensor_infos
        .iter()
        .filter_map(|i| {
            i.name()
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|d| d.parse::<usize>().ok())
        })
        .max()
        .map(|m| m + 1)
        .ok_or("no blk.N.* tensors found — not a per-layer GGUF")?;

    if tokenizer.is_none() {
        eprintln!(
            "  WARNING: no --tokenizer given; the vindex will have no \
             tokenizer.json and larql-server won't be able to serve it."
        );
    }
    eprintln!(
        "  Building DSv4 vindex: {n_layer} layers → {}",
        output.display()
    );
    let manifest = build_dsv4_vindex(&gguf, &hp, n_layer, &model_id, output, tokenizer)
        .map_err(|e| format!("build_dsv4_vindex: {e}"))?;
    eprintln!(
        "  Done: {} layers, model_id={:?}, compress_ratios={:?}",
        manifest.n_layer, manifest.model_id, manifest.compress_ratios
    );
    Ok(())
}

fn run_add_feature_major_down(
    input: &std::path::Path,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::add_feature_major_down;

    if !quiet {
        eprintln!("Retrofitting feature-major down → {}", input.display());
    }
    let report = add_feature_major_down(input)?;
    if report.skipped {
        if !quiet {
            eprintln!(
                "  down_features_q4k.bin already present — no-op (skipped {} layers)",
                report.num_layers,
            );
        }
        return Ok(());
    }
    if !quiet {
        let mb = report.bytes_written as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  wrote down_features_q4k.bin: {} layers, {:.1} MB, {:.2?}",
            report.num_layers, mb, report.wall_time,
        );
        eprintln!(
            "  per-feature down decode now skips kquant_ffn_layer cache \
             (verify via GET /v1/stats → q4k_ffn.feature_major_down: true)"
        );
    }
    Ok(())
}

fn run_quantize(cmd: QuantizeCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        QuantizeCommand::Fp4 {
            input,
            output,
            policy,
            compliance_floor,
            threshold,
            force,
            strict,
            no_sidecar,
            quiet,
        } => run_quantize_fp4(QuantizeFp4Opts {
            input,
            output,
            policy,
            compliance_floor,
            threshold,
            force,
            strict,
            no_sidecar,
            quiet,
        }),
        QuantizeCommand::Q4K {
            input,
            output,
            down_q4k,
            feature_major_down,
            force,
            quiet,
        } => run_quantize_q4k(QuantizeQ4kOpts {
            input,
            output,
            down_q4k,
            feature_major_down,
            force,
            quiet,
        }),
    }
}

struct QuantizeQ4kOpts {
    input: PathBuf,
    output: PathBuf,
    down_q4k: bool,
    feature_major_down: bool,
    force: bool,
    quiet: bool,
}

fn run_quantize_q4k(opts: QuantizeQ4kOpts) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::{vindex_to_q4k, Q4kConvertConfig};

    let config = Q4kConvertConfig {
        down_q4k: opts.down_q4k,
        feature_major_down: opts.feature_major_down,
        force: opts.force,
    };

    if !opts.quiet {
        eprintln!("== quantize q4k ==");
        eprintln!("  in       : {}", opts.input.display());
        eprintln!("  out      : {}", opts.output.display());
        eprintln!(
            "  down_q4k : {} ({})",
            opts.down_q4k,
            if opts.down_q4k {
                "Q4_K down (uniform)"
            } else {
                "Q6_K down (Q4_K_M mix)"
            }
        );
        eprintln!();
    }

    let report = vindex_to_q4k(&opts.input, &opts.output, &config)?;

    if !opts.quiet {
        eprintln!("── summary ──");
        eprintln!(
            "  FFN storage : {:.2} GB → {:.2} GB  ({:.2}× compression)",
            report.src_ffn_bytes as f64 / 1_073_741_824.0,
            report.dst_ffn_bytes as f64 / 1_073_741_824.0,
            report.compression,
        );
        eprintln!(
            "  Linked aux  : {} files ({:.2} GB)",
            report.aux_linked_count,
            report.aux_linked_bytes as f64 / 1_073_741_824.0
        );
        eprintln!("  Wall time   : {:.1}s", report.wall_time.as_secs_f64());
        eprintln!("  Walk backend: {}", report.walk_backend);
        eprintln!();
        eprintln!("→ {}", opts.output.display());
    }

    Ok(())
}

struct QuantizeFp4Opts {
    input: PathBuf,
    output: PathBuf,
    policy: String,
    compliance_floor: f32,
    threshold: f32,
    force: bool,
    strict: bool,
    no_sidecar: bool,
    quiet: bool,
}

fn run_quantize_fp4(opts: QuantizeFp4Opts) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::{vindex_to_fp4, Fp4ConvertConfig, Policy, ProjectionOutcome};

    let policy = Policy::parse(&opts.policy)?;
    let config = Fp4ConvertConfig {
        policy,
        compliance_floor: opts.compliance_floor,
        threshold: opts.threshold,
        strict: opts.strict,
        force: opts.force,
        emit_sidecar: !opts.no_sidecar,
    };

    if !opts.quiet {
        eprintln!("== quantize fp4 ==");
        eprintln!("  in     : {}", opts.input.display());
        eprintln!("  out    : {}", opts.output.display());
        eprintln!("  policy : {}", policy.label());
        eprintln!(
            "  floor  : {:.1}% @ R<{}",
            opts.compliance_floor * 100.0,
            opts.threshold
        );
        eprintln!();
    }

    let (report, _scan) = vindex_to_fp4(&opts.input, &opts.output, &config)?;

    if !opts.quiet {
        eprintln!("── per-projection ──");
        for p in &report.per_projection {
            let compliance = p
                .compliance_at_threshold
                .map(|c| format!("{:.4}%", c * 100.0))
                .unwrap_or_else(|| "N/A".into());
            let downgrade_flag = matches!(
                p.outcome,
                ProjectionOutcome::DowngradedFp4ToFp8 | ProjectionOutcome::DowngradedFp4ToF16,
            );
            let marker = if downgrade_flag { "⚠" } else { " " };
            eprintln!(
                "  {marker} {:<5}  compliance={:<12}  → {:?}  ({})",
                p.name,
                compliance,
                p.chosen_precision,
                p.outcome.action_str(),
            );
        }
        eprintln!();
        eprintln!("── summary ──");
        eprintln!(
            "  FFN storage : {:.2} GB → {:.2} GB  ({:.2}× compression)",
            report.src_ffn_bytes as f64 / 1_073_741_824.0,
            report.dst_ffn_bytes as f64 / 1_073_741_824.0,
            report.compression,
        );
        eprintln!(
            "  Linked aux  : {} files ({:.2} GB)",
            report.aux_linked_count,
            report.aux_linked_bytes as f64 / 1_073_741_824.0
        );
        eprintln!("  Wall time   : {:.1}s", report.wall_time.as_secs_f64());
        eprintln!("  Walk backend: {}", report.walk_backend);
        eprintln!();
        if report.per_projection.iter().any(|p| {
            matches!(
                p.outcome,
                ProjectionOutcome::DowngradedFp4ToFp8 | ProjectionOutcome::DowngradedFp4ToF16
            )
        }) {
            eprintln!("⚠ compliance floor missed on ≥ 1 projection; see fp4_compliance.json.");
            if !opts.strict {
                eprintln!("(Use --strict to treat this as a fatal error.)");
            }
        }
        eprintln!("→ {}", opts.output.display());
    }

    Ok(())
}

fn run_gguf_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
    quant: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let q4k = matches!(quant, "q4k");
    if q4k && level == "browse" {
        return Err(
            "--quant q4k requires --level inference or --level all (the Q4_K writer needs weights)"
                .into(),
        );
    }
    eprintln!("Loading GGUF: {}", input.display());

    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    // Show metadata summary
    if let Some(name) = gguf.metadata.get("general.name") {
        eprintln!("  Model: {:?}", name);
    }
    if let Some(arch) = gguf.metadata.get("general.architecture") {
        eprintln!("  Architecture: {:?}", arch);
    }

    eprintln!("  Loading and dequantizing tensors...");

    // For MoE models, pre-scan the GGUF tensor list for 3-D packed
    // expert tensors (`*ffn_{gate,up,down}_exps.weight`) and route them
    // through the lazy loader — `load_gguf_lazy_tensors` populates
    // `weights.quant_tensors` with per-expert HF-style aliases (post
    // #120) over the same Arc mmap, zero copies. The dense
    // `load_gguf` would skip 3-D tensors entirely (its
    // `match info.n_dims` only handles 2-D / 1-D), leaving MoE expert
    // weight files as 0 bytes — the silent-broken-vindex footgun that
    // PR #119 guarded against with a fail-fast.
    //
    // `build_vindex`'s MoE branch now falls back to `quant_tensors`
    // with on-demand dequant (see `lookup_expert_weight` in
    // extract/build.rs), so the lazy aliases get consumed correctly
    // when writing `gate_vectors.bin` / down meta.
    let gguf_file = larql_models::loading::gguf::GgufFile::open(input)?;

    // Lazy keys: 3-D MoE expert tensors (per-expert HF aliases, post #120)
    // PLUS 2-D **quantised** tensors so their raw GGUF bytes are preserved
    // in `quant_tensors` for the Q4_K writer's bit-passthrough fast paths
    // (PRs #194/#195/#196: re-quantizing imatrix-aware GGUFs through f32
    // diverges by ~0.1% per element; Unsloth Q4_K_M ships attn / lm_head /
    // shared experts as Q8_0; Qwen3-Coder-Next ships attn_qkv / ssm_out as
    // Q5_K — they must round-trip in their source format, NOT get
    // downquantized to Q4_K).
    use larql_models::quant::ggml::{TYPE_MXFP4, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0};
    let moe_lazy_keys: std::collections::HashSet<String> = gguf_file
        .tensor_infos
        .iter()
        .filter(|info| {
            // Exclude the input embed and output (lm_head). They have
            // special-cased extract paths (`write_embeddings`,
            // `write_lm_head_q4k`) that assume `weights.embed` /
            // `weights.lm_head` are populated as dense f32 — the lazy
            // loader instead steers them into `embed_quant` /
            // `lm_head_quant` and zeroes the dense field, which makes
            // the existing writers emit 0-byte files. The passthrough
            // optimisation only matters for the per-layer matmuls
            // (attn / deltanet) anyway; vocab-sized embed/lm_head
            // matmuls are 1-shot per forward, not the per-layer
            // compounding drift source.
            let name = info.name();
            if name == "token_embd.weight" || name == "output.weight" || name == "lm_head.weight" {
                return false;
            }
            (info.dims().len() == 3
                && (name.ends_with("ffn_gate_exps.weight")
                    || name.ends_with("ffn_up_exps.weight")
                    || name.ends_with("ffn_down_exps.weight")))
                || (info.dims().len() == 2
                    && matches!(
                        info.tensor_type(),
                        TYPE_Q4_K | TYPE_Q5_K | TYPE_Q6_K | TYPE_Q8_0 | TYPE_MXFP4
                    ))
        })
        .map(|info| larql_models::loading::gguf::normalize_gguf_key(info.name()))
        .collect();

    let weights = if !moe_lazy_keys.is_empty() {
        eprintln!(
            "  Lazy loader: {} tensors (3-D MoE experts + 2-D Q4_K/Q5_K/Q6_K/Q8_0/MXFP4 matmuls)",
            moe_lazy_keys.len()
        );
        larql_models::loading::gguf::load_gguf_lazy_tensors(input, &moe_lazy_keys)?
    } else {
        larql_models::load_gguf(input)?
    };

    eprintln!(
        "  {} layers, hidden_size={}, intermediate_size={}, vocab_size={}",
        weights.num_layers, weights.hidden_size, weights.intermediate_size, weights.vocab_size
    );

    // Hybrid SSM/DeltaNet arches (Qwen 3.6 dense + MoE) now flow through
    // `write_q4k::deltanet` (matmul tensors), `write_q4k::norms`
    // (ssm_norm/dt/a/conv1d), and `write_q4k::moe_layers` (PerExpert MoE
    // layout). The earlier fast-fail guard at this point was removed in
    // change `vindex-qwen35moe-extraction`.

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = gguf
        .metadata
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf-model")
        .to_string();

    // Find tokenizer — check same directory as GGUF file
    let tokenizer = input.parent().and_then(|dir| {
        let tok_path = dir.join(TOKENIZER_JSON);
        if tok_path.exists() {
            larql_vindex::tokenizers::Tokenizer::from_file(&tok_path).ok()
        } else {
            None
        }
    });

    let tokenizer_ref = tokenizer
        .as_ref()
        .ok_or("tokenizer.json not found next to GGUF file. Place it in the same directory.")?;

    eprintln!("\nExtracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    larql_vindex::build_vindex(
        &weights,
        tokenizer_ref,
        &model_name,
        output,
        10,
        extract_level,
        dtype,
        &mut callbacks,
    )?;

    // --quant q4k: overlay the Q4_K fast-decode artefacts and drop the
    // now-redundant f16 weight files. The Q4_K writer reads from the
    // same in-memory `ModelWeights` (no separate vindex_to_q4k step)
    // and patches `index.json` to `quant=Q4K`. Production decode then
    // reads `interleaved_q4k.bin` / `attn_weights_q4k.bin` / `lm_head_q4.bin`
    // instead of the f16 forms. Internally this still roundtrips
    // GGUF Q4_K → f32 → Q4_K; true bitwise passthrough is a follow-up.
    if q4k {
        eprintln!("  Writing Q4_K weight artefacts...");
        let q4k_opts = larql_vindex::Q4kWriteOptions::default();
        larql_vindex::write_model_weights_q4k_with_opts(
            &weights,
            output,
            &mut callbacks,
            q4k_opts,
        )?;
        // The f16 weight files written by build_vindex are now
        // strictly redundant — the Q4_K writer's manifest supersedes
        // them and the decode path no longer reads the f16 names when
        // `index.json::quant == q4k`. Delete the leftovers so the
        // vindex on disk reflects only the Q4_K layout.
        for name in [
            ATTN_WEIGHTS_BIN,
            UP_WEIGHTS_BIN,
            DOWN_WEIGHTS_BIN,
            LM_HEAD_BIN,
        ] {
            let p = output.join(name);
            if p.exists() {
                if let Err(e) = std::fs::remove_file(&p) {
                    eprintln!(
                        "  warning: could not remove redundant f16 file {}: {e}",
                        p.display()
                    );
                }
            }
        }
        eprintln!("  Q4_K artefacts written.");
    }

    // GGUF conversion: HF metadata (tokenizer_config.json etc.) is not
    // packed in the GGUF itself, but if the user kept the HF files next
    // to the `.gguf`, snapshot them. Missing-file case is a no-op.
    if let Some(src_dir) = input.parent() {
        if let Err(e) = larql_vindex::snapshot_hf_metadata(src_dir, output) {
            eprintln!("  warning: failed to snapshot HF metadata: {e}");
        }
    }

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_safetensors_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // This is essentially extract-index
    eprintln!("Loading safetensors: {}", input.display());
    let weights = larql_models::load_model_dir(input)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(input).or_else(|_| {
        // Try to load from the model directory
        let tok_path = input.join(TOKENIZER_JSON);
        larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| larql_vindex::VindexError::Parse(e.to_string()))
    })?;

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = input
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());

    eprintln!("Extracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        &model_name,
        output,
        10,
        extract_level,
        dtype,
        &mut callbacks,
    )?;
    // Snapshot HF-side metadata (chat template, special tokens, generation
    // config) from the source directory. `input` here is the safetensors
    // model dir, which is where these files live in the HF cache.
    if let Err(e) = larql_vindex::snapshot_hf_metadata(input, output) {
        eprintln!("  warning: failed to snapshot HF metadata: {e}");
    }

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_gguf_info(input: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    println!("GGUF: {}", input.display());
    println!();

    // Print metadata
    println!("Metadata ({} keys):", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in &keys {
        let val = &gguf.metadata[*key];
        match val {
            larql_models::loading::gguf::GgufValue::String(s) => {
                if s.len() > 80 {
                    println!("  {}: \"{}...\"", key, &s[..80]);
                } else {
                    println!("  {}: \"{}\"", key, s);
                }
            }
            larql_models::loading::gguf::GgufValue::Array(arr) => {
                println!("  {}: [{} elements]", key, arr.len());
            }
            other => println!("  {}: {:?}", key, other),
        }
    }

    println!();

    // Print tensor info table (name, dims, ggml type id) — the layout spec a
    // consumer (e.g. a vindex→GGUF exporter) must match. Sorted by name.
    println!();
    println!("Tensors ({}):", gguf.tensor_infos.len());
    let mut infos: Vec<&larql_models::loading::gguf::GgufTensorInfo> =
        gguf.tensor_infos.iter().collect();
    infos.sort_by(|a, b| a.name().cmp(b.name()));
    for t in &infos {
        println!(
            "  {:<40} dims={:?} type={}",
            t.name(),
            t.dims(),
            t.tensor_type(),
        );
    }

    println!();

    // Print synthesised config
    let config = gguf.to_config_json();
    println!("Detected config:");
    println!("  {}", serde_json::to_string_pretty(&config)?);

    Ok(())
}

struct SilentCallbacks;
impl larql_vindex::IndexBuildCallbacks for SilentCallbacks {}
