use clap::{Parser, Subcommand};

mod commands;
mod formatting;
mod utils;

use commands::extraction::*;
use commands::query::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL knowledge graph extraction and querying"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ── Extraction ──
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// Build gate index for graph-based FFN (offline, run once per model).
    IndexGates(index_gates_cmd::IndexGatesArgs),

    /// Extract attention routing patterns from forward passes.
    ExtractRoutes(extract_routes_cmd::ExtractRoutesArgs),

    /// Walk the model as a local vector index — gate KNN + down token lookup.
    Walk(walk_cmd::WalkArgs),

    /// Capture and compare attention patterns across prompts.
    AttentionCapture(attention_capture_cmd::AttentionCaptureArgs),

    /// Extract attention template circuits from QK weight decomposition.
    QkTemplates(qk_templates_cmd::QkTemplatesArgs),

    /// SVD rank analysis of attention QK products — how many modes per head.
    QkRank(qk_rank_cmd::QkRankArgs),

    /// Extract interpretable modes from low-rank QK heads via SVD → gate projection.
    QkModes(qk_modes_cmd::QkModesArgs),

    /// Map attention OV circuits to FFN gate features (what each head activates).
    OvGate(ov_gate_cmd::OvGateArgs),

    /// Discover attention→FFN circuits from weight decomposition. No forward passes.
    CircuitDiscover(circuit_discover_cmd::CircuitDiscoverArgs),

    /// Bottleneck analysis of attention components.
    AttnBottleneck(attn_bottleneck_cmd::AttnBottleneckArgs),

    /// Benchmark FFN performance: dense vs sparse at various K values.
    FfnBench(ffn_bench_cmd::FfnBenchArgs),

    /// Bottleneck analysis of FFN components.
    FfnBottleneck(ffn_bottleneck_cmd::FfnBottleneckArgs),

    /// Measure overlap between entity-routed and ground-truth gate features.
    FfnOverlap(ffn_overlap_cmd::FfnOverlapArgs),

    /// Knowledge graph retrieval benchmark — zero matmul entity lookup.
    KgBench(kg_bench_cmd::KgBenchArgs),

    /// Measure FFN throughput: tokens/second at various access patterns.
    FfnThroughput(ffn_throughput_cmd::FfnThroughputArgs),

    /// Build a .vindex — the model decompiled to a standalone vector index.
    ExtractIndex(extract_index_cmd::ExtractIndexArgs),

    /// Build a custom model from a Vindexfile (declarative: FROM + PATCH + INSERT).
    Build(build_cmd::BuildArgs),

    /// Convert between model formats (GGUF → vindex, safetensors → vindex).
    Convert(convert_cmd::ConvertArgs),

    /// HuggingFace Hub: download or publish vindexes.
    Hf(hf_cmd::HfArgs),

    /// Verify vindex file integrity (SHA256 checksums).
    Verify(verify_cmd::VerifyArgs),

    // GraphWalk removed — used deprecated FeatureListFfn

    /// Trace residual stream trajectories on the sphere across layers.
    TrajectoryTrace(trajectory_trace_cmd::TrajectoryTraceArgs),

    // VindexBench removed — used deprecated DownClusteredFfn

    /// Test rank-k projection: replace L0→L_inject with a linear map, run the rest dense.
    ProjectionTest(projection_test_cmd::ProjectionTestArgs),

    /// Extract OV fingerprint basis from attention weights (zero forward passes).
    FingerprintExtract(fingerprint_extract_cmd::FingerprintExtractArgs),

    /// Test rule-based bottleneck: 9 if-else rules replace L0-13, run L14-33 dense.
    BottleneckTest(bottleneck_test_cmd::BottleneckTestArgs),

    /// Embedding jump: raw token embeddings → projected L13 → decoder. Zero layers for L0-13.
    EmbeddingJump(embedding_jump_cmd::EmbeddingJumpArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    // ── Query ──
    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),

    /// Filter graph edges by confidence, layer, selectivity, relation, source, etc.
    Filter(filter_cmd::FilterArgs),

    // ── LQL ──
    /// Launch the LQL interactive REPL.
    Repl,

    /// Execute an LQL statement.
    Lql(LqlArgs),

    // ── Server ──
    /// Serve a vindex over HTTP.
    Serve(ServeArgs),
}

#[derive(clap::Args)]
struct LqlArgs {
    /// LQL statement to execute (e.g., 'WALK "The capital of France is" TOP 5;')
    statement: String,
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<std::path::PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// API key for authentication.
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit per IP (e.g., "100/min", "10/sec").
    #[arg(long)]
    rate_limit: Option<String>,

    /// Max concurrent requests.
    #[arg(long, default_value = "100")]
    max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value = "0")]
    cache_ttl: u64,

    /// gRPC port.
    #[arg(long)]
    grpc_port: Option<u16>,

    /// TLS certificate path.
    #[arg(long)]
    tls_cert: Option<std::path::PathBuf>,

    /// TLS private key path.
    #[arg(long)]
    tls_key: Option<std::path::PathBuf>,

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        // Extraction
        Commands::WeightExtract(args) => weight_walk_cmd::run(args),
        Commands::AttentionExtract(args) => attention_walk_cmd::run(args),
        Commands::VectorExtract(args) => vector_extract_cmd::run(args),
        Commands::Residuals(args) => residuals_cmd::run(args),
        Commands::Predict(args) => predict_cmd::run(args),
        Commands::IndexGates(args) => index_gates_cmd::run(args),
        Commands::AttentionCapture(args) => attention_capture_cmd::run(args),
        Commands::QkTemplates(args) => qk_templates_cmd::run(args),
        Commands::QkRank(args) => qk_rank_cmd::run(args),
        Commands::QkModes(args) => qk_modes_cmd::run(args),
        Commands::OvGate(args) => ov_gate_cmd::run(args),
        Commands::CircuitDiscover(args) => circuit_discover_cmd::run(args),
        Commands::ExtractRoutes(args) => extract_routes_cmd::run(args),
        Commands::Walk(args) => walk_cmd::run(args),
        Commands::AttnBottleneck(args) => attn_bottleneck_cmd::run(args),
        Commands::FfnBench(args) => ffn_bench_cmd::run(args),
        Commands::FfnBottleneck(args) => ffn_bottleneck_cmd::run(args),
        Commands::FfnOverlap(args) => ffn_overlap_cmd::run(args),
        Commands::KgBench(args) => kg_bench_cmd::run(args),
        Commands::FfnThroughput(args) => ffn_throughput_cmd::run(args),
        Commands::ExtractIndex(args) => extract_index_cmd::run(args),
        Commands::Build(args) => build_cmd::run(args),
        Commands::Convert(args) => convert_cmd::run(args),
        Commands::Hf(args) => hf_cmd::run(args),
        Commands::Verify(args) => verify_cmd::run(args),
        // Commands::GraphWalk removed
        Commands::TrajectoryTrace(args) => trajectory_trace_cmd::run(args),
        // Commands::VindexBench removed
        Commands::ProjectionTest(args) => projection_test_cmd::run(args),
        Commands::FingerprintExtract(args) => fingerprint_extract_cmd::run(args),
        Commands::BottleneckTest(args) => bottleneck_test_cmd::run(args),
        Commands::EmbeddingJump(args) => embedding_jump_cmd::run(args),
        Commands::Bfs(args) => bfs_cmd::run(args),
        // Query
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
        Commands::Filter(args) => filter_cmd::run(args),
        // LQL
        Commands::Repl => {
            larql_lql::run_repl();
            Ok(())
        }
        Commands::Lql(args) => {
            match larql_lql::run_statement(&args.statement) {
                Ok(lines) => {
                    for line in &lines {
                        println!("{line}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
        Commands::Serve(args) => {
            // Build the argument list and exec larql-server.
            let mut cmd_args = Vec::new();
            if let Some(ref path) = args.vindex_path {
                cmd_args.push(path.clone());
            }
            if let Some(ref dir) = args.dir {
                cmd_args.push("--dir".into());
                cmd_args.push(dir.display().to_string());
            }
            cmd_args.push("--port".into());
            cmd_args.push(args.port.to_string());
            cmd_args.push("--host".into());
            cmd_args.push(args.host.clone());
            cmd_args.push("--log-level".into());
            cmd_args.push(args.log_level.clone());
            cmd_args.push("--max-concurrent".into());
            cmd_args.push(args.max_concurrent.to_string());
            if args.no_infer {
                cmd_args.push("--no-infer".into());
            }
            if args.cors {
                cmd_args.push("--cors".into());
            }
            if let Some(ref key) = args.api_key {
                cmd_args.push("--api-key".into());
                cmd_args.push(key.clone());
            }
            if let Some(ref rl) = args.rate_limit {
                cmd_args.push("--rate-limit".into());
                cmd_args.push(rl.clone());
            }
            if args.cache_ttl > 0 {
                cmd_args.push("--cache-ttl".into());
                cmd_args.push(args.cache_ttl.to_string());
            }
            if let Some(port) = args.grpc_port {
                cmd_args.push("--grpc-port".into());
                cmd_args.push(port.to_string());
            }
            if let Some(ref cert) = args.tls_cert {
                cmd_args.push("--tls-cert".into());
                cmd_args.push(cert.display().to_string());
            }
            if let Some(ref key) = args.tls_key {
                cmd_args.push("--tls-key".into());
                cmd_args.push(key.display().to_string());
            }

            // Try to find larql-server binary next to this binary.
            let exe = std::env::current_exe().ok();
            let server_bin = exe
                .as_ref()
                .and_then(|e| e.parent())
                .map(|d| d.join("larql-server"))
                .filter(|p| p.exists());

            let bin = server_bin
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "larql-server".into());

            let status = std::process::Command::new(&bin)
                .args(&cmd_args)
                .status();

            match status {
                Ok(s) if s.success() => Ok(()),
                Ok(s) => {
                    eprintln!("larql-server exited with {}", s);
                    std::process::exit(s.code().unwrap_or(1));
                }
                Err(e) => {
                    eprintln!("Failed to start larql-server: {e}");
                    eprintln!("Make sure larql-server is installed (cargo install --path crates/larql-server)");
                    std::process::exit(1);
                }
            }
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
