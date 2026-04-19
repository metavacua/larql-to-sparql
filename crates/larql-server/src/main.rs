//! larql-server — HTTP server for vindex knowledge queries.

mod announce;
mod auth;
mod cache;
mod embed_store;
mod error;
mod etag;
mod ffn_l2_cache;
mod grpc;
mod ratelimit;
mod routes;
mod session;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use axum::middleware;
use clap::Parser;
use tokio::sync::RwLock;
use tracing::{info, warn};

use larql_vindex::{
    PatchedVindex, SilentLoadCallbacks, VectorIndex,
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};

use cache::DescribeCache;
use session::SessionManager;
use state::{AppState, LoadedModel, model_id_from_name, load_probe_labels};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser)]
#[command(
    name = "larql-server",
    version,
    about = "HTTP server for vindex knowledge queries and inference"
)]
struct Cli {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Run as an FFN-service endpoint for remote `RemoteWalkBackend`
    /// clients. Disables `/v1/infer` (like `--no-infer`) and advertises
    /// `mode: ffn-service` in `/v1/stats`. This is Act 2 of the demo —
    /// the server holds the FFN weights, clients hold attention.
    ///
    /// Also skips the f16→f32 gate-vector warmup, which is the largest
    /// eager cost on startup (~2x the gate_vectors.bin size). Gate
    /// decode happens lazily per layer on first request instead.
    #[arg(long)]
    ffn_only: bool,

    /// Run as an embed-service endpoint.
    ///
    /// Loads only embeddings.bin, lm_head, and the tokenizer — skips all
    /// FFN and attention weights. Advertises `mode: embed-service` in
    /// `/v1/stats`. Enables `/v1/embed`, `/v1/logits`, and `/v1/token/*`.
    ///
    /// Use this to offload the static embedding + lm_head lookup from
    /// attention-only clients (ADR-0007). The embed slice is ~2-5% of the
    /// full model weight — a minimal VPS can host it independently.
    #[arg(long)]
    embed_only: bool,

    /// Only load and serve layers in this range (inclusive, e.g. "0-19").
    /// Layers outside the range are not dequantized and their mmap pages are
    /// never touched, keeping RSS proportional to the shard size.
    /// Requests for out-of-range layers are rejected with HTTP 400.
    #[arg(long)]
    layers: Option<String>,

    /// Cap the number of decoded f16 gate layers held in the lazy cache.
    /// 0 = unlimited (default; matches historical behaviour). Each decoded
    /// layer is roughly `intermediate × hidden × 4 bytes` — on 31B that's
    /// ~433 MB per layer, so a 60-layer model fully decoded is ~26 GB.
    /// Set to N to cap at N layers via LRU eviction.
    ///
    /// Use when RSS headroom matters (e.g. co-hosting multiple models) at
    /// the cost of re-decode when evicted layers are re-accessed.
    #[arg(long, default_value = "0")]
    max_gate_cache_layers: usize,

    /// Ask the kernel to drop resident mmap pages after each walk-ffn
    /// request (calls `madvise(MADV_DONTNEED)` on every mapping). On
    /// Linux RSS drops immediately; on Darwin the kernel may defer.
    /// Pairs with `--max-gate-cache-layers` to enforce a hard bound.
    ///
    /// Prefer `--layers START-END` for real deployments — sharding
    /// prevents out-of-range pages from ever being touched. This flag
    /// is for the single-shard-holds-everything demo topology.
    #[arg(long)]
    release_mmap_after_request: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// API key for authentication (clients send Authorization: Bearer <key>).
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

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// gRPC port (enables gRPC server alongside HTTP).
    #[arg(long)]
    grpc_port: Option<u16>,

    /// TLS certificate path for HTTPS.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key path for HTTPS.
    #[arg(long)]
    tls_key: Option<PathBuf>,

    /// Join one or more router grids (comma-separated gRPC addresses).
    /// Example: "http://router-a:50052,http://router-b:50052"
    /// Each router gets an independent announce stream — stateless fan-out.
    /// Requires --public-url so routers know where to send clients.
    #[arg(long)]
    join: Option<String>,

    /// Public HTTP URL clients should use to reach this server.
    /// Used when announcing to the grid with --join.
    /// Example: "http://server-a:8080"
    #[arg(long)]
    public_url: Option<String>,

    /// Shared secret matching the router's --grid-key.
    /// Required when the router enforces grid authentication.
    #[arg(long, env = "LARQL_GRID_KEY")]
    grid_key: Option<String>,
}

fn parse_layer_range(s: &str) -> Result<(usize, usize), BoxError> {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() != 2 {
        return Err(format!("--layers: expected 'START-END' (e.g. '0-19'), got '{s}'").into());
    }
    let start: usize = parts[0].trim().parse()
        .map_err(|_| format!("--layers: invalid start '{}'", parts[0]))?;
    let end: usize = parts[1].trim().parse()
        .map_err(|_| format!("--layers: invalid end '{}'", parts[1]))?;
    if end < start {
        return Err(format!("--layers: end ({end}) must be >= start ({start})").into());
    }
    // CLI uses inclusive end; internally we use exclusive end.
    Ok((start, end + 1))
}

fn load_single_vindex(
    path_str: &str,
    no_infer: bool,
    ffn_only: bool,
    embed_only: bool,
    layer_range: Option<(usize, usize)>,
    max_gate_cache_layers: usize,
    release_mmap_after_request: bool,
) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex_with_range(&path, &mut cb, layer_range)?;
    if max_gate_cache_layers > 0 {
        index.set_gate_cache_max_layers(max_gate_cache_layers);
        info!("  Gate cache: LRU, max {} layers", max_gate_cache_layers);
    }
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    let has_weights = config.has_model_weights
        || config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All;

    if let Some((start, end)) = layer_range {
        info!("  Layers: {start}–{} (of {})", end - 1, config.num_layers);
    }
    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    // Load mmap'd feature-major vectors for walk FFN optimization.
    // Skip for embed_only — we never touch FFN paths.
    if !embed_only {
        match index.load_down_features(&path) {
            Ok(()) => info!("  Down features: loaded (mmap walk enabled)"),
            Err(_) => info!("  Down features: not available"),
        }
        if let Ok(()) = index.load_up_features(&path) { info!("  Up features: loaded (full mmap FFN)") }
    }

    // Warmup eagerly dequantises f16 gate vectors to f32 (~2x blowup). On a
    // 31B vindex that's ~13 GB f16 → ~26 GB f32 resident before the first
    // request. Skip it under `--ffn-only` / `--embed-only`.
    if ffn_only || embed_only {
        let reason = if embed_only { "--embed-only" } else { "--ffn-only" };
        info!("  Warmup: skipped ({reason})");
    } else {
        index.warmup();
        info!("  Warmup: done");
    }

    let (embeddings, embed_scale) = load_vindex_embeddings(&path)?;
    info!("  Embeddings: {}x{}", embeddings.shape()[0], embeddings.shape()[1]);

    // In --embed-only mode, attempt an f16-at-rest store to halve RSS.
    // Falls back silently if embeddings.bin is f32 (older vindexes).
    let embed_store = if embed_only {
        match crate::embed_store::EmbedStoreF16::open(
            &path,
            embed_scale,
            config.vocab_size,
            config.hidden_size,
            5_000,
        ) {
            Ok(store) => {
                let f16_bytes = config.vocab_size * config.hidden_size * 2;
                info!(
                    "  Embed store: f16 mmap ({:.1} GB, L1 cap 5000 tokens)",
                    f16_bytes as f64 / 1e9
                );
                Some(std::sync::Arc::new(store))
            }
            Err(e) => {
                info!("  Embed store: f16 mmap unavailable ({e}), using f32 heap");
                None
            }
        }
    } else {
        None
    };

    let tokenizer = load_vindex_tokenizer(&path)?;
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    // --ffn-only and --embed-only both disable /v1/infer.
    let infer_disabled = no_infer || ffn_only || embed_only;
    if embed_only {
        info!("  Mode: embed-service (--embed-only)");
        info!("  Infer: disabled (embed-service mode)");
    } else if ffn_only {
        info!("  Mode: ffn-service (--ffn-only)");
        info!("  Infer: disabled (FFN-service mode)");
    } else if no_infer {
        info!("  Infer: disabled (--no-infer)");
    } else if has_weights {
        info!("  Infer: available (weights detected, will lazy-load on first request)");
    } else {
        info!("  Infer: not available (no model weights in vindex)");
    }

    if release_mmap_after_request {
        info!("  Mmap release: enabled (MADV_DONTNEED after each walk-ffn request)");
    }

    let num_layers = config.num_layers;
    Ok(LoadedModel {
        id,
        path,
        config,
        patched: RwLock::new(patched),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled,
        ffn_only,
        embed_only,
        embed_store,
        release_mmap_after_request,
        weights: std::sync::OnceLock::new(),
        probe_labels,
        ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(num_layers),
    })
}

fn discover_vindexes(dir: &PathBuf) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("index.json").exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Accept both `larql-server <path>` and `larql-server serve <path>`.
    let args: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = if args.len() > 1 && args[1] == "serve" {
        std::iter::once(args[0].clone()).chain(args[2..].iter().cloned()).collect()
    } else {
        args
    };
    let cli = Cli::parse_from(filtered);

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();

    let layer_range = cli.layers.as_deref().map(parse_layer_range).transpose()?;

    if let Some(ref dir) = cli.dir {
        let paths = discover_vindexes(dir);
        if paths.is_empty() {
            return Err(format!("no .vindex directories found in {}", dir.display()).into());
        }
        info!("Found {} vindexes in {}", paths.len(), dir.display());
        for p in &paths {
            match load_single_vindex(&p.to_string_lossy(), cli.no_infer, cli.ffn_only, cli.embed_only, layer_range, cli.max_gate_cache_layers, cli.release_mmap_after_request) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path, cli.no_infer, cli.ffn_only, cli.embed_only, layer_range, cli.max_gate_cache_layers, cli.release_mmap_after_request)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    // Parse rate limiter if specified.
    let rate_limiter = cli.rate_limit.as_ref().and_then(|spec| {
        match ratelimit::RateLimiter::parse(spec) {
            Some(rl) => {
                info!("Rate limit: {}", spec);
                Some(Arc::new(rl))
            }
            None => {
                warn!("Invalid rate limit format: {} (expected e.g. '100/min')", spec);
                None
            }
        }
    });

    let state = Arc::new(AppState {
        models: models.clone(),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(cli.cache_ttl),
    });

    if cli.cache_ttl > 0 {
        info!("DESCRIBE cache: {}s TTL", cli.cache_ttl);
    }

    let is_multi = state.is_multi_model();
    let mut app = if is_multi {
        info!("Multi-model mode ({} models)", state.models.len());
        for m in &state.models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(Arc::clone(&state))
    } else {
        let m = &models[0];
        info!("Single-model mode: {}", m.config.model);
        routes::single_model_router(Arc::clone(&state))
    };

    // Rate limiting middleware.
    if let Some(ref rl) = rate_limiter {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(rl),
            ratelimit::rate_limit_middleware,
        ));
    }

    // Auth middleware (if --api-key set).
    if cli.api_key.is_some() {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::auth_middleware,
        ));
        info!("Auth: API key required");
    }

    // CORS middleware.
    if cli.cors {
        use tower_http::cors::CorsLayer;
        app = app.layer(CorsLayer::permissive());
        info!("CORS: enabled");
    }

    // Concurrency limit.
    app = app.layer(tower::limit::ConcurrencyLimitLayer::new(cli.max_concurrent));
    info!("Max concurrent: {}", cli.max_concurrent);

    // Trace middleware.
    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    // gRPC server (if --grpc-port set).
    if let Some(grpc_port) = cli.grpc_port {
        let grpc_addr = format!("{}:{}", cli.host, grpc_port).parse()?;
        let grpc_state = Arc::clone(&state);
        info!("gRPC: listening on {}", grpc_addr);
        tokio::spawn(async move {
            let svc = grpc::VindexGrpcService { state: grpc_state };
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(grpc::proto::vindex_service_server::VindexServiceServer::new(svc))
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {}", e);
            }
        });
    }

    let addr = format!("{}:{}", cli.host, cli.port);

    // Grid announce (if --join provided).
    if let Some(join_spec) = cli.join.clone() {
        let listen_url = cli.public_url.clone().unwrap_or_else(|| {
            let host = if cli.host == "0.0.0.0" { "127.0.0.1" } else { &cli.host };
            format!("http://{}:{}", host, cli.port)
        });
        let join_urls: Vec<String> = join_spec
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        if join_urls.len() > 1 {
            info!("Joining {} routers (stateless fan-out)", join_urls.len());
        }
        for m in &models {
            let (layer_start, layer_end) = match layer_range {
                Some((s, e)) => (s as u32, (e - 1) as u32),
                None => (0, (m.config.num_layers.saturating_sub(1)) as u32),
            };
            let vhash = announce::vindex_identity_hash(&m.id, m.config.num_layers);
            for join_url in &join_urls {
                announce::run_announce(announce::AnnounceConfig {
                    join_url: join_url.clone(),
                    model_id: m.id.clone(),
                    layer_start,
                    layer_end,
                    listen_url: listen_url.clone(),
                    ram_bytes: 0,
                    grid_key: cli.grid_key.clone(),
                    vindex_hash: vhash.clone(),
                });
            }
        }
    }

    // TLS or plain HTTP.
    if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        info!("TLS: enabled ({}, {})", cert_path.display(), key_path.display());
        info!("Listening: https://{}", addr);

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            cert_path, key_path,
        )
        .await?;

        axum_server::bind_rustls(addr.parse()?, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        info!("Listening: http://{}", addr);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
