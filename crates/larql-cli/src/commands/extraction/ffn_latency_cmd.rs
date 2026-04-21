//! `larql dev ffn-latency` — measure HTTP round-trip overhead vs FFN compute
//! against a running `larql-server`.
//!
//! Reports:
//!   total_ms   — wall-clock (client stopwatch)
//!   server_ms  — FFN compute time from the binary response header
//!   overhead_ms — total − server (TCP + HTTP framing + serialization)
//!
//! Use this to decide whether a gRPC transport would meaningfully cut latency.
//! If overhead_ms is small relative to server_ms, gRPC won't help much.

use clap::Args;
use larql_inference::ffn::{RemoteFfnConfig, RemoteWalkBackend};

#[derive(Args)]
pub struct FfnLatencyArgs {
    /// URL of a running `larql-server` (e.g. `http://127.0.0.1:9183`).
    #[arg(long, default_value = "http://127.0.0.1:9183")]
    pub server: String,

    /// Number of calls to make. First call is warmup (excluded from stats).
    #[arg(long, short = 'n', default_value = "11")]
    pub samples: usize,

    /// Comma-separated layer indices to include in each batch request.
    /// Defaults to a single mid-stack layer.
    #[arg(long, default_value = "16")]
    pub layers: String,

    /// Per-request timeout in seconds.
    #[arg(long, default_value = "120")]
    pub timeout: u64,
}

pub fn run(args: FfnLatencyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let layers: Vec<usize> = args
        .layers
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;

    let config = RemoteFfnConfig::new(&args.server)
        .with_timeout(std::time::Duration::from_secs(args.timeout));

    println!("Connecting to {} …", args.server);
    let backend = RemoteWalkBackend::connect(config)?;
    println!("  hidden_size = {}", backend.hidden_size());

    let n = args.samples.max(2);
    println!(
        "Running {} calls ({} warmup + {} measured), layers = {:?}",
        n,
        1,
        n - 1,
        layers
    );

    let stats = backend.probe_latency(&layers, n)?;
    println!("\n{stats}");

    let overhead_pct = if stats.total_ms > 0.0 {
        (stats.overhead_ms / stats.total_ms) * 100.0
    } else {
        0.0
    };
    println!(
        "\n  → overhead is {overhead_pct:.1}% of round-trip ({:.2} ms)",
        stats.overhead_ms
    );

    if stats.overhead_ms < 1.0 {
        println!("  gRPC unlikely to help — overhead is already < 1 ms.");
    } else if stats.overhead_ms < 3.0 {
        println!("  gRPC might save 0.5–2 ms/token; worthwhile if token budget is large.");
    } else {
        println!("  gRPC worth evaluating — overhead is significant.");
    }

    Ok(())
}
