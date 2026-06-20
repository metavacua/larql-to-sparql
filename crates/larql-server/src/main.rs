//! larql-server — HTTP server for vindex knowledge queries.
//!
//! Thin binary entry point: parse `Cli`, install tracing, hand off to
//! `bootstrap::serve`. Boot orchestration (vindex loading, warmups, listener
//! setup, grid announce) lives in `larql_server::bootstrap` so that
//! integration tests can drive the same code path without going through
//! `clap::Parser::parse_from`.

use clap::Parser;
use larql_compute::resource_guard::{
    apply_cpu_reservation, detect_physical_memory_bytes, set_memory_cap, CapAllocator,
};
use larql_server::bootstrap::{self, normalize_serve_alias, BoxError, Cli};

#[global_allocator]
static ALLOC: CapAllocator = CapAllocator;

fn main() -> Result<(), BoxError> {
    // Reserve 1 core for the OS and set memory cap from physical RAM.
    // Must run before tokio's thread pool is built.
    let workers = apply_cpu_reservation();
    let phys = detect_physical_memory_bytes();
    if phys > 0 {
        set_memory_cap(phys * 85 / 100);
    }

    // Build tokio runtime with the same thread budget as rayon:
    // available_parallelism() - 1 (minimum 1).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()?;

    rt.block_on(async {
        // Accept both `larql-server <path>` and `larql-server serve <path>`.
        let cli = Cli::parse_from(normalize_serve_alias(std::env::args().collect()));

        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
            )
            .init();

        bootstrap::serve(cli).await
    })
}
