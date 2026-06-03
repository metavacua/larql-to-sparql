//! larql-server — HTTP server for vindex knowledge queries.
//!
//! Thin binary entry point: parse `Cli`, install tracing, hand off to
//! `bootstrap::serve`. Boot orchestration (vindex loading, warmups, listener
//! setup, grid announce) lives in `larql_server::bootstrap` so that
//! integration tests can drive the same code path without going through
//! `clap::Parser::parse_from`.

use clap::Parser;

use larql_server::bootstrap::{self, normalize_serve_alias, BoxError, Cli};

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Accept both `larql-server <path>` and `larql-server serve <path>`.
    let cli = Cli::parse_from(normalize_serve_alias(std::env::args().collect()));

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    bootstrap::serve(cli).await
}
