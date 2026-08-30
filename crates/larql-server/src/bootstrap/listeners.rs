//! Optional transport listeners spawned alongside the primary HTTP/1.1
//! TCP listener. Currently: the ADR-0019 HTTP/3 (QUIC) listener; the
//! TLS / UDS / gRPC listeners are assembled inline in
//! [`super::serve`].

#[cfg(feature = "http3")]
use tracing::info;

#[cfg(feature = "http3")]
use super::{BoxError, Cli};

/// ADR-0019 — spawn an HTTP/3 listener alongside the existing
/// HTTP/1.1 TCP listener when `--http3-port` is set. Reuses the
/// TLS cert from `--tls-cert`/`--tls-key` if both are set;
/// otherwise auto-generates a self-signed leaf cert and prints its
/// fingerprint so the router operator can pin it.
///
/// The h3 listener serves the same `axum::Router` as the dense
/// path — handlers are identical, only the transport differs.
#[cfg(feature = "http3")]
pub(super) async fn spawn_http3_listener_if_configured(
    cli: &Cli,
    app: axum::Router,
) -> Result<(), BoxError> {
    let Some(port) = cli.http3_port else {
        return Ok(());
    };

    use larql_router_protocol::transport::h3 as h3_transport;
    use larql_router_protocol::transport::quic as quic_transport;

    // Install the rustls ring crypto provider once. Safe to call
    // multiple times — second call is a no-op.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // TLS material: prefer `--tls-cert`/`--tls-key` (reuses the HTTPS
    // pair); fall back to an auto-generated self-signed cert. We
    // print the fingerprint either way so operators have one log
    // line they can hand to the router's `--shard-cert-fingerprint`.
    let tls = if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        let cert_pem = std::fs::read_to_string(cert_path)
            .map_err(|e| format!("read --tls-cert {}: {e}", cert_path.display()))?;
        let key_pem = std::fs::read_to_string(key_path)
            .map_err(|e| format!("read --tls-key {}: {e}", key_path.display()))?;
        // Server name embedded in the cert isn't used by the router
        // when fingerprint-pinning, but we keep the convention here.
        quic_transport::SelfSignedTls {
            cert_pem,
            key_pem,
            fingerprint: String::new(),
            server_name: "larql-server".to_string(),
        }
    } else {
        let generated = quic_transport::self_signed_tls("larql-server")
            .map_err(|e| format!("self-signed cert generation: {e}"))?;
        info!(
            fingerprint = %generated.fingerprint,
            "HTTP/3: generated self-signed cert. Routers must pin this \
             fingerprint via --shard-cert-fingerprint when opting into \
             --http3-shards."
        );
        generated
    };

    let addr: std::net::SocketAddr = format!("{}:{}", cli.host, port).parse()?;
    let endpoint = h3_transport::server_endpoint(addr, &tls)
        .map_err(|e| format!("h3 endpoint bind {addr}: {e}"))?;
    info!("Listening: h3 (HTTP/3 over QUIC) on {addr}");

    tokio::spawn(async move {
        if let Err(e) = h3_transport::serve_axum(endpoint, app).await {
            tracing::error!("h3 listener crashed: {e:#}");
        }
    });
    Ok(())
}
