//! Axum middleware applying the per-IP [`RateLimiter`] to each request.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::RateLimiter;
use crate::http::HEALTH_PATH;

/// Runtime configuration for rate-limit middleware.
pub struct RateLimitState {
    pub limiter: Arc<RateLimiter>,
    pub trust_forwarded_for: bool,
}

/// Middleware that applies per-IP rate limiting.
/// Uses ConnectInfo to get the client IP. Falls back to allowing if IP is unavailable.
pub async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<Arc<RateLimitState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Prefer the socket peer. Only trust proxy-provided client IPs when the
    // server was explicitly configured to sit behind a trusted proxy.
    let connect_ip = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip());
    let forwarded_ip = if state.trust_forwarded_for {
        request
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.trim().parse::<IpAddr>().ok())
    } else {
        None
    };
    let ip = forwarded_ip.or(connect_ip);

    // Health check exempt from rate limiting.
    if request.uri().path() == HEALTH_PATH {
        return next.run(request).await;
    }

    if let Some(ip) = ip {
        if !state.limiter.check(ip) {
            return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
        }
    }

    next.run(request).await
}
