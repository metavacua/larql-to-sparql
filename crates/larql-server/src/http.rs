//! HTTP protocol constants shared by routes and middleware.

pub const API_PREFIX: &str = "/v1";
pub const HEALTH_PATH: &str = "/v1/health";
pub const BINARY_FFN_CONTENT_TYPE: &str = "application/x-larql-ffn";
pub const JSON_CONTENT_TYPE: &str = "application/json";
pub const BEARER_PREFIX: &str = "Bearer ";

/// Default upper bound for HTTP request bodies on most routes (FFN binary,
/// embed JSON, walk-ffn binary). Sized for the largest realistic per-request
/// residual + decoder payload at present model dims.
pub const REQUEST_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Larger upper bound for routes that ship full-vocab logits payloads (e.g.
/// `/v1/embed/logits`), where the wire is residual_dim × vocab f32.
pub const REQUEST_BODY_LIMIT_LARGE_BYTES: usize = 256 * 1024 * 1024;
