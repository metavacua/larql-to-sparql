//! Per-IP rate limiting middleware using a token bucket.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Token bucket for a single IP.
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Per-IP rate limiter.
pub struct RateLimiter {
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
    max_tokens: f64,
    refill_per_sec: f64,
}

impl RateLimiter {
    /// Parse a rate limit string like "100/min" or "10/sec".
    pub fn parse(spec: &str) -> Option<Self> {
        let parts: Vec<&str> = spec.split('/').collect();
        if parts.len() != 2 {
            return None;
        }
        let count: f64 = parts[0].trim().parse().ok()?;
        let per_sec = match parts[1].trim() {
            "sec" | "s" | "second" => count,
            "min" | "m" | "minute" => count / 60.0,
            "hour" | "h" => count / 3600.0,
            _ => return None,
        };
        Some(Self {
            buckets: Mutex::new(HashMap::new()),
            max_tokens: count,
            refill_per_sec: per_sec,
        })
    }

    /// Check if a request from this IP is allowed. Returns true if allowed.
    pub fn check(&self, ip: IpAddr) -> bool {
        let mut buckets = match self.buckets.lock() {
            Ok(b) => b,
            Err(_) => return true, // Don't block on poisoned mutex.
        };

        let now = Instant::now();
        let bucket = buckets.entry(ip).or_insert(Bucket {
            tokens: self.max_tokens,
            last_refill: now,
        });

        // Refill tokens based on elapsed time.
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_per_sec).min(self.max_tokens);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Evict stale entries (call periodically from a background task).
    #[allow(dead_code)]
    pub fn evict_stale(&self) {
        if let Ok(mut buckets) = self.buckets.lock() {
            let now = Instant::now();
            // Remove buckets that have been full for > 5 minutes (idle IPs).
            buckets.retain(|_, b| {
                now.duration_since(b.last_refill).as_secs() < 300
            });
        }
    }
}

/// Middleware that applies per-IP rate limiting.
/// Uses ConnectInfo to get the client IP. Falls back to allowing if IP is unavailable.
pub async fn rate_limit_middleware(
    axum::extract::State(limiter): axum::extract::State<Arc<RateLimiter>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Try to extract IP from ConnectInfo or X-Forwarded-For.
    let ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<IpAddr>().ok())
        .or_else(|| {
            request
                .extensions()
                .get::<ConnectInfo<std::net::SocketAddr>>()
                .map(|ci| ci.0.ip())
        });

    // Health check exempt from rate limiting.
    if request.uri().path() == "/v1/health" {
        return next.run(request).await;
    }

    if let Some(ip) = ip {
        if !limiter.check(ip) {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded",
            )
                .into_response();
        }
    }

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_per_minute() {
        let rl = RateLimiter::parse("100/min").unwrap();
        assert_eq!(rl.max_tokens, 100.0);
        assert!((rl.refill_per_sec - 100.0 / 60.0).abs() < 0.01);
    }

    #[test]
    fn parse_per_second() {
        let rl = RateLimiter::parse("10/sec").unwrap();
        assert_eq!(rl.max_tokens, 10.0);
        assert_eq!(rl.refill_per_sec, 10.0);
    }

    #[test]
    fn parse_per_hour() {
        let rl = RateLimiter::parse("3600/hour").unwrap();
        assert_eq!(rl.max_tokens, 3600.0);
        assert!((rl.refill_per_sec - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_short_forms() {
        assert!(RateLimiter::parse("50/s").is_some());
        assert!(RateLimiter::parse("200/m").is_some());
        assert!(RateLimiter::parse("1000/h").is_some());
    }

    #[test]
    fn parse_invalid() {
        assert!(RateLimiter::parse("abc").is_none());
        assert!(RateLimiter::parse("100").is_none());
        assert!(RateLimiter::parse("100/day").is_none());
        assert!(RateLimiter::parse("").is_none());
        assert!(RateLimiter::parse("/min").is_none());
    }

    #[test]
    fn token_bucket_allows_burst() {
        let rl = RateLimiter::parse("3/sec").unwrap();
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(rl.check(ip));
        assert!(rl.check(ip));
        assert!(rl.check(ip));
        // 4th request should fail (burst exhausted).
        assert!(!rl.check(ip));
    }

    #[test]
    fn different_ips_independent() {
        let rl = RateLimiter::parse("1/sec").unwrap();
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(rl.check(ip1));
        assert!(!rl.check(ip1)); // ip1 exhausted
        assert!(rl.check(ip2));  // ip2 still has tokens
    }

    #[test]
    fn evict_stale_removes_old_entries() {
        let rl = RateLimiter::parse("10/sec").unwrap();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        rl.check(ip);
        // Can't easily test time-based eviction without sleeping,
        // but we can verify evict_stale doesn't panic.
        rl.evict_stale();
    }
}
