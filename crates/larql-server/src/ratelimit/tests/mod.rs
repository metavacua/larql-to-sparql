//! Unit tests for the token-bucket limiter: spec parsing, bucket
//! semantics, and stale-bucket eviction. The middleware half is
//! exercised through a real axum router in `tests/test_unit_state.rs`.

use super::*;
use std::net::IpAddr;
use std::time::Duration;

#[test]
fn parse_per_minute() {
    let rl = RateLimiter::parse("100/min").unwrap();
    assert_eq!(rl.max_tokens(), 100.0);
    assert!((rl.refill_per_sec() - 100.0 / 60.0).abs() < 0.01);
}

#[test]
fn parse_per_second() {
    let rl = RateLimiter::parse("10/sec").unwrap();
    assert_eq!(rl.max_tokens(), 10.0);
    assert_eq!(rl.refill_per_sec(), 10.0);
}

#[test]
fn parse_per_hour() {
    let rl = RateLimiter::parse("3600/hour").unwrap();
    assert_eq!(rl.max_tokens(), 3600.0);
    assert!((rl.refill_per_sec() - 1.0).abs() < 0.01);
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
    assert!(rl.check(ip2)); // ip2 still has tokens
}

#[test]
fn evict_stale_removes_idle_buckets() {
    let rl = RateLimiter::parse("10/sec").unwrap();
    let ip: IpAddr = "10.0.0.1".parse().unwrap();
    rl.check(ip);
    assert_eq!(rl.bucket_count(), 1);

    // Advance an explicit clock past the idle GC window: the bucket goes.
    let past_window = Instant::now() + Duration::from_secs(BUCKET_IDLE_GC_SECS + 1);
    assert_eq!(rl.evict_stale_at(past_window), 1);
    assert_eq!(rl.bucket_count(), 0);
}

#[test]
fn evict_stale_keeps_active_buckets() {
    let rl = RateLimiter::parse("10/sec").unwrap();
    let ip: IpAddr = "10.0.0.1".parse().unwrap();
    rl.check(ip);

    // Within the idle window nothing is dropped.
    assert_eq!(rl.evict_stale_at(Instant::now()), 0);
    assert_eq!(rl.bucket_count(), 1);
}

#[test]
fn evict_stale_wall_clock_entry_point() {
    let rl = RateLimiter::parse("10/sec").unwrap();
    let ip: IpAddr = "10.0.0.1".parse().unwrap();
    rl.check(ip);
    // A just-used bucket survives the wall-clock sweep.
    assert_eq!(rl.evict_stale(), 0);
    assert_eq!(rl.bucket_count(), 1);
}

#[test]
fn evicted_ip_returns_with_a_fresh_bucket() {
    // Eviction is lossless for well-behaved clients: after the bucket is
    // dropped, the next request re-creates it full.
    let rl = RateLimiter::parse("2/sec").unwrap();
    let ip: IpAddr = "10.0.0.9".parse().unwrap();
    assert!(rl.check(ip));
    assert!(rl.check(ip));
    assert!(!rl.check(ip)); // exhausted

    let past_window = Instant::now() + Duration::from_secs(BUCKET_IDLE_GC_SECS + 1);
    assert_eq!(rl.evict_stale_at(past_window), 1);

    // Fresh bucket: burst available again.
    assert!(rl.check(ip));
}
