//! Criterion benchmarks for the grid routing hot-path (ADR-0012).
//!
//! Measures ns/op for the operations that run on every inference request
//! going through the router.
//!
//! Run with:
//!   cargo bench -p larql-router --bench routing
//!
//! All bench IDs use server counts and layer counts, not model names.

use std::collections::HashMap;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use larql_router::grid::{GridState, ServerEntry};
use larql_router_protocol::LayerLatency;

const SERVER_COUNTS: &[(usize, &str)] = &[(1, "1srv"), (10, "10srv"), (100, "100srv")];
const LAYER_COUNTS: &[(usize, &str)] = &[(30, "30layers"), (62, "62layers")];

fn make_entry(id: usize, layer_start: u32, layer_end: u32) -> ServerEntry {
    ServerEntry {
        server_id: format!("srv-{id}"),
        listen_url: format!("http://10.0.0.{id}:8080"),
        model_id: "bench-model".into(),
        layer_start,
        layer_end,
        cpu_pct: 0.0,
        ram_used: 4 * 1024 * 1024 * 1024,
        requests_in_flight: id as u32 % 10,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
    }
}

/// Build a state with `n_servers`, each owning all `n_layers` layers (replicated).
fn build_state(n_servers: usize, n_layers: usize) -> GridState {
    let mut state = GridState::default();
    for i in 0..n_servers {
        state.register(make_entry(i, 0, (n_layers - 1) as u32));
    }
    state
}

// ── route() hot path ──────────────────────────────────────────────────────────

fn bench_route_single_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_single_layer");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let state = build_state(n_servers, 30);
        group.bench_with_input(BenchmarkId::new(slabel, n_servers), &n_servers, |b, _| {
            b.iter(|| state.route(Some("bench-model"), 15));
        });
    }
    group.finish();
}

// ── route_all() — full forward pass routing ───────────────────────────────────

fn bench_route_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            let state = build_state(n_servers, n_layers);
            let layers: Vec<usize> = (0..n_layers).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &layers,
                |b, layers| {
                    b.iter(|| state.route_all(Some("bench-model"), layers));
                },
            );
        }
    }
    group.finish();
}

// ── update_heartbeat() — load metric update ───────────────────────────────────

fn bench_heartbeat_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/heartbeat_update");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let mut state = build_state(n_servers, 30);
        let server_ids: Vec<String> = (0..n_servers).map(|i| format!("srv-{i}")).collect();
        let layer_stats: Vec<LayerLatency> = (0..30u32)
            .map(|l| LayerLatency {
                layer: l,
                avg_ms: 2.0,
                p99_ms: 5.0,
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::new(slabel, n_servers),
            &server_ids,
            |b, ids| {
                b.iter(|| {
                    // Update the first server's heartbeat.
                    state.update_heartbeat(&ids[0], 50.0, 2 << 30, 5, layer_stats.clone());
                });
            },
        );
    }
    group.finish();
}

// ── rebuild_route_table() — cold-path topology change ────────────────────────

fn bench_rebuild_route_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/rebuild_route_table");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &(n_servers, n_layers),
                |b, &(ns, nl)| {
                    b.iter(|| {
                        // register triggers rebuild_route_table internally.
                        let mut state = GridState::default();
                        for i in 0..ns {
                            state.register(make_entry(i, 0, (nl - 1) as u32));
                        }
                        state
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_route_single_layer,
    bench_route_all,
    bench_heartbeat_update,
    bench_rebuild_route_table,
);
criterion_main!(benches);
