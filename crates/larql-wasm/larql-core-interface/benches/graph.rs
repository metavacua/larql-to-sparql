use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_core::*;
use std::time::Duration;

const SMALL_EDGE_COUNT: usize = 1_000;
const LARGE_EDGE_COUNT: usize = 25_000;
const RELATION_MODULUS: usize = 8;
const SUBJECT_BUCKET_SIZE: usize = 10;
const BASE_CONFIDENCE: f64 = 0.5;
const CONFIDENCE_STEPS: usize = 50;
const SERIALIZATION_SAMPLE_SIZE: usize = 50;
const SERIALIZATION_MEASUREMENT_SECS: u64 = 10;

fn build_graph(edge_count: usize) -> Graph {
    let mut graph = Graph::new();
    for i in 0..edge_count {
        graph.add_edge(
            Edge::new(
                format!("Entity_{}", i / SUBJECT_BUCKET_SIZE),
                format!("rel_{}", i % RELATION_MODULUS),
                format!("Target_{i}"),
            )
            .with_confidence(BASE_CONFIDENCE + (i % CONFIDENCE_STEPS) as f64 / 100.0),
        );
    }
    graph
}

fn build_connected_graph(node_count: usize) -> Graph {
    let mut graph = Graph::new();
    for i in 0..node_count {
        graph.add_edge(
            Edge::new(
                format!("N{i}"),
                "connects",
                format!("N{}", (i + 1) % node_count),
            )
            .with_confidence(0.9),
        );
    }
    graph
}

fn bench_graph_queries(c: &mut Criterion) {
    let graph = build_graph(LARGE_EDGE_COUNT);

    c.bench_function("select_subject_relation", |b| {
        b.iter(|| black_box(&graph).select(black_box("Entity_42"), Some(black_box("rel_0"))))
    });
    c.bench_function("exists_triple", |b| {
        b.iter(|| {
            black_box(&graph).exists(
                black_box("Entity_42"),
                black_box("rel_4"),
                black_box("Target_424"),
            )
        })
    });
    c.bench_function("keyword_search", |b| {
        b.iter(|| black_box(&graph).search(black_box("Entity_42"), black_box(10)))
    });
}

fn bench_algorithms(c: &mut Criterion) {
    let graph = build_connected_graph(SMALL_EDGE_COUNT);
    let mut group = c.benchmark_group("algorithms");

    group.bench_function("shortest_path_ring", |b| {
        b.iter(|| shortest_path(black_box(&graph), black_box("N0"), black_box("N500")))
    });
    group.bench_function("connected_components_ring", |b| {
        b.iter(|| connected_components(black_box(&graph)))
    });
    group.bench_function("bfs_depth_5", |b| {
        b.iter(|| bfs_traversal(black_box(&graph), black_box("N0"), black_box(5)))
    });

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    group.sample_size(SERIALIZATION_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SERIALIZATION_MEASUREMENT_SECS));

    for edge_count in [SMALL_EDGE_COUNT, LARGE_EDGE_COUNT] {
        let graph = build_graph(edge_count);
        let json = to_bytes(&graph, Format::Json).unwrap();
        let packed = to_packed_bytes(&graph).unwrap();

        group.bench_with_input(
            BenchmarkId::new("json_encode", edge_count),
            &graph,
            |b, g| b.iter(|| to_bytes(black_box(g), Format::Json).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("json_decode", edge_count),
            &json,
            |b, bytes| b.iter(|| from_bytes(black_box(bytes), Format::Json).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("packed_encode", edge_count),
            &graph,
            |b, g| b.iter(|| to_packed_bytes(black_box(g)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("packed_decode", edge_count),
            &packed,
            |b, bytes| b.iter(|| from_packed_bytes(black_box(bytes)).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_queries,
    bench_algorithms,
    bench_serialization
);
criterion_main!(benches);
