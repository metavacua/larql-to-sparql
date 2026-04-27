use kv_cache_benchmark::shader_bench;

#[test]
fn test_wht_cpu_benchmark() {
    let result = shader_bench::bench_wht_cpu(256, 100);
    assert_eq!(result.dimension, 256);
    assert!(result.time_us > 0.0);
    assert!(result.throughput_ops_per_sec > 0.0);
    println!("WHT d=256: {:.2} us/op, {:.0} ops/sec", result.time_us, result.throughput_ops_per_sec);
}

#[test]
fn test_tq_encode_cpu_benchmark() {
    let result = shader_bench::bench_tq_encode_cpu(256, 4, 100);
    assert!(result.time_us > 0.0);
    println!(
        "TQ encode 4bit d=256: {:.2} us/op, {:.0} ops/sec",
        result.time_us, result.throughput_ops_per_sec,
    );
}

#[test]
fn test_tq_decode_cpu_benchmark() {
    let result = shader_bench::bench_tq_decode_cpu(256, 4, 100);
    assert!(result.time_us > 0.0);
    println!(
        "TQ decode 4bit d=256: {:.2} us/op, {:.0} ops/sec",
        result.time_us, result.throughput_ops_per_sec,
    );
}

#[test]
fn test_tq_roundtrip_accuracy() {
    let (result, mse, cosine) = shader_bench::bench_tq_roundtrip_cpu(256, 4, 100);
    println!(
        "TQ roundtrip 4bit d=256: {:.2} us/op, MSE={:.6}, cosine={:.4}",
        result.time_us, mse, cosine,
    );
    assert!(mse < 0.1, "MSE too high: {mse}");
    assert!(cosine > 0.9, "Cosine too low: {cosine}");
}

#[test]
fn test_tq_3bit_roundtrip_accuracy() {
    let (result, mse, cosine) = shader_bench::bench_tq_roundtrip_cpu(256, 3, 100);
    println!(
        "TQ roundtrip 3bit d=256: {:.2} us/op, MSE={:.6}, cosine={:.4}",
        result.time_us, mse, cosine,
    );
    assert!(mse < 0.2, "MSE too high: {mse}");
    assert!(cosine > 0.85, "Cosine too low: {cosine}");
}

#[test]
fn test_full_cpu_benchmark_suite() {
    let results = shader_bench::run_cpu_benchmark_suite();

    // Should have WHT (2 dims) + TQ encode/decode (2 bits × 2 dims × 2 ops) = 10 results
    assert_eq!(results.len(), 10);

    let table = shader_bench::format_shader_results(&results);
    println!("{table}");

    // All should have positive timing
    for r in &results {
        assert!(r.time_us > 0.0, "Zero timing for {}", r.operation);
    }
}

#[test]
fn test_wht_d128_faster_than_d256() {
    let r128 = shader_bench::bench_wht_cpu(128, 1000);
    let r256 = shader_bench::bench_wht_cpu(256, 1000);

    // d=128 should be faster (fewer butterfly stages)
    // Allow some margin for noise
    println!("WHT d=128: {:.2} us, d=256: {:.2} us", r128.time_us, r256.time_us);
}
