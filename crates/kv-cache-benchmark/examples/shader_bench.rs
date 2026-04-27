//! Shader Benchmark: CPU vs Metal for TurboQuant and Graph Walk operations.
//!
//! Usage:
//!   cargo run --example shader_bench

fn main() {
    use kv_cache_benchmark::shader_bench;

    println!("=== KV Cache Shader Benchmark ===\n");

    // CPU suite
    let cpu_results = shader_bench::run_cpu_benchmark_suite();
    println!("{}", shader_bench::format_shader_results(&cpu_results));

    // Accuracy checks
    println!("\n=== Accuracy (TurboQuant roundtrip) ===\n");
    for bits in [3u8, 4] {
        for dim in [128, 256] {
            let (_, mse, cosine) = shader_bench::bench_tq_roundtrip_cpu(dim, bits, 1000);
            println!("  {bits}-bit d={dim}: MSE={mse:.6}, cosine={cosine:.4}");
        }
    }

    // Memory comparison table (KV-reconstructing strategies only).
    let config = kv_cache_benchmark::model_config::ModelConfig::gemma_4b();
    println!("\n{}", kv_cache_benchmark::benchmark::format_comparative_table(
        &config,
        &[
            &kv_cache_benchmark::standard_kv::StandardKv as &dyn kv_cache_benchmark::KvStrategy,
            &kv_cache_benchmark::turboquant::TurboQuant::new(4),
            &kv_cache_benchmark::markov_residual::MarkovResidual::new(512),
        ],
    ));

    // Graph Walk is projected (no K/V reconstruction); report memory separately.
    let gw = kv_cache_benchmark::graph_walk::GraphWalk::gemma_4b();
    println!(
        "\n{} @ 370K tokens: {} bytes per-conversation, {} bytes shared infrastructure",
        gw.name(),
        gw.memory_bytes(370_000),
        gw.shared_bytes(),
    );
}
