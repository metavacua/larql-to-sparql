//! Benchmark: per-call sampling overhead at production vocab sizes.
//!
//! Measures the four sampling configurations the inference loop uses to
//! pick the next token. Reported numbers are the cost per `Sampler::sample`
//! call, exclusive of LM-head gemv and detokenisation. The intent is to
//! confirm sampling is well below the per-step decode budget (~10ms on
//! Metal Q4K) so non-greedy modes don't move the needle on tok/s.
//!
//! Vocab sizes tested:
//!   - 32K   (Llama 1/2)
//!   - 128K  (Gemma 2/3)
//!   - 256K  (Gemma 3 4B+)
//!
//! Run: cargo run --release -p larql-inference --example bench_sampling

use larql_inference::{Sampler, SamplingConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const VOCAB_SIZES: &[usize] = &[32_000, 128_000, 256_000];
const ITERATIONS: usize = 1000;
const WARMUP: usize = 50;

fn make_logits(vocab: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..vocab).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

fn bench_sampling(label: &str, vocab: usize, cfg: SamplingConfig) {
    let logits = make_logits(vocab, 7);
    let mut sampler = Sampler::new(cfg);
    // Warmup
    for _ in 0..WARMUP {
        let _ = sampler.sample(&logits);
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = sampler.sample(&logits);
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_secs_f64() * 1e6 / ITERATIONS as f64;
    println!("  {label:<42}  vocab={vocab:>7}  {per_call_us:>7.2} µs/call");
}

fn bench_topk_path(label: &str, k: usize, cfg: SamplingConfig) {
    // Sparse path: vindex KNN already truncated to k hits.
    let mut rng = StdRng::seed_from_u64(11);
    let hits: Vec<(u32, f32)> = (0..k)
        .map(|i| (i as u32, rng.gen_range(-10.0..10.0)))
        .collect();
    let mut sampler = Sampler::new(cfg);
    for _ in 0..WARMUP {
        let _ = sampler.sample_from_topk(&hits);
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = sampler.sample_from_topk(&hits);
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_secs_f64() * 1e6 / ITERATIONS as f64;
    println!("  {label:<42}  hits={k:>5}    {per_call_us:>7.2} µs/call");
}

fn main() {
    println!("=== larql-inference: Sampling Benchmark ===\n");
    println!("Iterations per measurement: {ITERATIONS} (warmup {WARMUP})\n");

    println!("Full-vocab sampler (Sampler::sample):");
    for &vocab in VOCAB_SIZES {
        bench_sampling("greedy", vocab, SamplingConfig::greedy());
    }
    println!();
    for &vocab in VOCAB_SIZES {
        bench_sampling(
            "temperature=0.8",
            vocab,
            SamplingConfig::temperature(0.8).with_seed(1),
        );
    }
    println!();
    for &vocab in VOCAB_SIZES {
        bench_sampling(
            "temperature=1.0 + top_p=0.9",
            vocab,
            SamplingConfig::temperature(1.0)
                .with_top_p(0.9)
                .with_seed(1),
        );
    }
    println!();
    for &vocab in VOCAB_SIZES {
        bench_sampling(
            "temperature=1.0 + top_k=40",
            vocab,
            SamplingConfig::temperature(1.0).with_top_k(40).with_seed(1),
        );
    }

    println!("\nSparse top-K sampler (Sampler::sample_from_topk):");
    bench_topk_path("greedy", 5, SamplingConfig::greedy());
    bench_topk_path(
        "temperature=0.8 (k=64)",
        64,
        SamplingConfig::temperature(0.8).with_seed(1),
    );
    bench_topk_path(
        "temperature=1.0 + top_p=0.9 (k=64)",
        64,
        SamplingConfig::temperature(1.0)
            .with_top_p(0.9)
            .with_seed(1),
    );
    bench_topk_path(
        "temperature=1.0 + top_k=40 (k=64)",
        64,
        SamplingConfig::temperature(1.0).with_top_k(40).with_seed(1),
    );

    println!();
    println!("Reference: Metal Q4K decode budget ≈ 10ms/tok = 10000 µs.");
    println!("Sampling should be < 1% of that for greedy and < 5% for sampling modes.");
}
