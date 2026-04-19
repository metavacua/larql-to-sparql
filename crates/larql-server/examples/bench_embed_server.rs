//! Embed server benchmark — measures real latency and memory on a live vindex.
//!
//! Tests all operations the embed-service endpoints perform:
//!   1. Load time (embeddings.bin + tokenizer)
//!   2. Embed lookup: single token (decode step), N-token prefill
//!   3. Token encode / decode throughput
//!   4. Binary wire-format encode/decode overhead
//!   5. Memory footprint vs full / ffn-only modes
//!
//! Usage:
//!   cargo run --release -p larql-server --example bench_embed_server -- \
//!     output/gemma3-4b-q4k-v2.vindex
//!
//!   # Optional: also bench logits (requires weights to be present)
//!   cargo run --release -p larql-server --example bench_embed_server -- \
//!     output/gemma3-4b-q4k-v2.vindex --logits

use std::path::PathBuf;
use std::time::Instant;

use larql_vindex::{
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
    ndarray::Array2,
};
use memmap2::Mmap;

// ── Memory ────────────────────────────────────────────────────────────────────

fn mem_mb() -> (u64, u64) {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=,vsz=", "-p", &pid])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            let parts: Vec<&str> = s.split_whitespace().collect();
            let rss = parts.first().and_then(|p| p.parse::<u64>().ok()).unwrap_or(0);
            let vsz = parts.get(1).and_then(|p| p.parse::<u64>().ok()).unwrap_or(0);
            (rss / 1024, vsz / 1024)
        }
        Err(_) => (0, 0),
    }
}

fn checkpoint(label: &str, started: Instant, baseline: (u64, u64)) -> (u64, u64) {
    let (rss, vsz) = mem_mb();
    let dr = rss as i64 - baseline.0 as i64;
    println!(
        "  [{:>5.1}s]  {label:<44}  RSS={rss:>6} MB  Δ={dr:>+7} MB  VSZ={vsz:>7} MB",
        started.elapsed().as_secs_f64()
    );
    (rss, vsz)
}

// ── Bench harness ─────────────────────────────────────────────────────────────

fn bench<F: Fn() -> R, R>(name: &str, warmup: usize, iters: usize, f: F) {
    for _ in 0..warmup { let _ = f(); }
    let t = Instant::now();
    for _ in 0..iters { let _ = f(); }
    let elapsed = t.elapsed();
    let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let ops = iters as f64 / elapsed.as_secs_f64();
    println!(
        "  {:<48}  {:>8.2} µs/op   {:>10.0} ops/s   ({} iters)",
        name, us, ops, iters,
    );
}

fn bench_ns<F: Fn() -> R, R>(name: &str, warmup: usize, iters: usize, f: F) {
    for _ in 0..warmup { let _ = f(); }
    let t = Instant::now();
    for _ in 0..iters { let _ = f(); }
    let elapsed = t.elapsed();
    let ns = elapsed.as_secs_f64() * 1_000_000_000.0 / iters as f64;
    let ops = iters as f64 / elapsed.as_secs_f64();
    println!(
        "  {:<48}  {:>8.1} ns/op   {:>10.0} ops/s   ({} iters)",
        name, ns, ops, iters,
    );
}

// ── Wire format helpers (mirrors routes/embed.rs) ─────────────────────────────

fn encode_embed_binary_request(token_ids: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + token_ids.len() * 4);
    buf.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &id in token_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    buf
}

fn decode_embed_binary_request(bytes: &[u8]) -> Vec<u32> {
    if bytes.len() < 4 { return vec![]; }
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    (0..n)
        .map(|i| u32::from_le_bytes(bytes[4 + i * 4..4 + i * 4 + 4].try_into().unwrap()))
        .collect()
}

fn encode_embed_binary_response(residual: &Array2<f32>) -> Vec<u8> {
    let seq_len = residual.shape()[0];
    let hidden = residual.shape()[1];
    let mut buf = Vec::with_capacity(8 + seq_len * hidden * 4);
    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for &v in residual.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn encode_logits_binary_request(residual: &[f32]) -> Vec<u8> {
    residual.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_embed_server <vindex_path> [--logits]");
        eprintln!("  Example: cargo run --release -p larql-server \\");
        eprintln!("    --example bench_embed_server -- output/gemma3-4b-q4k-v2.vindex");
        std::process::exit(1);
    }
    let vindex_path = PathBuf::from(&args[1]);
    let bench_logits = args.iter().any(|a| a == "--logits");

    println!("LARQL Embed Server Benchmark");
    println!("════════════════════════════");
    println!("Vindex: {}", vindex_path.display());
    println!();

    let started = Instant::now();
    let baseline = mem_mb();
    println!("Memory checkpoints:");
    println!("  [  0.0s]  {:<44}  RSS={:>6} MB", "baseline", baseline.0);

    // ── Load config ───────────────────────────────────────────────────────────
    let config = load_vindex_config(&vindex_path).expect("load config");
    println!();
    println!("Model:       {}", config.model);
    println!("Hidden:      {}", config.hidden_size);
    println!("Vocab:       {}", config.vocab_size);
    println!("Embed scale: {:.4}", config.embed_scale);
    println!("Layers:      {}", config.num_layers);
    println!("Quant:       {:?}", config.quant);
    println!();

    // ── Load tokenizer ────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let tok_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let after_tok = checkpoint("after tokenizer load", started, baseline);
    println!("  Tokenizer load: {:.1}ms", tok_ms);

    // ── Load embeddings ───────────────────────────────────────────────────────
    println!();
    println!("Loading embeddings.bin ({} × {} f32 = {:.1} GB)...",
        config.vocab_size, config.hidden_size,
        config.vocab_size as f64 * config.hidden_size as f64 * 4.0 / 1e9
    );
    let t0 = Instant::now();
    let (embeddings, embed_scale) = load_vindex_embeddings(&vindex_path).expect("load embeddings");
    let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let after_embed = checkpoint("after embeddings load", started, baseline);
    println!("  Embeddings load: {:.1}ms  ({:.2} GB/s effective throughput)",
        embed_ms,
        (config.vocab_size as f64 * config.hidden_size as f64 * 2.0 / 1e9) / (embed_ms / 1000.0)
    );
    let _ = (after_tok, after_embed);

    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let scale = embed_scale;

    // ── Embed lookup benchmarks ───────────────────────────────────────────────
    println!();
    println!("── Embed lookup ──");

    // Single hot token (decode step — most common case)
    bench_ns("embed 1 token (decode step)", 10_000, 1_000_000, || {
        let tok: usize = 9515 % vocab;
        let row = embeddings.row(tok);
        std::hint::black_box(row[0] * scale) // prevent elision
    });

    // Full row copy into Vec — this is what the handler actually returns
    bench_ns("embed 1 token (full row copy)", 10_000, 1_000_000, || {
        let tok: usize = 9515 % vocab;
        let row = embeddings.row(tok);
        let v: Vec<f32> = row.iter().map(|&v| v * scale).collect();
        std::hint::black_box(v.len())
    });

    // Prefill: 32 / 128 / 512 tokens
    for &seq_len in &[1usize, 32, 128, 512] {
        let token_ids: Vec<usize> = (0..seq_len).map(|i| (i * 7 + 13) % vocab).collect();
        let iters = if seq_len <= 32 { 50_000 } else if seq_len <= 128 { 10_000 } else { 2_000 };
        bench(&format!("embed {seq_len} tokens (prefill)"), iters / 10, iters, || {
            let mut h = Array2::<f32>::zeros((seq_len, hidden));
            for (i, &tok) in token_ids.iter().enumerate() {
                let src = embeddings.row(tok);
                for (dst, &s) in h.row_mut(i).iter_mut().zip(src.iter()) {
                    *dst = s * scale;
                }
            }
            h
        });
    }

    // ── Tokenizer benchmarks ──────────────────────────────────────────────────
    println!();
    println!("── Tokenizer ──");

    let prompts = [
        "Paris",
        "The capital of France is",
        "In a distant future where technology has advanced beyond our wildest dreams, humanity found itself",
    ];
    for prompt in &prompts {
        let words = prompt.split_whitespace().count();
        bench(&format!("encode {words}w: {:.30}…", prompt), 1_000, 50_000, || {
            tokenizer.encode(*prompt, false).unwrap()
        });
    }

    // Decode single token
    bench_ns("decode 1 token id (9515)", 10_000, 1_000_000, || {
        tokenizer.decode(&[9515u32], true).unwrap()
    });
    bench_ns("decode 5 token ids", 10_000, 500_000, || {
        tokenizer.decode(&[9515u32, 235, 1234, 100, 7], true).unwrap()
    });

    // ── Wire format benchmarks ────────────────────────────────────────────────
    println!();
    println!("── Binary wire format ──");

    bench_ns("encode embed request (1 token)", 100_000, 5_000_000, || {
        encode_embed_binary_request(&[9515u32])
    });
    bench_ns("encode embed request (512 tokens)", 1_000, 100_000, || {
        let ids: Vec<u32> = (0..512u32).collect();
        encode_embed_binary_request(&ids)
    });
    bench_ns("decode embed request (1 token)", 100_000, 5_000_000, || {
        let req = [0x01, 0x00, 0x00, 0x00, 0x2B, 0x25, 0x00, 0x00u8];
        decode_embed_binary_request(&req)
    });

    // Build a 1-token residual for response encoding
    let single_residual = {
        let mut h = Array2::<f32>::zeros((1, hidden));
        for j in 0..hidden { h[[0, j]] = j as f32 / hidden as f32; }
        h
    };
    bench(&format!("encode embed response (1×{hidden} f32)"), 10_000, 500_000, || {
        encode_embed_binary_response(&single_residual)
    });

    let logits_request: Vec<f32> = (0..hidden).map(|i| i as f32 / hidden as f32).collect();
    bench_ns("encode logits request (f32 slice → bytes)", 10_000, 500_000, || {
        encode_logits_binary_request(&logits_request)
    });

    // ── JSON serialization ────────────────────────────────────────────────────
    println!();
    println!("── JSON serialization ──");

    let sample_embed_resp = serde_json::json!({
        "residual": vec![vec![0.1f32; 256]; 1],
        "seq_len": 1,
        "hidden_size": hidden,
        "latency_ms": 0.01f32,
    });
    bench(&format!("JSON embed response (1×{hidden} floats)"), 1_000, 50_000, || {
        serde_json::to_string(&sample_embed_resp).unwrap()
    });

    let sample_logits_resp = serde_json::json!({
        "top_k": [
            {"token_id": 9515u32, "token": "Paris", "prob": 0.801f32},
            {"token_id": 235u32, "token": "the", "prob": 0.042f32},
            {"token_id": 100u32, "token": "a", "prob": 0.012f32},
            {"token_id": 5u32, "token": "▁", "prob": 0.008f32},
            {"token_id": 1u32, "token": "<bos>", "prob": 0.003f32},
        ],
        "latency_ms": 2.1f32,
    });
    bench("JSON logits response (top-5)", 1_000, 500_000, || {
        serde_json::to_string(&sample_logits_resp).unwrap()
    });

    // ── Logits projection (optional) ──────────────────────────────────────────
    if bench_logits {
        println!();
        println!("── Logits projection (lm_head matmul via tied embeddings) ──");
        println!("  NOTE: embed server uses weights.lm_head loaded separately.");
        println!("  Benchmarking embeddings-as-lm_head approximation (tied-weight models).");

        let query: Vec<f32> = (0..hidden).map(|i| (i as f32) / (hidden as f32)).collect();
        let after_logits_baseline = mem_mb();
        println!();

        // Sub-vocab slice to avoid OOM on systems with <16 GB RAM
        let sub_vocab = vocab.min(65536);
        let lm_head = embeddings.slice(larql_vindex::ndarray::s![..sub_vocab, ..]);
        println!("  Using first {sub_vocab} rows of lm_head (full vocab = {vocab})");

        bench(&format!("logits matmul {sub_vocab}×{hidden} (dot products)"), 10, 200, || {
            let mut scores: Vec<f32> = Vec::with_capacity(sub_vocab);
            for row in lm_head.rows() {
                scores.push(row.iter().zip(query.iter()).map(|(&e, &r)| e * r).sum());
            }
            // top-5 partial sort
            let k = 5.min(scores.len());
            scores.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap());
            scores.truncate(k);
            scores
        });

        let after_logits = mem_mb();
        let dr = after_logits.0 as i64 - after_logits_baseline.0 as i64;
        println!("  RSS after logits bench: {} MB (Δ{:+} MB)", after_logits.0, dr);

        println!();
        println!("  Full-vocab projection ({}×{}):", vocab, hidden);
        println!("    CPU naive:  ~{:.0}ms", vocab as f64 * hidden as f64 * 2.0 / 4e9 * 1000.0);
        println!("    BLAS gemv:  ~{:.1}ms  (@ ~50 GFLOP/s)", vocab as f64 * hidden as f64 * 2.0 / 50e9 * 1000.0);
        println!("    Metal gemv: ~{:.2}ms  (@ ~2 TFLOP/s on Apple Silicon)", vocab as f64 * hidden as f64 * 2.0 / 2000e9 * 1000.0);
    }

    // ── f16-at-rest store benchmark ───────────────────────────────────────────
    println!();
    println!("── f16-at-rest store (EmbedStoreF16, ADR-0008) ──");

    let embed_bin = vindex_path.join("embeddings.bin");
    let expected_f16 = vocab * hidden * 2;
    let f16_file_size = std::fs::metadata(&embed_bin).map(|m| m.len()).unwrap_or(0);

    if f16_file_size as usize == expected_f16 {
        // Open f16 mmap (no copy, no decode — kernel maps pages on access).
        let t0 = Instant::now();
        let f16_file = std::fs::File::open(&embed_bin).unwrap();
        let f16_mmap: Mmap = unsafe { Mmap::map(&f16_file).unwrap() };
        let open_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Drop the f32 matrix to get a clean measurement — we measure the
        // RSS overhead of just the mmap after cold open (before any page faults).
        drop(embeddings);
        let (rss_after_mmap, _) = mem_mb();
        println!("  mmap open (cold, no pages faulted):  {:.1}ms  RSS={} MB",
            open_ms, rss_after_mmap);

        // Touch 5000 tokens (L1 cache fill): fault exactly those pages.
        let l1_cap = 5_000usize;
        let mut l1_cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
        let t0 = Instant::now();
        for i in 0..l1_cap {
            let tok = (i * 7 + 13) % vocab;
            if !l1_cache.contains_key(&(tok as u32)) {
                let offset = tok * hidden * 2;
                let row: Vec<f32> = f16_mmap[offset..offset + hidden * 2]
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        larql_models::quant::half::f16_to_f32(bits) * embed_scale
                    })
                    .collect();
                l1_cache.insert(tok as u32, row);
            }
        }
        let fill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (rss_after_l1, _) = mem_mb();
        println!("  L1 cache fill ({l1_cap} tokens):          {:.1}ms  RSS={} MB",
            fill_ms, rss_after_l1);

        // Benchmark: L1 hit (hot token, already in HashMap)
        // Use the first key actually inserted into the cache.
        let l1_hot_tok = *l1_cache.keys().next().unwrap();
        bench_ns("f16 embed 1 token — L1 hit", 100_000, 1_000_000, || {
            let row = l1_cache.get(&l1_hot_tok).unwrap();
            std::hint::black_box(row[0])
        });

        // Benchmark: L1 miss — decode from f16 mmap every time (cold)
        bench_ns("f16 embed 1 token — mmap decode (L1 miss)", 10_000, 500_000, || {
            let tok = 9515usize % vocab;
            let offset = tok * hidden * 2;
            let raw = &f16_mmap[offset..offset + hidden * 2];
            let row: Vec<f32> = raw.chunks_exact(2).map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                larql_models::quant::half::f16_to_f32(bits) * embed_scale
            }).collect();
            std::hint::black_box(row[0])
        });

        // Prefill via f16 decode
        for &seq_len in &[1usize, 32, 128, 512] {
            let token_ids: Vec<usize> = (0..seq_len).map(|i| (i * 7 + 13) % vocab).collect();
            let iters = if seq_len <= 32 { 20_000 } else if seq_len <= 128 { 5_000 } else { 1_000 };
            bench(&format!("f16 embed {seq_len} tokens (prefill, mmap decode)"), iters / 10, iters, || {
                let mut h = Array2::<f32>::zeros((seq_len, hidden));
                for (i, &tok) in token_ids.iter().enumerate() {
                    let offset = tok * hidden * 2;
                    let raw = &f16_mmap[offset..offset + hidden * 2];
                    let mut dst = h.row_mut(i);
                    for (j, b) in raw.chunks_exact(2).enumerate() {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        dst[j] = larql_models::quant::half::f16_to_f32(bits) * embed_scale;
                    }
                }
                h
            });
        }

        // Final RSS — all accessed pages now resident.
        let (rss_full, _) = mem_mb();
        println!();
        println!("  RSS after prefill bench (pages faulted): {} MB", rss_full);

        // ── Memory comparison: f32 heap vs f16 mmap ──
        println!();
        println!("── Memory comparison: f32 heap vs f16 mmap ──");
        let embed_f32_gb = vocab as f64 * hidden as f64 * 4.0 / 1e9;
        let embed_f16_gb = vocab as f64 * hidden as f64 * 2.0 / 1e9;
        let tok_gb = 0.234f64;
        let l1_gb = l1_cap as f64 * hidden as f64 * 4.0 / 1e9;
        println!("  embeddings.bin on disk (f16):          {:.2} GB", embed_f16_gb);
        println!("  f32 heap (eager decode):               {:.2} GB", embed_f32_gb);
        println!("  f16 mmap + L1 cache ({l1_cap} tokens):   {:.2} GB  ({:.0} MB mmap + {:.0} MB L1)",
            embed_f16_gb + l1_gb,
            embed_f16_gb * 1000.0, l1_gb * 1000.0);
        println!();
        println!("  --embed-only (f32 heap):               ~{:.1} GB RSS",
            embed_f32_gb + tok_gb);
        println!("  --embed-only (f16 mmap, ADR-0008):     ~{:.1} GB RSS  ({:.0}% reduction)",
            embed_f16_gb + l1_gb + tok_gb,
            (1.0 - (embed_f16_gb + l1_gb) / embed_f32_gb) * 100.0);
        let _ = f16_mmap;
    } else {
        println!("  embeddings.bin is f32 (size {} != f16 expected {}) — f16 bench skipped",
            f16_file_size, expected_f16);
        let (final_rss, _) = mem_mb();
        println!("  RSS: {} MB", final_rss);
    }

    println!();
    println!("  Logits: {:.1}ms CPU (full vocab), ~{:.2}ms Metal",
        vocab as f64 * hidden as f64 * 2.0 / 4e9 * 1000.0,
        vocab as f64 * hidden as f64 * 2.0 / 2000e9 * 1000.0);
    println!();
    println!("  Run with --logits to benchmark the lm_head projection.");

    println!();
    println!("  Total elapsed: {:.1}s", started.elapsed().as_secs_f64());
}
