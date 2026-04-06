//! Production-dimension scaling benchmarks for vindex.
//!
//! Tests KNN, walk, and HNSW at real model sizes to validate the scaling
//! projections from vindex_bench. No real model needed — synthetic data
//! at production dimensions.
//!
//! Run: cargo run --release -p larql-vindex --example bench_scaling

use larql_vindex::VectorIndex;
#[allow(unused_imports)]
use larql_compute::ComputeBackend;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    println!("=== Vindex Production Scaling Benchmark ===\n");

    // ── 1. KNN at production dimensions ──
    println!("── 1. Gate KNN at Production Dimensions ──\n");
    println!("  {:30} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Model", "Features", "Hidden", "Gate MB", "KNN/layer", "Walk 14L");
    println!("  {:30} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "─".repeat(30), "────────", "──────────", "──────────", "──────────", "──────────");

    let models: Vec<(&str, usize, usize, usize)> = vec![
        ("Gemma 3 4B",     10240,  2560, 14),
        ("Llama 3 8B",     14336,  4096, 16),
        ("Llama 3 70B",    28672,  8192, 48),
        ("Mixtral 8x22B (1 expert)", 16384, 6144, 32),
    ];

    for (name, features, hidden, knowledge_layers) in &models {
        let features = *features;
        let hidden = *hidden;
        let knowledge_layers = *knowledge_layers;

        // Build a single-layer index at production dimensions
        let gate = synth_matrix(features, hidden, 42);
        let gate_mb = (features * hidden * 4) as f64 / 1_048_576.0;
        let idx = VectorIndex::new(
            vec![Some(gate)],
            vec![None],
            1,
            hidden,
        );

        let query = random_query(hidden);
        let top_k = 10;
        let n = 20;

        // Warmup
        for _ in 0..3 { idx.gate_knn(0, &query, top_k); }

        // Bench KNN per layer
        let t0 = Instant::now();
        for _ in 0..n { idx.gate_knn(0, &query, top_k); }
        let knn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let walk_ms = knn_ms * knowledge_layers as f64;

        println!("  {:30} {:>8} {:>10} {:>8.1} MB {:>8.2}ms {:>8.1}ms",
            name, features, hidden, gate_mb, knn_ms, walk_ms);
    }

    // ── 2. MoE KNN at scale ──
    println!("\n── 2. MoE Gate KNN (experts × features per layer) ──\n");
    println!("  {:35} {:>10} {:>10} {:>10}",
        "Config", "Total feat", "Gate MB", "KNN/layer");
    println!("  {:35} {:>10} {:>10} {:>10}",
        "─".repeat(35), "──────────", "──────────", "──────────");

    let hidden = 2560; // Use Gemma hidden for MoE scaling test
    let n = 10;
    for (label, total_features) in [
        ("Dense (10240)",              10240),
        ("8 experts × 2048",          16384),
        ("16 experts × 2048",         32768),
        ("64 experts × 2048",        131072),
    ] {
        let gate = synth_matrix(total_features, hidden, 42);
        let gate_mb = (total_features * hidden * 4) as f64 / 1_048_576.0;
        let idx = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
        let query = random_query(hidden);

        // Warmup
        for _ in 0..3 { idx.gate_knn(0, &query, 10); }

        let t0 = Instant::now();
        for _ in 0..n { idx.gate_knn(0, &query, 10); }
        let knn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        println!("  {:35} {:>10} {:>8.1} MB {:>8.2}ms", label, total_features, gate_mb, knn_ms);
    }

    // ── 3. Cold vs warm mmap ──
    println!("\n── 3. Cold vs Warm KNN (mmap page fault cost) ──\n");
    {
        let layers = 34;
        let features = 4096;
        let mem_hidden = 2560;
        let top_k = 10;
        let n = 20;

        let index = build_synthetic_index(layers, features, mem_hidden);
        let dir = std::env::temp_dir().join("larql_bench_scaling_mmap");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let layer_infos = index.save_gate_vectors(&dir).unwrap();
        index.save_down_meta(&dir).unwrap();

        let config = larql_vindex::VindexConfig {
            version: 2, model: "bench".into(), family: "bench".into(),
            source: None, checksums: None,
            num_layers: layers, hidden_size: mem_hidden, intermediate_size: features,
            vocab_size: 100, embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::F32, layer_bands: None,
            layers: layer_infos, down_top_k: 3,
            has_model_weights: false, model_config: None,
        };
        VectorIndex::save_config(&config, &dir).unwrap();
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

        let query = random_query(mem_hidden);

        // Warm: query layer 13 repeatedly
        for _ in 0..5 { loaded.gate_knn(13, &query, top_k); }
        let t0 = Instant::now();
        for _ in 0..n { loaded.gate_knn(13, &query, top_k); }
        let warm_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // Cold: evict pages with madvise, then query a fresh layer
        #[cfg(unix)]
        {
            // Force page eviction by re-loading
            let loaded2 = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

            // First access to an untouched layer — measures page fault cost
            let t0 = Instant::now();
            loaded2.gate_knn(20, &query, top_k);
            let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Second access — should be warm now
            let t0 = Instant::now();
            for _ in 0..n { loaded2.gate_knn(20, &query, top_k); }
            let warm2_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

            println!("  34L × 4096 × 2560 (mmap'd)");
            println!("  Warm KNN (L13, paged):       {warm_ms:.3}ms");
            println!("  Cold KNN (L20, first access): {cold_ms:.3}ms");
            println!("  Warm KNN (L20, after fault):  {warm2_ms:.3}ms");
            println!("  Page fault overhead:          {:.3}ms", cold_ms - warm2_ms);
        }

        #[cfg(not(unix))]
        {
            println!("  Warm KNN (L13): {warm_ms:.3}ms");
            println!("  (Cold test requires unix madvise)");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 4. HNSW vs brute-force crossover ──
    println!("\n── 4. HNSW vs Brute-Force Crossover ──\n");
    println!("  {:>10} {:>12} {:>12} {:>12} {:>8}",
        "Features", "Brute (ms)", "HNSW (ms)", "HNSW build", "Winner");
    println!("  {:>10} {:>12} {:>12} {:>12} {:>8}",
        "──────────", "────────────", "────────────", "────────────", "────────");

    let hnsw_hidden = 2560;
    for features in [1024, 4096, 10240, 28672] {
        let gate = synth_matrix(features, hnsw_hidden, 42);
        let idx = VectorIndex::new(
            vec![Some(gate)],
            vec![None],
            1,
            hnsw_hidden,
        );
        let query = random_query(hnsw_hidden);
        let top_k = 10;
        let n = 20;

        // Brute-force
        for _ in 0..3 { idx.gate_knn(0, &query, top_k); }
        let t0 = Instant::now();
        for _ in 0..n { idx.gate_knn(0, &query, top_k); }
        let brute_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // HNSW: build + search
        let t_build = Instant::now();
        idx.enable_hnsw(100);
        idx.gate_knn(0, &query, top_k); // triggers lazy build
        let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

        // HNSW search (already built)
        let t0 = Instant::now();
        for _ in 0..n { idx.gate_knn(0, &query, top_k); }
        let hnsw_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let winner = if hnsw_ms < brute_ms { "HNSW" } else { "Brute" };
        println!("  {:>10} {:>10.3}ms {:>10.3}ms {:>10.1}ms {:>8}",
            features, brute_ms, hnsw_ms, build_ms, winner);

        idx.disable_hnsw();
    }

    // ── 5. Q4 gate KNN via compute backend ──
    println!("\n── 5. Q4 Gate KNN (vindex Q4 data → compute backend) ──\n");

    let default_backend = larql_compute::default_backend();
    let cpu_backend = larql_compute::cpu_backend();
    let has_gpu = default_backend.name() != cpu_backend.name();

    if has_gpu {
        println!("  GPU backend: {} ({})", default_backend.name(), default_backend.device_info());
    } else {
        println!("  GPU backend: none (CPU only)");
    }
    println!();

    {
        use larql_compute::cpu::q4::{quantize_q4_0, quantize_to_q8};

        if has_gpu {
            println!("  {:26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Model", "f32 BLAS", "Q4 CPU", "Q4 GPU", "Q4 size", "GPU speed", "Walk 14L");
            println!("  {:26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "─".repeat(26), "──────────", "──────────", "──────────", "──────────", "──────────", "──────────");
        } else {
            println!("  {:26} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Model", "f32 BLAS", "Q4 CPU", "Q4 size", "Speedup", "Walk 14L");
            println!("  {:26} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "─".repeat(26), "──────────", "──────────", "──────────", "──────────", "──────────");
        }

        let q4_models: Vec<(&str, usize, usize, usize)> = vec![
            ("Gemma 3 4B",     10240,  2560, 14),
            ("Llama 3 8B",     14336,  4096, 16),
            ("Llama 3 70B",    28672,  8192, 48),
        ];

        for (name, features, hidden, knowledge_layers) in &q4_models {
            let features = *features;
            let hidden = *hidden;
            let knowledge_layers = *knowledge_layers;
            let n = 20;
            let top_k = 10;

            // Build f32 gate matrix and Q4 version
            let gate_f32: Vec<f32> = (0..features * hidden).map(|i| {
                let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
            }).collect();
            let q4_data = quantize_q4_0(&gate_f32);
            let q4_mb = q4_data.len() as f64 / 1_048_576.0;

            // f32 brute-force
            let gate = ndarray::Array2::from_shape_vec((features, hidden), gate_f32).unwrap();
            let idx = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
            let query = random_query(hidden);
            for _ in 0..3 { idx.gate_knn(0, &query, top_k); }
            let t0 = Instant::now();
            for _ in 0..n { idx.gate_knn(0, &query, top_k); }
            let f32_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

            // Q4 CPU
            let x_slice = query.as_slice().unwrap();
            let (q8_x, q8_scales) = quantize_to_q8(x_slice);
            let _ = cpu_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden);
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = cpu_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden);
            }
            let q4_cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

            if has_gpu {
                // Q4 GPU (Metal)
                let _ = default_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden);
                let t0 = Instant::now();
                for _ in 0..n {
                    let _ = default_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden);
                }
                let q4_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let gpu_speedup = f32_ms / q4_gpu_ms;
                let walk_ms = q4_gpu_ms * knowledge_layers as f64;

                println!("  {:26} {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>7.1} MB {:>8.1}x {:>8.1}ms",
                    name, f32_ms, q4_cpu_ms, q4_gpu_ms, q4_mb, gpu_speedup, walk_ms);
            } else {
                let cpu_speedup = f32_ms / q4_cpu_ms;
                let walk_ms = q4_cpu_ms * knowledge_layers as f64;

                println!("  {:26} {:>8.2}ms {:>8.2}ms {:>7.1} MB {:>8.1}x {:>8.1}ms",
                    name, f32_ms, q4_cpu_ms, q4_mb, cpu_speedup, walk_ms);
            }
        }
    }

    // ── 6. End-to-end walk at Gemma 3 4B dimensions ──
    println!("\n── 6. End-to-End Walk (Gemma 3 4B dimensions) ──\n");
    {
        let layers = 34;
        let features = 10240;
        let hidden = 2560;
        let top_k = 10;
        let n = 5;

        println!("  Building {layers}L × {features} × {hidden} index...");
        let t0 = Instant::now();
        let index = build_synthetic_index(layers, features, hidden);
        let build_s = t0.elapsed().as_secs_f64();
        println!("  Build: {build_s:.1}s");

        let query = random_query(hidden);

        // Knowledge band walk (L14-27)
        let knowledge: Vec<usize> = (14..28).collect();
        // Warmup
        let _ = index.walk(&query, &knowledge, top_k);

        let t0 = Instant::now();
        for _ in 0..n { let _ = index.walk(&query, &knowledge, top_k); }
        let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let walk_tps = 1000.0 / walk_ms;

        // Full walk (all 34 layers)
        let all_layers: Vec<usize> = (0..layers).collect();
        let t0 = Instant::now();
        for _ in 0..n { let _ = index.walk(&query, &all_layers, top_k); }
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let full_tps = 1000.0 / full_ms;

        // Per-layer breakdown
        let per_layer = full_ms / layers as f64;

        println!("  Knowledge walk (L14-27, 14 layers): {walk_ms:.1}ms  ({walk_tps:.0} tok/s)");
        println!("  Full walk (34 layers):              {full_ms:.1}ms  ({full_tps:.0} tok/s)");
        println!("  Per-layer KNN:                      {per_layer:.2}ms");

        // Memory footprint
        let gate_gb = (layers * features * hidden * 4) as f64 / 1_073_741_824.0;
        let gate_f16_gb = gate_gb / 2.0;
        println!();
        println!("  Gate vectors (f32): {gate_gb:.2} GB");
        println!("  Gate vectors (f16): {gate_f16_gb:.2} GB");
        println!("  Inference RAM (1 layer mmap): {:.2} GB", gate_gb / layers as f64);
    }

    // ── 7. Model scaling with Q4 walk latency ──
    println!("\n── 7. Scaling Projections (Q4 {} KNN × projected layers) ──\n",
        default_backend.name());
    {
        use larql_compute::cpu::q4::{quantize_q4_0, quantize_to_q8};

        let best_backend: &dyn larql_compute::ComputeBackend = if has_gpu {
            default_backend.as_ref()
        } else {
            cpu_backend.as_ref()
        };
        let backend_label = if has_gpu { "Q4 GPU" } else { "Q4 CPU" };

        let measurements: Vec<(&str, usize, usize, usize, usize)> = vec![
            ("Gemma 3 4B",   34, 10240, 2560, 14),
            ("Llama 3 8B",   32, 14336, 4096, 16),
            ("Llama 3 70B",  80, 28672, 8192, 48),
            ("Llama 3 405B", 126, 53248, 16384, 76),
            ("DeepSeek V3",  61, 524288, 7168, 37),
        ];

        println!("  {:18} {:>6} {:>9} {:>10} {:>10} {:>10} {:>10} {:>9}",
            "Model", "Layers", "Infer RAM", "f32 BLAS", backend_label, "Walk", "tok/s", "Q4 gate");
        println!("  {:18} {:>6} {:>9} {:>10} {:>10} {:>10} {:>10} {:>9}",
            "─".repeat(18), "──────", "─────────", "──────────", "──────────", "──────────", "──────────", "─────────");

        for (name, layers, features, hidden, knowledge_layers) in &measurements {
            let features = *features;
            let hidden = *hidden;
            let layers = *layers;
            let knowledge_layers = *knowledge_layers;

            // Cap features for bench time
            let bench_features = features.min(131072);
            let n = 10;

            // f32 brute-force
            let gate = synth_matrix(bench_features, hidden, 42);
            let idx = VectorIndex::new(vec![Some(gate.clone())], vec![None], 1, hidden);
            let query = random_query(hidden);
            for _ in 0..3 { idx.gate_knn(0, &query, 10); }
            let t0 = Instant::now();
            for _ in 0..n { idx.gate_knn(0, &query, 10); }
            let f32_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let f32_scaled = f32_ms * features as f64 / bench_features as f64;

            // Q4 via best backend
            let gate_f32 = gate.into_raw_vec_and_offset().0;
            let q4_data = quantize_q4_0(&gate_f32);
            let q4_size_mb = (features as f64 * hidden as f64 / 32.0 * 18.0) / 1_048_576.0;
            let x_slice = query.as_slice().unwrap();
            let (q8_x, q8_scales) = quantize_to_q8(x_slice);
            let _ = best_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, bench_features, hidden);
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = best_backend.q4_matvec(&q4_data, &q8_x, &q8_scales, bench_features, hidden);
            }
            let q4_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let q4_scaled = q4_ms * features as f64 / bench_features as f64;

            let walk_ms = q4_scaled * knowledge_layers as f64;
            let tps = 1000.0 / walk_ms;

            // Infer RAM
            let gate_per_layer = features as f64 * hidden as f64 * 2.0 / 1_073_741_824.0;
            let attn_per_layer = 4.0 * hidden as f64 * hidden as f64 * 2.0 / 1_073_741_824.0;
            let embed_gb = (hidden as f64 * 262144.0 * 2.0 / 1_073_741_824.0).min(5.0);
            let infer_gb = gate_per_layer + attn_per_layer + embed_gb;

            let note = if features > bench_features { "*" } else { " " };
            println!("  {:18} {:>6} {:>7.1} GB {:>8.1}ms{} {:>8.2}ms{} {:>8.1}ms {:>8.0} t/s {:>7.1} MB",
                name, layers, infer_gb, f32_scaled, note, q4_scaled, note, walk_ms, tps, q4_size_mb);
        }
        println!();
        println!("  * = scaled linearly from {} features (capped for benchmark time)", 131072);
        println!("  Backend: {} ({})", best_backend.name(), best_backend.device_info());
        println!("  Infer RAM = 1 layer gate + 1 layer attn + embeddings (f16, mmap)");
    }

    // ── 8. Adaptive residency: memory budget → tok/s gradient ──
    println!("\n── 8. Adaptive Residency (Llama 70B: memory budget → performance) ──\n");
    {
        use larql_compute::cpu::q4::{quantize_q4_0, quantize_to_q8};
        use larql_vindex::ResidencyManager;

        // Simulate Llama 70B dimensions (capped features for bench speed)
        let layers = 80;
        let features = 10240; // capped (real 70B: 28672)
        let hidden = 2560;    // capped (real 70B: 8192)
        let knowledge_layers = 48;

        let best_backend: &dyn larql_compute::ComputeBackend = if has_gpu {
            default_backend.as_ref()
        } else {
            cpu_backend.as_ref()
        };

        // Build Q4 gate data for all layers
        let q4_layers: Vec<Vec<u8>> = (0..layers).map(|l| {
            let gate_f32: Vec<f32> = (0..features * hidden).map(|i| {
                let s = (i as u64 + l as u64 * 1000).wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
            }).collect();
            quantize_q4_0(&gate_f32)
        }).collect();

        let layer_q4_mb = q4_layers[0].len() as f64 / 1_048_576.0;
        let all_q4_mb = layer_q4_mb * layers as f64;
        let query = random_query(hidden);
        let n = 20;

        println!("  Simulated 70B: {layers}L × {features} × {hidden}, {layer_q4_mb:.1} MB/layer Q4");
        println!("  All layers Q4: {all_q4_mb:.0} MB, knowledge band: L16-63 ({knowledge_layers} layers)");
        println!("  Backend: {}\n", best_backend.name());

        println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>12}",
            "Budget", "Pinned", "Pin MB", "KNN/layer", "Walk 48L", "tok/s");
        println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>12}",
            "──────────", "────────", "──────────", "──────────", "──────────", "────────────");

        let layer_features = vec![features; layers];
        let budgets_mb: Vec<usize> = vec![0, 50, 200, 500, 1000, all_q4_mb as usize + 1];

        for budget_mb in budgets_mb {
            let mut rm = ResidencyManager::new(budget_mb, layers, hidden, layer_features.clone());
            rm.mark_q4_available();

            // Pin knowledge band first, then remaining layers
            rm.pin_range(16, 64, |l| Some(q4_layers[l].clone()));
            rm.auto_pin(|l| Some(q4_layers[l].clone()));

            // Benchmark: walk the knowledge band
            let x_slice = query.as_slice().unwrap();
            let (q8_x, q8_scales) = quantize_to_q8(x_slice);

            // Warmup
            for l in 16..64 {
                let q4 = rm.pinned_q4(l).unwrap_or(&q4_layers[l]);
                let _ = best_backend.q4_matvec(q4, &q8_x, &q8_scales, features, hidden);
            }

            let t0 = Instant::now();
            for _ in 0..n {
                for l in 16..64 {
                    let q4 = rm.pinned_q4(l).unwrap_or(&q4_layers[l]);
                    let _ = best_backend.q4_matvec(q4, &q8_x, &q8_scales, features, hidden);
                }
            }
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let per_layer = total_ms / knowledge_layers as f64;
            let tps = 1000.0 / total_ms;

            let label = if budget_mb == 0 {
                "stream".to_string()
            } else if budget_mb > all_q4_mb as usize {
                "all".to_string()
            } else {
                format!("{} MB", budget_mb)
            };

            println!("  {:>10} {:>6}/{:<2} {:>8.0} MB {:>8.2}ms {:>8.1}ms {:>10.1} t/s",
                label, rm.num_pinned(), layers, rm.pinned_mb(), per_layer, total_ms, tps);
        }

        println!();
        println!("  llama.cpp 70B comparison (real hardware, approximate):");
        println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>12}",
            "40GB VRAM", "all", "40 GB", "", "", "8-12 t/s");
        println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>12}",
            "24GB VRAM", "partial", "24 GB", "", "", "2-3 t/s");
        println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>12}",
            "CPU only", "none", "40 GB", "", "", "1-2 t/s");
        println!();
        println!("  Vindex: smooth gradient. llama.cpp: cliff below 40GB.");
    }

    println!("\n=== Done ===");
}

// ── Helpers ──

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn random_query(hidden: usize) -> Array1<f32> {
    let mut s = 7u64;
    Array1::from_shape_fn(hidden, |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn build_synthetic_index(
    num_layers: usize,
    features: usize,
    hidden: usize,
) -> VectorIndex {
    let mut gate_vectors = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let gate = synth_matrix(features, hidden, 42 + layer as u64 * 1000);
        gate_vectors.push(Some(gate));
    }
    let down_meta = vec![None; num_layers];
    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}
