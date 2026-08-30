//! KV-A/KV-B — the production attention kernels, benched against span.
//!
//! Every attention kernel in this crate was `status: "inventory"` in
//! `diag/shader_bench` — listed, never measured — while the end-to-end
//! evidence pointed straight at them: gpt-oss decodes in ~11.5 ms/token at
//! short context and ~16.1 ms at `-n 2048`.
//!
//! The end-to-end instrument cannot resolve it. Decode runs one command
//! buffer per token, so `GPUStartTime/EndTime` gives a token window rather
//! than a per-layer one; `LARQL_PROFILE_SPLIT=1` disables the merged
//! command buffer and misattributes; `LARQL_PROFILE_DECODE=1` inflates ~2x
//! at high step counts; and the machine drifts after ~3 heavy runs. So this
//! drives the kernels directly, at the shape production uses.
//!
//! Methodology from `bench_moe_expert_format_split` — the form that
//! predicted the MXFP4 expert-read delta within ~0.1 ms of end-to-end:
//! `BATCH` dispatches in ONE command buffer so the number is the kernel and
//! not submit/complete latency, each with its own cache so the batch reads
//! cold rather than re-reading one hot working set.
//!
//! ## What it measures
//!
//! - **production** `kv_attention` / `kv_attention_long` at their production
//!   geometry (`head_dim` threads).
//! - **phases 1-2 alone** via `kv_attention_phase12_only`, attributing the
//!   remainder to phase 3's weighted-V accumulation.
//! - **KV-B1 `kv_attention_seqpar`** at 2/4/8 sequence slices. The slice
//!   count is swept rather than assumed: past some point extra threads cost
//!   occupancy and the partial reduction wins back the benefit, and only
//!   measurement locates that.
//!
//! Run: `cargo run --release -p larql-compute-metal --example bench_attention_span`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_attention_span requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::ops::kv_cache::{
        LayerKVCache, LONG_ATTENTION_SPAN, SHORT_ATTENTION_SPAN,
    };
    use larql_compute_metal::MetalBackend;

    // gpt-oss-20b attention geometry, from the vindex's own config.
    const NUM_Q_HEADS: usize = 64;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;
    /// One dispatch per full-attention layer — the 12 that keep growing past
    /// the sliding window, i.e. the term this bench exists to explain.
    const BATCH: usize = 12;
    const WARMUP: usize = 8;
    const ITERS: usize = 40;
    /// Sequence slices for the KV-B1 kernel. 16 slices = 1024 threads is
    /// the ceiling — `tg_partial` holds `n_slices * head_dim`, and 1024 is
    /// also Metal's threadgroup limit. The first sweep stopped at 8 and
    /// found the optimum sitting ON that bound at every span, which is a
    /// measurement artefact rather than a result; hence 16.
    const SLICES: &[usize] = &[2, 4, 8, 12, 16];

    const SPANS: &[u32] = &[128, 256, 512, 768, 1024, 1025, 1536, 2048];

    let Some(metal) = MetalBackend::new() else {
        eprintln!("no Metal device");
        return;
    };
    let bufs = metal.bufs();

    // A distinct cache per batched dispatch: 12 x 4096 rows x 8 kv heads x
    // 64 dims x 4 B x 2 (K+V) = ~200 MB, so the batch reads a cold working
    // set rather than one cache already resident from dispatch 1.
    let mut caches: Vec<LayerKVCache> = (0..BATCH)
        .map(|_| LayerKVCache::new(bufs, LONG_ATTENTION_SPAN, NUM_KV_HEADS, HEAD_DIM))
        .collect();
    let q = bufs.output((NUM_Q_HEADS * HEAD_DIM * 4) as u64);
    let out = bufs.output((NUM_Q_HEADS * HEAD_DIM * 4) as u64);
    let sinks = bufs.output((NUM_Q_HEADS * 4) as u64);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    #[derive(Clone, Copy)]
    enum Arm {
        Production,
        Phase12Only,
        SeqPar(usize),
    }

    println!("KV-A/B: production attention kernels vs span");
    println!(
        "  geometry  q_heads={NUM_Q_HEADS} kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} (gpt-oss-20b)"
    );
    println!(
        "  batched   {BATCH} dispatches/cmdbuf, one cache each, warmup {WARMUP} iters {ITERS}"
    );
    println!();
    println!(
        "  {:>5}  {:>9} {:>7}  {:>8} {:>8} {:>7}   KV-B1 seqpar (slices)",
        "span", "prod us", "GB/s", "p1-2 us", "p3 us", "p3 %"
    );
    println!("  {}", "-".repeat(100));

    for &span in SPANS {
        for c in caches.iter_mut() {
            c.current_len = (span - 1) as usize;
            c.abs_position = (span - 1) as usize;
        }

        let measure = |arm: Arm| -> f64 {
            let (pipeline, threads) = match arm {
                Arm::Production => (
                    if span > SHORT_ATTENTION_SPAN {
                        &metal.attention.kv_attend_long_pipeline
                    } else {
                        &metal.attention.kv_attend_pipeline
                    },
                    HEAD_DIM as u64,
                ),
                Arm::Phase12Only => (
                    &metal.attention.kv_attend_phase12_only_pipeline,
                    HEAD_DIM as u64,
                ),
                Arm::SeqPar(n) => (
                    if span > SHORT_ATTENTION_SPAN {
                        &metal.attention.kv_attend_seqpar_long_pipeline
                    } else {
                        &metal.attention.kv_attend_seqpar_pipeline
                    },
                    (n * HEAD_DIM) as u64,
                ),
            };
            let mut times: Vec<f64> = Vec::with_capacity(ITERS);
            for i in 0..WARMUP + ITERS {
                let t = std::time::Instant::now();
                let cmd = metal.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for c in caches.iter() {
                    let t_val = (c.current_len + 1) as u32;
                    let hd = HEAD_DIM as u32;
                    let nq = NUM_Q_HEADS as u32;
                    let nkv = NUM_KV_HEADS as u32;
                    let win = 0u32;
                    let has_sinks = 0u32;
                    let softcap = 0.0f32;
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&q), 0);
                    enc.set_buffer(1, Some(&c.k_cache), 0);
                    enc.set_buffer(2, Some(&c.v_cache), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &nq as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &nkv as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &win as *const u32 as *const std::ffi::c_void);
                    enc.set_buffer(10, Some(&sinks), 0);
                    enc.set_bytes(11, 4, &has_sinks as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(12, 4, &softcap as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(NUM_Q_HEADS as u64, 1, 1),
                        metal::MTLSize::new(threads, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= WARMUP {
                    times.push(ms / BATCH as f64);
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[times.len() / 2] * 1000.0 // median us, robust to a stray hit
        };

        let prod = measure(Arm::Production);
        // The phase-1-2 arm carries a 1024-entry tg_scores, so it is only
        // valid where the span fits it.
        let p12 = (span <= SHORT_ATTENTION_SPAN).then(|| measure(Arm::Phase12Only));
        let bytes = span as usize * NUM_KV_HEADS * HEAD_DIM * 4 * 2;
        let gbs = bytes as f64 / (prod / 1e6) / 1e9;

        print!("  {span:>5}  {prod:>9.2} {gbs:>7.1}");
        match p12 {
            Some(p) => print!(
                "  {:>8.2} {:>8.2} {:>6.1}%",
                p,
                prod - p,
                100.0 * (prod - p) / prod
            ),
            None => print!("  {:>8} {:>8} {:>7}", "-", "-", "-"),
        }
        for &n in SLICES {
            let t = measure(Arm::SeqPar(n));
            print!("   {n}x {t:>7.2}us {:>4.2}x", prod / t);
        }
        // Same bracket as the fused table below: the baseline is measured
        // again after the slice arms, and a row whose two readings
        // disagree is not evidence about slice count. Added after a
        // whole e2e sweep had to be discarded for exactly this.
        let prod_close = measure(Arm::Production);
        print!("   drift {:>+5.1}%", 100.0 * (prod_close - prod) / prod);
        println!();
    }

    // ── The kernel decode actually runs ────────────────────────────────
    //
    // `kv_append_attend_fused` serves every span <= SHORT_ATTENTION_SPAN
    // (see `decode::encode_attn`), which is every sliding-window layer at
    // every depth plus every full-attention layer to 1024 — i.e. the
    // common case. The table above measures the fallback. This one decides
    // the shipping policy, so it carries its own bracket: the baseline is
    // measured again after the slice arms, and a row whose two baseline
    // readings disagree is not evidence about slice count.
    let new_k = bufs.transient_from_f32(&vec![0.01f32; NUM_KV_HEADS * HEAD_DIM]);
    let new_v = bufs.transient_from_f32(&vec![0.02f32; NUM_KV_HEADS * HEAD_DIM]);

    println!();
    println!("KV-B1 policy: kv_append_attend_fused (the DEFAULT decode path, span <= 1024)");
    println!("  drift = second baseline vs first; a row above ~5% is not usable");
    println!();
    println!(
        "  {:>5}  {:>9}  {:>26}  {:>8}",
        "span", "base us", "seqpar (slices)", "drift"
    );
    println!("  {}", "-".repeat(78));

    // Spans stop at SHORT_ATTENTION_SPAN: past it this kernel's
    // tg_scores[1024] cannot hold the span and decode takes the unfused
    // path measured above.
    const FUSED_SPANS: &[u32] = &[64, 128, 192, 256, 384, 512, 768, 1024];

    for &span in FUSED_SPANS {
        for c in caches.iter_mut() {
            c.current_len = (span - 1) as usize;
            c.abs_position = (span - 1) as usize;
        }

        let measure_fused = |slices: usize| -> f64 {
            let pipeline = if slices == 0 {
                &metal.attention.kv_append_attend_fused_pipeline
            } else {
                &metal.attention.kv_append_attend_fused_seqpar_pipeline
            };
            let threads = if slices == 0 {
                HEAD_DIM as u64
            } else {
                (slices * HEAD_DIM) as u64
            };
            let mut times: Vec<f64> = Vec::with_capacity(ITERS);
            for i in 0..WARMUP + ITERS {
                let t = std::time::Instant::now();
                let cmd = metal.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for c in caches.iter() {
                    let t_val = (c.current_len + 1) as u32;
                    let hd = HEAD_DIM as u32;
                    let nq = NUM_Q_HEADS as u32;
                    let nkv = NUM_KV_HEADS as u32;
                    let win = 0u32;
                    let has_sinks = 1u32; // gpt-oss runs sinks on every layer
                    let softcap = 0.0f32;
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&q), 0);
                    enc.set_buffer(1, Some(&c.k_cache), 0);
                    enc.set_buffer(2, Some(&c.v_cache), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &nq as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &nkv as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &win as *const u32 as *const std::ffi::c_void);
                    enc.set_buffer(10, Some(&new_k), 0);
                    enc.set_buffer(11, Some(&new_v), 0);
                    enc.set_buffer(12, Some(&sinks), 0);
                    enc.set_bytes(13, 4, &has_sinks as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(14, 4, &softcap as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(NUM_Q_HEADS as u64, 1, 1),
                        metal::MTLSize::new(threads, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= WARMUP {
                    times.push(ms / BATCH as f64);
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[times.len() / 2] * 1000.0
        };

        let base_open = measure_fused(0);
        print!("  {span:>5}  {base_open:>9.2}");
        let mut cells = String::new();
        for &n in SLICES {
            let t = measure_fused(n);
            cells.push_str(&format!("  {n}x {:>5.2}x", base_open / t));
        }
        let base_close = measure_fused(0);
        let drift = 100.0 * (base_close - base_open) / base_open;
        println!("  {cells}  {drift:>+6.1}%");
    }
}
