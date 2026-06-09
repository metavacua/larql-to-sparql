//! High-level DSv4 text generation entry point.
//!
//! [`dsv4_generate`] is the one-shot "give me N tokens from DSv4"
//! API that wires together:
//! - per-variant cache allocation via
//!   [`super::dsv4_kv_cache::DsV4LayerCache::for_variant`]
//! - cached prefill via
//!   [`super::dsv4_streaming_model_forward::dsv4_streaming_model_forward_cached`]
//! - cached single-token decode loop
//! - sampling via
//!   [`super::dsv4_sampling::dsv4_sample_next_token`]
//!
//! Use this as the production entry point for DSv4-Flash text
//! generation. The lower-level building blocks remain available for
//! callers that need finer control (per-step token inspection, custom
//! cache management, etc.).

use rand::Rng;

use larql_compute::ComputeBackend;
use larql_models::loading::gguf::GgufFile;

use super::dsv4_decode_loop::DecodeConfig;
use super::dsv4_full_loader::DsV4LoadError;
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_layer_variants::detect_layer_variant;
use super::dsv4_sampling::dsv4_sample_next_token;
use super::dsv4_storage_build::DsV4Hyperparams;
use super::dsv4_streaming_model_forward::dsv4_streaming_model_forward_cached;

/// Generate up to `decode_config.max_new_tokens` new tokens given a
/// prompt, running DSv4 in cached-decode mode for each step after the
/// prefill.
///
/// Returns the full token sequence: `prompt + generated`. The
/// generated portion has at most `max_new_tokens` entries (fewer if
/// the EOS token from `decode_config` was sampled).
///
/// `layer_indices` is the sequence of layers to run (typically
/// `0..n_layer` for a full DSv4-Flash forward; a prefix runs a
/// partial model — useful for testing).
///
/// Cache pre-allocation: each layer's cache is sized to
/// `prompt.len() + max_new_tokens`. The variant is detected from the
/// GGUF metadata, so callers don't need to know per-layer variant
/// shapes themselves.
pub fn dsv4_generate(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    layer_indices: &[usize],
    prompt: &[u32],
    decode_config: DecodeConfig,
    rng: &mut impl Rng,
    backend: Option<&dyn ComputeBackend>,
) -> Result<Vec<u32>, DsV4LoadError> {
    assert!(!prompt.is_empty(), "prompt must be non-empty");

    let max_seq_len = prompt.len() + decode_config.max_new_tokens;
    let mut layer_caches: Vec<DsV4LayerCache> = layer_indices
        .iter()
        .map(|&i| {
            let variant = detect_layer_variant(gguf, i, hp.head_dim);
            DsV4LayerCache::for_variant(hp, &variant, max_seq_len)
        })
        .collect();

    // 1. Prefill — runs the full prompt through the cached forward.
    let mut logits = dsv4_streaming_model_forward_cached(
        gguf,
        hp,
        head,
        prompt,
        layer_indices,
        0,
        Some(&mut layer_caches),
        backend,
    )?;
    let mut tokens = prompt.to_vec();

    // 2. Decode loop. Each iteration: sample from the last logit row,
    //    append the new token, then run one cached forward step on
    //    just that new token to produce logits for the *next* token.
    for i in 0..decode_config.max_new_tokens {
        let last_row_idx = logits.shape()[0] - 1;
        let last_row = logits.row(last_row_idx);
        let next = dsv4_sample_next_token(last_row, decode_config.sampling, rng);
        tokens.push(next);

        if decode_config.eos_token == Some(next) {
            break;
        }
        // Skip the trailing forward if this was the final iteration —
        // its logits would be discarded.
        if i + 1 >= decode_config.max_new_tokens {
            break;
        }

        let pos = tokens.len() - 1; // absolute position of the new token
        logits = dsv4_streaming_model_forward_cached(
            gguf,
            hp,
            head,
            &[next],
            layer_indices,
            pos,
            Some(&mut layer_caches),
            backend,
        )?;
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_head_storage::load_dsv4_head;
    use super::super::dsv4_sampling::SamplingConfig;
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Real-GGUF smoke: 16-token prompt + 2 greedy decode steps on
    /// layers 0..3 of DSv4-Flash. Verifies the entire one-shot
    /// generation API works end-to-end.
    ///
    /// Wall: ~500 s release (prefill ~110 s + 2 decode steps ~110 s
    /// each + head load + lm_head matmuls).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn generate_real_gguf_3_layer_greedy_smoke() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_prompt = 16;
        let prompt: Vec<u32> = (0..n_prompt)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let decode_config = DecodeConfig {
            max_new_tokens: 2,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);

        let tokens = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng,
            None,
        )
        .expect("generate");

        // prompt (16) + 2 generated = 18 tokens.
        assert_eq!(tokens.len(), n_prompt + 2);
        // Prompt prefix preserved.
        assert_eq!(&tokens[..n_prompt], &prompt[..]);
        // Generated tokens are valid IDs (< n_vocab).
        for &t in &tokens[n_prompt..] {
            assert!(
                (t as usize) < head.n_vocab,
                "generated token {t} out of vocab"
            );
        }
    }

    /// Real-GGUF sampled decode: same setup as the greedy smoke but
    /// with `SamplingConfig::default_sampling()` (T=0.7, top_k=40,
    /// top_p=0.95). Also exercises the seed-determinism contract:
    /// running twice with the same seed produces identical tokens.
    ///
    /// Wall: ~500 s release per call × 2 calls for the determinism
    /// check = ~1000 s. Tradeoff is worth it because the determinism
    /// check rules out subtle RNG/state-leakage bugs across the
    /// cached-decode pipeline.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk (~17 min wall)"]
    fn generate_real_gguf_sampled_seeded_is_deterministic() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_prompt = 16;
        let prompt: Vec<u32> = (0..n_prompt)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let decode_config = DecodeConfig {
            max_new_tokens: 2,
            eos_token: None,
            sampling: SamplingConfig::default_sampling(), // T=0.7, top_k=40, top_p=0.95
        };

        // Two independent runs with the same RNG seed.
        let mut rng_a = StdRng::seed_from_u64(0xc0ffee);
        let tokens_a = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng_a,
            None,
        )
        .expect("generate run A");

        let mut rng_b = StdRng::seed_from_u64(0xc0ffee);
        let tokens_b = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng_b,
            None,
        )
        .expect("generate run B");

        assert_eq!(
            tokens_a, tokens_b,
            "same seed must produce identical tokens"
        );
        assert_eq!(tokens_a.len(), n_prompt + 2);
        // Generated tokens are valid vocab IDs.
        for &t in &tokens_a[n_prompt..] {
            assert!((t as usize) < head.n_vocab);
        }
    }

    /// **GPU milestone test**: real-GGUF `dsv4_generate` with the
    /// default ComputeBackend (CUDA when the `cuda` feature is on +
    /// a CUDA device is present, CPU otherwise).
    ///
    /// Originally added in PR #340 to verify the lm_head GPU path
    /// end-to-end. As of the PR #339-#361 series, **every matmul-shaped
    /// hot loop on the DSv4 forward path routes through
    /// ComputeBackend** — meaning this test now exercises the full GPU
    /// coverage in a single end-to-end greedy-decode run:
    ///
    /// | Path                        | Started routing in |
    /// |-----------------------------|--------------------|
    /// | lm_head                     | #339               |
    /// | Cached HCA Q/KV/O (all 3 variants) | #341 / #342 / #343 |
    /// | Prefill HCA Q/KV/O (all 3 variants) | #345 / #346 / #347 |
    /// | grouped_o_proj (vectorized) | #344               |
    /// | Compressor prefill + cached step | #349 / #356  |
    /// | Indexer scores              | #350               |
    /// | Router gate_inp             | #351               |
    /// | mHC bookend + output head   | #352 / #353        |
    /// | Shared expert FFN           | #348               |
    /// | Routed MoE dispatch (scatter-gather + rayon) | #354 / #355 |
    /// | Masked attention (Q@K^T + softmax@V batched GEMMs) | #358 |
    /// | Sliding-window attention (delegates to masked) | #359 |
    /// | masked_attn softmax (rayon)  | #360               |
    /// | masked_attn second GEMM (matmul_gpu, no V^T copy) | #361 |
    ///
    /// Equivalence claim: greedy decode picks `argmax` over the logits.
    /// CPU and cuBLAS differ only in summation order (parallel reduction
    /// is not bit-identical), but for non-degenerate logit gaps the
    /// argmax is stable, so the **token sequence is identical** across
    /// backends. If a future change introduces a degenerate top-2 logit
    /// case at any of the predicted positions this test could become
    /// flaky — relax to `tokens.len() == expected` if that happens.
    ///
    /// Requires `--features cuda` AND the real GGUF on disk.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "Requires --features cuda + real DSv4-Flash GGUF + CUDA device"]
    fn generate_real_gguf_with_default_backend_matches_cpu() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_prompt = 16;
        let prompt: Vec<u32> = (0..n_prompt)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let decode_config = DecodeConfig {
            max_new_tokens: 1,
            eos_token: None,
            sampling: super::super::dsv4_sampling::SamplingConfig::greedy(),
        };

        // CPU-only baseline.
        let mut rng_cpu = StdRng::seed_from_u64(0);
        let tokens_cpu = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng_cpu,
            None,
        )
        .expect("CPU generate");

        // GPU-routed (default_backend on this build returns CudaBackend
        // when the cuda feature is enabled and a device is present;
        // otherwise it falls back to CpuBackend and this still passes,
        // just without GPU acceleration).
        let backend = larql_compute::default_backend();
        eprintln!(
            "[dsv4-gpu] running with backend: {} ({})",
            backend.name(),
            backend.device_info()
        );
        let mut rng_gpu = StdRng::seed_from_u64(0);
        let tokens_gpu = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng_gpu,
            Some(backend.as_ref()),
        )
        .expect("GPU-routed generate");

        // Greedy is stable across CPU/GPU when argmax has a clear winner.
        assert_eq!(
            tokens_cpu, tokens_gpu,
            "GPU and CPU greedy generation diverged"
        );
        assert_eq!(tokens_cpu.len(), n_prompt + 1);
    }

    /// **DSv4-Flash benchmark**: time prefill + decode separately for
    /// CPU vs CUDA and snapshot peak VRAM via `nvidia-smi`.
    ///
    /// This is the deliverable that answers "did the GPU push (#339-#362)
    /// pay off?" — it runs the same prompt + decode steps under both
    /// backends back-to-back and prints a side-by-side table. Output
    /// goes to stderr because Rust's test harness captures stdout by
    /// default; run with `cargo test -- --nocapture` to see it.
    ///
    /// Prefill and decode are timed independently (matches the qwen35
    /// bench format: `prefill_s / decode_s / ms/tok / tok/s`) by
    /// inlining the prefill + decode loop here instead of wrapping
    /// `dsv4_generate` whole-cloth. Production `dsv4_generate` is
    /// untouched.
    ///
    /// Knobs (env vars):
    /// - `LARQL_DSV4_BENCH_PROMPT_TOKENS` — prefill length (default 16)
    /// - `LARQL_DSV4_BENCH_DECODE_TOKENS` — decode steps (default 4)
    /// - `LARQL_DSV4_BENCH_LAYERS` — comma-separated layer indices
    ///   (default 0,1,2 to keep wall-time reasonable; pass e.g. 0,1,…,42
    ///   for the full 43-layer forward)
    /// - `LARQL_DSV4_BENCH_SKIP_CPU=1` — skip the CPU baseline (useful
    ///   when iterating on GPU and the CPU run is too slow)
    /// - `LARQL_DSV4_BENCH_VERBOSE=1` — print each decode step's wall
    ///   time individually. Useful for inspecting the warmup-decay
    ///   curve and sanity-checking whether the step==0 warmup
    ///   heuristic captures the real cold-start cost.
    ///
    /// VRAM is captured by shelling out to `nvidia-smi
    /// --query-gpu=memory.used --format=csv,noheader,nounits -i 0`
    /// after each backend's run completes. This is the same number
    /// `nvidia-smi` shows in interactive use. Returns 0 if nvidia-smi
    /// isn't on PATH (silent on non-CUDA hosts).
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "Requires --features cuda + real DSv4-Flash GGUF + CUDA device; LARQL_DSV4_BENCH_LAYERS env var controls layer count"]
    fn dsv4_bench_cpu_vs_cuda() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_prompt: usize = std::env::var("LARQL_DSV4_BENCH_PROMPT_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let n_decode: usize = std::env::var("LARQL_DSV4_BENCH_DECODE_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let layers: Vec<usize> = std::env::var("LARQL_DSV4_BENCH_LAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![0, 1, 2]);
        let skip_cpu = std::env::var("LARQL_DSV4_BENCH_SKIP_CPU").is_ok();

        let prompt: Vec<u32> = (0..n_prompt)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let decode_config = DecodeConfig {
            max_new_tokens: n_decode,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };

        eprintln!();
        eprintln!("=== DSv4-Flash benchmark ===");
        eprintln!(
            "  prompt_tokens={n_prompt}  decode_tokens={n_decode}  layers={:?}  (n={})",
            layers,
            layers.len()
        );

        // Drop unused-import warning for SamplingConfig now that the
        // bench uses argmax directly. We keep the use as it scopes
        // through the parent test module's `use` line.
        let _ = SamplingConfig::greedy;
        let _ = decode_config;

        let mut cpu: Option<BenchPhase> = None;
        if !skip_cpu {
            eprintln!("[cpu]   starting...");
            cpu = Some(run_phase(
                "cpu", &gguf, &hp, &head, &layers, &prompt, n_decode, None,
            ));
        } else {
            eprintln!("[cpu]   skipped (LARQL_DSV4_BENCH_SKIP_CPU set)");
        }

        // CudaBackend caches device buffers across calls, so a fresh
        // process would give a cleaner VRAM peak. This in-process
        // number is "VRAM in use right now" after the CUDA run —
        // for a single workload that doesn't free aggressively it's
        // close to the high-water mark.
        let backend = larql_compute::default_backend();
        eprintln!(
            "[cuda]  backend: {} ({})",
            backend.name(),
            backend.device_info()
        );
        let cuda = if std::env::var("LARQL_DSV4_BENCH_SKIP_CUDA").is_ok() {
            eprintln!("[cuda]  skipped (LARQL_DSV4_BENCH_SKIP_CUDA set)");
            None
        } else {
            eprintln!("[cuda]  starting...");
            Some(run_phase(
                "cuda",
                &gguf,
                &hp,
                &head,
                &layers,
                &prompt,
                n_decode,
                Some(backend.as_ref()),
            ))
        };

        // Resident-quant phase (dsv4-quant-residency): load the layers
        // once (Q4_K experts held in RAM) and decode against them — no
        // per-token streaming reload. CPU backend (FFN-on-CPU-resident,
        // the driving-goal config). Opt out on low-RAM hosts running
        // many layers via LARQL_DSV4_BENCH_SKIP_RESIDENT=1.
        let resident = if std::env::var("LARQL_DSV4_BENCH_SKIP_RESIDENT").is_ok() {
            eprintln!("[resident] skipped (LARQL_DSV4_BENCH_SKIP_RESIDENT set)");
            None
        } else {
            eprintln!("[resident] starting (CPU, Q4_K-resident experts)...");
            Some(run_phase_resident(
                "resident", &gguf, &hp, &head, &layers, &prompt, n_decode, None,
            ))
        };

        // P4 hybrid (dsv4-quant-residency): resident layers + CUDA
        // backend. The routed-MoE quant path runs CPU (it ignores
        // `backend`), so this routes only the f32 work — attention,
        // mHC, shared expert, router — to the GPU while the experts
        // (the bulk) stay CPU-quant. Because resident layers have
        // stable weight pointers across decode steps, the PR #368
        // device-resident f32 weight cache (enabled here) uploads each
        // attention weight once and reuses it — unlike streaming, where
        // per-step reload changes pointers and the cache always misses.
        let hybrid = if std::env::var("LARQL_DSV4_BENCH_SKIP_HYBRID").is_ok() {
            eprintln!("[hybrid] skipped (LARQL_DSV4_BENCH_SKIP_HYBRID set)");
            None
        } else {
            // Enable the device-resident weight cache (off by default).
            // Threshold covers attention + shared-expert f32 weights;
            // the routed experts never reach this path (CPU-quant).
            if std::env::var("LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS").is_err() {
                std::env::set_var("LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS", "67108864");
            }
            eprintln!(
                "[hybrid] starting (attn/mHC/shared→GPU + #368 weight cache, routed experts→CPU-quant)..."
            );
            Some(run_phase_resident(
                "hybrid",
                &gguf,
                &hp,
                &head,
                &layers,
                &prompt,
                n_decode,
                Some(backend.as_ref()),
            ))
        };

        // Summary table. `decode_s` is the steady-state wall (decode
        // steps 2..=n_decode); `warmup_s` is step 1, reported
        // separately. tok/s_dec is computed over steady-state only.
        let steady_steps = n_decode.saturating_sub(1).max(1);
        eprintln!();
        eprintln!("=== Summary ===");
        eprintln!(
            "                {:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "load_s", "prefill_s", "warmup_s", "decode_s", "ms/tok", "tok/s_dec", "vram_MiB"
        );
        let row = |tag: &str, p: &BenchPhase| {
            eprintln!(
                "  {tag:<11}{:>9.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.1}  {:>10.3}  {:>10}",
                p.load_s,
                p.prefill_s,
                p.warmup_s,
                p.decode_s,
                1000.0 * p.decode_s / steady_steps as f64,
                steady_steps as f64 / p.decode_s.max(1e-9),
                p.vram_mib
            );
        };
        if let Some(p) = &cpu {
            row("cpu", p);
        }
        if let Some(p) = &cuda {
            row("cuda", p);
        }
        if let Some(p) = &resident {
            row("resident", p);
        }
        if let Some(p) = &hybrid {
            row("hybrid", p);
        }
        if let (Some(p), Some(c)) = (&cpu, &cuda) {
            let prefill_speedup = p.prefill_s / c.prefill_s.max(1e-9);
            let decode_speedup = p.decode_s / c.decode_s.max(1e-9);
            eprintln!(
                "  cpu/cuda     {:>10.2}x prefill   {:>10.2}x decode (steady-state)",
                prefill_speedup, decode_speedup
            );
        }
        if let Some(r) = &resident {
            // The headline: resident steady-state decode vs the
            // streaming CPU baseline (per-token reload+dequant).
            if let Some(p) = &cpu {
                let decode_speedup = p.decode_s.max(1e-9) / r.decode_s.max(1e-9);
                eprintln!(
                    "  resident decode is {decode_speedup:>.1}x the cpu-streaming steady-state \
                     (load-once vs per-token reload)"
                );
            }
            // P4: hybrid (attn→GPU) vs resident (all-CPU) steady-state.
            if let Some(h) = &hybrid {
                let ratio = r.decode_s.max(1e-9) / h.decode_s.max(1e-9);
                eprintln!(
                    "  hybrid (attn→GPU) decode is {ratio:>.2}x the all-CPU resident steady-state"
                );
            }
        }
        eprintln!();

        // Sanity: at least one phase produced timings. Deeper
        // equivalence is the milestone test's job.
        let any_prefill = [
            cpu.as_ref(),
            cuda.as_ref(),
            resident.as_ref(),
            hybrid.as_ref(),
        ]
        .iter()
        .flatten()
        .any(|p| p.prefill_s > 0.0);
        assert!(any_prefill, "no phase ran");
    }

    /// One backend's measurement: prefill wall + warmup decode +
    /// steady-state decode + VRAM snapshot at the end of the run.
    /// `decode_s` is the **steady-state** wall (excludes the first
    /// decode step's warmup cost — first step pays device-resident
    /// weight load for CUDA / first-touch cache warming for CPU).
    /// `warmup_s` is reported separately for transparency.
    #[cfg(feature = "cuda")]
    struct BenchPhase {
        /// One-time weight load (resident phase only; 0 for streaming,
        /// which loads per layer per step inside the forward).
        load_s: f64,
        prefill_s: f64,
        warmup_s: f64,
        decode_s: f64,
        vram_mib: u64,
    }

    /// Run the same prefill + N-step decode loop that `dsv4_generate`
    /// runs, but time the prefill and decode phases independently
    /// and capture VRAM at the end. Mirrors `dsv4_generate`'s
    /// structure (cache allocation → prefill via
    /// `dsv4_streaming_model_forward_cached` → decode loop with
    /// argmax sampling) without modifying that production path.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_phase(
        tag: &str,
        gguf: &GgufFile,
        hp: &DsV4Hyperparams,
        head: &DsV4HeadStorage,
        layers: &[usize],
        prompt: &[u32],
        n_decode: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> BenchPhase {
        let max_seq_len = prompt.len() + n_decode;
        let mut layer_caches: Vec<DsV4LayerCache> = layers
            .iter()
            .map(|&i| {
                let variant = detect_layer_variant(gguf, i, hp.head_dim);
                DsV4LayerCache::for_variant(hp, &variant, max_seq_len)
            })
            .collect();

        // 1. Prefill: full prompt through cached forward.
        let t_prefill = std::time::Instant::now();
        let mut logits = dsv4_streaming_model_forward_cached(
            gguf,
            hp,
            head,
            prompt,
            layers,
            0,
            Some(&mut layer_caches),
            backend,
        )
        .expect("prefill");
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        eprintln!(
            "[{tag}]   prefill done in {prefill_s:.2}s ({} tokens)",
            prompt.len()
        );

        // 2. Decode loop with argmax (deterministic, cheap; the
        //    bench measures wall, not sampling behavior). The first
        //    step is timed separately as `warmup` — for CUDA it pays
        //    the lazy weight htod, for CPU it pays first-touch cache
        //    warming. `decode_s` reports steady-state across the
        //    remaining N-1 steps.
        let verbose = std::env::var("LARQL_DSV4_BENCH_VERBOSE").is_ok();
        let mut tokens = prompt.to_vec();
        let mut warmup_s = 0.0_f64;
        let mut decode_s = 0.0_f64;
        for step in 0..n_decode {
            let last = logits.shape()[0] - 1;
            let row = logits.row(last);
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .expect("non-empty logits");
            tokens.push(next);
            let pos = tokens.len() - 1;
            let t_step = std::time::Instant::now();
            logits = dsv4_streaming_model_forward_cached(
                gguf,
                hp,
                head,
                &[next],
                layers,
                pos,
                Some(&mut layer_caches),
                backend,
            )
            .expect("decode step");
            let dt = t_step.elapsed().as_secs_f64();
            if step == 0 {
                warmup_s = dt;
            } else {
                decode_s += dt;
            }
            if verbose {
                let kind = if step == 0 { "warmup" } else { "steady" };
                eprintln!(
                    "[{tag}]     step {step:>3} ({kind}): {:.2}s ({:.1} ms)",
                    dt,
                    dt * 1000.0
                );
            }
        }
        let steady_steps = n_decode.saturating_sub(1).max(1);
        let vram_mib = read_nvidia_smi_used_mib();
        eprintln!(
            "[{tag}]   decode done   warmup={warmup_s:.2}s  steady={decode_s:.2}s ({} steps, {:.1} ms/tok)  vram={vram_mib} MiB",
            steady_steps,
            1000.0 * decode_s / steady_steps as f64
        );
        BenchPhase {
            load_s: 0.0,
            prefill_s,
            warmup_s,
            decode_s,
            vram_mib,
        }
    }

    /// Resident-quant phase: load all `layers` once with routed experts
    /// held Q4_K-resident (`load_dsv4_resident_layer`), then run prefill
    /// + N-step decode against them via
    /// [`super::super::dsv4_streaming_model_forward::dsv4_resident_model_forward_cached`]
    /// — no per-token reload. This is the `dsv4-quant-residency` payoff:
    /// compare its steady-state ms/tok against the streaming phases'.
    /// `backend` is typically `None` (CPU resident — the driving-goal
    /// FFN-on-CPU config).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_phase_resident(
        tag: &str,
        gguf: &GgufFile,
        hp: &DsV4Hyperparams,
        head: &DsV4HeadStorage,
        layers: &[usize],
        prompt: &[u32],
        n_decode: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> BenchPhase {
        use super::super::dsv4_full_loader::load_dsv4_resident_layer;
        use super::super::dsv4_streaming_model_forward::dsv4_resident_model_forward_cached;

        // 0. Load every layer's storage ONCE (resident-quant experts).
        let t_load = std::time::Instant::now();
        let resident: Vec<_> = layers
            .iter()
            .map(|&i| load_dsv4_resident_layer(gguf, hp, i).expect("resident layer load"))
            .collect();
        let load_s = t_load.elapsed().as_secs_f64();
        eprintln!(
            "[{tag}]   resident load done in {load_s:.2}s ({} layers held in RAM)",
            layers.len()
        );

        let max_seq_len = prompt.len() + n_decode;
        let mut layer_caches: Vec<DsV4LayerCache> = resident
            .iter()
            .map(|(_, variant)| DsV4LayerCache::for_variant(hp, variant, max_seq_len))
            .collect();

        // 1. Prefill against the resident layers.
        let t_prefill = std::time::Instant::now();
        let mut logits = dsv4_resident_model_forward_cached(
            &resident,
            hp,
            head,
            prompt,
            0,
            Some(&mut layer_caches),
            backend,
        )
        .expect("resident prefill");
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        eprintln!(
            "[{tag}]   prefill done in {prefill_s:.2}s ({} tokens)",
            prompt.len()
        );

        // 2. Decode loop (argmax). step 0 = warmup; 1.. = steady-state.
        let verbose = std::env::var("LARQL_DSV4_BENCH_VERBOSE").is_ok();
        let mut tokens = prompt.to_vec();
        let mut warmup_s = 0.0_f64;
        let mut decode_s = 0.0_f64;
        for step in 0..n_decode {
            let last = logits.shape()[0] - 1;
            let next = logits
                .row(last)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .expect("non-empty logits");
            tokens.push(next);
            let pos = tokens.len() - 1;
            let t_step = std::time::Instant::now();
            logits = dsv4_resident_model_forward_cached(
                &resident,
                hp,
                head,
                &[next],
                pos,
                Some(&mut layer_caches),
                backend,
            )
            .expect("resident decode step");
            let dt = t_step.elapsed().as_secs_f64();
            if step == 0 {
                warmup_s = dt;
                // Reset the per-stage profiler so it accumulates only the
                // steady-state steps (LARQL_DSV4_PROFILE; no-op when unset).
                super::super::dsv4_profile::reset();
            } else {
                decode_s += dt;
            }
            if verbose {
                let kind = if step == 0 { "warmup" } else { "steady" };
                eprintln!("[{tag}]     step {step:>3} ({kind}): {:.3}s", dt);
            }
        }
        let steady_steps = n_decode.saturating_sub(1).max(1);
        let vram_mib = read_nvidia_smi_used_mib();
        eprintln!(
            "[{tag}]   decode done   warmup={warmup_s:.3}s  steady={decode_s:.3}s ({} steps, {:.1} ms/tok)  vram={vram_mib} MiB",
            steady_steps,
            1000.0 * decode_s / steady_steps as f64
        );
        let prof = super::super::dsv4_profile::report();
        if !prof.is_empty() {
            eprintln!("[{tag}] steady-state stage breakdown ({steady_steps} steps):\n{prof}");
        }
        BenchPhase {
            load_s,
            prefill_s,
            warmup_s,
            decode_s,
            vram_mib,
        }
    }

    /// Shell out to `nvidia-smi` to fetch current GPU-0 memory usage
    /// in MiB. Returns 0 if nvidia-smi isn't on PATH or the call
    /// fails — silent on non-CUDA hosts. Captures the bench's peak
    /// only approximately: it's "memory in use *right now*" when the
    /// run completes, which for a single-process CUDA workload that
    /// doesn't free aggressively is close to the high-water mark.
    #[cfg(feature = "cuda")]
    fn read_nvidia_smi_used_mib() -> u64 {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ])
            .output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0),
            _ => 0,
        }
    }
}
