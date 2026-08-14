//! KV-cache engine implementations.
//!
//! Each engine implements [`crate::KvEngine`] (which lives in
//! `larql-inference::kv_engine` and is re-exported here) — a common
//! interface for prefill + autoregressive decode that manages inference
//! state differently:
//!
//! ## Engine ladder (Gemma 3 4B @ 370K tokens)
//!
//! | Engine | Mechanism | Memory | Accuracy |
//! |---|---|---|---|
//! | [`standard`] | Production K/V tensor cache (default) | O(seq) f32 K/V | exact — the reference |
//! | [`no_cache`] | Full re-forward per step | O(seq) token IDs | exact — correctness fallback |
//! | [`markov_residual`] | Residual-stream replacement | ~171 MB | exact (KL=0.0) under contract |
//! | [`windowed_checkpoint`] | Per-window K/V checkpoints | ~193 MB | exact within window |
//! | [`turbo_quant`] | WHT + Lloyd-Max 3/4-bit codec | ~12.7 GB | per-row cos≈0.9954 (4-bit, 2026-07-30) |
//! | [`apollo`] | Boundary store + residual injection | ~11 MB | task accuracy |
//!
//! ## Selecting an engine
//!
//! ```text
//! larql bench gemma3-4b-q4k --engine standard
//! larql bench gemma3-4b-q4k --engine standard:window=1024   # keeps the fused path
//! larql bench gemma3-4b-q4k --engine no-cache
//! larql bench gemma3-4b-q4k --engine markov-rs:window=512
//! larql bench gemma3-4b-q4k --engine windowed-checkpoint:window=256
//! larql bench gemma3-4b-q4k --engine turbo-quant:bits=3
//! larql bench gemma3-4b-q4k --engine apollo:layer=25,coef=8.0
//! ```
//!
//! See [`crate::EngineKind::from_name`] for the full parameter syntax.
//!
//! ## Architecture notes
//!
//! - **Coarse (fused) path** (`prefill_quant` / `decode_step_quant`): taken
//!   when a Q4K VectorIndex is present and the backend accepts the engine's
//!   window. Metal routes through its `decode_token` full pipeline; CPU
//!   through `cached_prefill_q4k` / `cached_decode_step_q4k`.
//!
//!   The engine's own state policy is NOT engaged here — the K/V lives in the
//!   backend behind a sentinel handle (Metal) or a whole-model handle (CPU),
//!   so `standard`, `markov-rs`, `markov-rs-codec` and `boundary-per-layer`
//!   execute the same kernels and measure within ~0.5% of each other.
//!   **Ranking those four against one another on this path measures nothing.**
//!
//!   The engine rows do beat the reference `larql-metal` row, but the reason
//!   is the KNN lm_head (~1.9ms) replacing a full vocab matmul (~6.6ms), not
//!   the K/V mechanism. An earlier note credited the engines with a ~93-95 vs
//!   76 tok/s win; that gap was inflated because the bench timed only the
//!   forward and left lm_head outside the timer. Both are inside it now, and
//!   the row note carries the `fwd=` / `head=` split.
//!
//! - **Windowed engines keep the coarse path** (since 2026-08). A window is
//!   requested through `coarse_prefill_windowed` / `coarse_decode_step_windowed`,
//!   which fail closed: a backend that cannot bound BOTH attention and K/V
//!   answers `None` and the engine falls back to per-layer. Both `CpuBackend`
//!   and `MetalBackend` implement them.
//!
//!   Each bounds attention and memory by different means, because neither
//!   mechanism does both jobs. CPU trims the cache to `w - 1` before each step
//!   (cheap — the cache is host arrays). Metal clamps the attention span every
//!   step via the layer window, and compacts its K/V buffers only once
//!   occupancy reaches 2x the window, so the memmove is O(1) amortised rather
//!   than O(window) per token; the surplus resident rows are never read
//!   because the kernel attends `[T - window, T)`.
//!
//!   Both decline a prompt LONGER than the window: the fused prefill has no
//!   per-query-position masking, so accepting would attend the whole prompt
//!   while the engine advertises a bound. That case still takes per-layer.
//!
//!   Measured on Gemma 3 4B Q4K, Metal, 80 steps at `window=8`: 11.61ms /
//!   2.4MB against 12.06ms / 23.6MB unwindowed. Before this, the same config
//!   cost 115.44ms.
//!
//! - **Per-layer path** (a prompt longer than the window, or an arch that
//!   declines coarse): `MetalBackend`'s `KvDispatch` impl delegates every
//!   per-layer method (`attention_step`, `attention_step_windowed`,
//!   `append_kv`, `clip_kv`, …) to `CpuBackend`. An engine on this path runs
//!   CPU attention **and** CPU FFN while the bench labels the row
//!   `[metal (GPU)]` — worth ~9x on Gemma 3 4B. The row's dispatch note says
//!   `[per-layer->host]` when that is what happened.
//!
//! - **Cross-backend parity**: Metal and CPU agree to ~5e-7 relative L2 on
//!   prefill, and per-step decode differs by a stable ~3e-3 that does not
//!   compound (two Q4K kernels rounding differently). Pinned by
//!   `tests/gpu_engine_parity`. A divergence on the Gemma-3 arch that was
//!   open through 2026-08 turned out to be a test fixture declaring
//!   QK-norm without supplying the weights — real checkpoints always
//!   carry them, so nothing shipped was affected.
//!
//! - **CPU fallback**: when Metal is unavailable, engines fall back to a CPU
//!   path using dequantised attention tensors (lazily inserted into the
//!   engine-owned `dequant_scratch`) and `WalkFfn` for Q4K FFN.
//!
//! - **Apollo compressed path**: when the store has boundary residuals captured
//!   at `crystal_layer` (default 30), `forward_from_layer` runs only
//!   `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per step.

pub mod apollo;
pub mod boundary_kv;
pub mod boundary_per_layer;
mod layer_ffn;
pub mod markov_residual;
pub mod markov_residual_codec;
pub mod no_cache;
pub mod no_expert_route;
pub mod semantic_promotion;
pub mod standard;
pub mod turbo_quant;
pub mod windowed_checkpoint;

pub(crate) use layer_ffn::{apply_ple_and_layer_scalar, layer_ffn_or_moe};
pub(crate) use no_expert_route::refuse_if_moe;

/// Whether W10 mask cascade is active.
///
/// Default: **on** (since 2026-05-21). The mask is bit-identical to
/// Full under each opted-in engine's exact_logits contract (proven by
/// `examples/w10_parity_gate.rs`) and closes the ~13% gap to
/// `standard`'s fused-kernel ceiling on Metal.
///
/// Opt out with `LARQL_W10_DISABLE=1` (debug instrument for
/// bisecting masked-backend regressions). `LARQL_W10_HONLY=1` is
/// also accepted for backwards compat with older bench scripts —
/// it's now a no-op since the cascade is on by default.
///
/// Used by the per-engine `dispatch.rs` modules
/// (markov_residual, markov_residual_codec, windowed_checkpoint,
/// boundary_per_layer). Engines that treat K/V as canonical state
/// (turbo_quant) don't call this — their dispatch path stays on
/// Full mask regardless.
///
/// Tests inject a value through `set_w10_disabled_override` (per-thread)
/// rather than mutating the process env, so they don't race other
/// parallel tests that also call this helper.
pub(crate) fn w10_enabled() -> bool {
    let overridden = W10_DISABLED_OVERRIDE.with(|o| *o.borrow());
    match overridden {
        Some(disabled) => !disabled,
        None => std::env::var("LARQL_W10_DISABLE").as_deref() != Ok("1"),
    }
}

std::thread_local! {
    /// Per-thread override for [`w10_enabled`]. `Some(true)` simulates
    /// `LARQL_W10_DISABLE=1` (cascade off); `Some(false)` simulates the
    /// var unset (cascade on); `None` falls through to the real env.
    /// Test-only escape hatch — production callers leave it `None`.
    static W10_DISABLED_OVERRIDE: std::cell::RefCell<Option<bool>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn set_w10_disabled_override(disabled: Option<bool>) {
    W10_DISABLED_OVERRIDE.with(|o| *o.borrow_mut() = disabled);
}

/// Test-only RAII helper to drive the Q4K decode fast-path flags via
/// `larql_compute::options`' **thread-local** override (NOT `std::env::set_var`,
/// which is thread-unsafe vs the concurrent `getenv` every parallel decode test
/// does — that race SIGSEGVs libc). Each entry sets one `ENV_*` flag on the
/// current thread; everything is cleared on drop. Because the override is
/// per-thread, these tests need no cross-test serialization.
#[cfg(test)]
pub(crate) struct Q4kFlagGuard;

#[cfg(test)]
impl Q4kFlagGuard {
    /// Override the given `(ENV_* name, on)` flags on this thread for the
    /// guard's lifetime. Also clears the `LARQL_MARKOV_*` thread-local overrides
    /// on drop (the A/B tests set `LARQL_MARKOV_INPLACE_KV` there).
    pub(crate) fn set(flags: &[(&'static str, bool)]) -> Self {
        for &(name, on) in flags {
            larql_compute::options::set_fast_path_override(name, on);
        }
        Q4kFlagGuard
    }
}

#[cfg(test)]
impl Drop for Q4kFlagGuard {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
        crate::engines::markov_residual::compute::clear_markov_env_overrides();
    }
}

#[cfg(test)]
mod resident_identity_tests {
    //! Structural-gap pin (2026-06-13): every pluggable engine overrides
    //! `decode_step_resident` (or forwards it, for wrappers) so the vindex
    //! reaches the attention step's Q4K-direct route. With the flags OFF —
    //! the default test environment — the resident path must be
    //! BIT-IDENTICAL to the plain path for every engine; a fork here means
    //! an engine's resident override drifted from its decode_step.

    use crate::EngineKind;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    #[test]
    fn every_engine_decode_step_resident_matches_decode_step_flag_off() {
        // The Q4K decode fast path is on by default now; this pin asserts the
        // flags-OFF f32 identity (resident must equal plain when the resident
        // route is *not* taking the Q4K-direct branch), so disable the stages
        // that change the resident hidden state — via the thread-local override
        // (no `set_var`, so no segfault race with parallel decode tests).
        let _flags_off = super::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, false),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
            (larql_compute::options::ENV_Q4K_DIRECT_FFN, false),
            (larql_compute::options::ENV_Q4K_LM_HEAD, false),
        ]);

        // Concrete specs (parameterised kinds need real params). Excluded:
        // apollo (bench-only, full re-forward by design; resident default =
        // forward to decode_step is the documented intent) and boundary-kv
        // (frame emission needs an archive sink; its resident forwarding is
        // a thin wrapper over StandardEngine — pinned here — plus its own
        // frame tests).
        let specs = [
            "standard",
            "standard:window=4",
            "no-cache",
            "markov-rs",
            "markov-rs-codec",
            "turbo-quant",
            "unlimited-context:window=4",
        ];
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let prompt = [0u32, 1, 2];

        let mut tested = 0usize;
        for spec in specs {
            let Some(kind) = EngineKind::from_name(spec) else {
                panic!("spec {spec:?} no longer parses — update this pin");
            };
            let mut plain = kind.clone().build(larql_inference::cpu_engine_backend());
            let mut resident = kind.build(larql_inference::cpu_engine_backend());

            let h_plain = plain
                .prefill(&weights, &ffn, &prompt)
                .unwrap_or_else(|e| panic!("{spec}: prefill failed: {e:?}"));
            let h_res = resident
                .prefill_resident(&weights, &ffn, &index, &prompt)
                .unwrap_or_else(|e| panic!("{spec}: prefill_resident failed: {e:?}"));
            assert_eq!(
                h_plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                h_res.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{spec}: prefill outputs diverged with flags off"
            );

            // Several decode steps so the growing context exercises engines'
            // per-step caches deeply — in particular markov-rs/-codec's new
            // hot-K/V cache (its debug parity assert fires every cached step,
            // which is most of these).
            for (step, tok) in (3u32..=12).enumerate() {
                let d_plain = plain
                    .decode_step(&weights, &ffn, tok)
                    .unwrap_or_else(|e| panic!("{spec}: decode_step failed: {e:?}"));
                let d_res = resident
                    .decode_step_resident(&weights, &ffn, &index, tok)
                    .unwrap_or_else(|e| panic!("{spec}: decode_step_resident failed: {e:?}"));
                assert_eq!(
                    d_plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    d_res.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{spec}: decode outputs diverged at step {step} with flags off"
                );
            }
            tested += 1;
        }
        assert!(
            tested >= 7,
            "engine coverage shrank ({tested} < 7) — no silent caps"
        );
    }
}
