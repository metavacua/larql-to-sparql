//! Fine-grained per-section accumulators for Qwen3.6 decode.
//!
//! Each section has a static `AtomicU64` counting cumulative
//! nanoseconds across every layer × every token in a forward call.
//! Sections are tagged by where they live in the per-layer flow
//! (DeltaNet block, FFN, full-attn block, top-level).
//!
//! Enable with `LARQL_QWEN35_FINE_PROFILE=1`. At the end of each
//! `qwen35_forward_step` the totals are printed and reset so a
//! multi-token bench gets one breakdown line per token.
//!
//! The overhead per `time_section!` is one env-var check (cached at
//! `enabled()` lookup, but each call still touches the static) plus
//! the `Instant::now` pair when enabled. Negligible compared to the
//! sections themselves (kernel launches, matvecs, etc.); the
//! `enabled` short-circuit branch is the common case.

use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! sections {
    ($($name:ident),* $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)*

        pub fn all() -> Vec<(&'static str, u64)> {
            vec![$((stringify!($name), $name.load(Ordering::Relaxed))),*]
        }

        pub fn reset() {
            $($name.store(0, Ordering::Relaxed);)*
        }
    }
}

sections!(
    // top-level forward
    EMBED,
    RESIDUAL_ADDS,
    FINAL_NORM,
    LM_HEAD,
    // DeltaNet linear-attn block (48 layers in Qwen3.6 27B)
    DN_RMS_NORM_X,       // pre-mixer RMSNorm
    DN_QKV_GATE_PAIR,    // paired Q4_K matvec
    DN_BETA_ALPHA_DOTS,  // CPU ndarray dot for ssm_beta + ssm_alpha
    DN_SIGMOID_SOFTPLUS, // beta + log_g compute
    DN_CONV1D,           // GPU conv1d step
    DN_SILU_QKV_CONV,    // CPU silu over qkv_conv
    DN_SPLIT_RESHAPE,    // host slice/reshape Q/K/V
    DN_L2_NORM,          // GPU L2 norm Q + K
    DN_RECURRENCE,       // GPU deltanet step
    DN_O_RESHAPE,        // host transpose o.t().to_owned()
    DN_RMS_NORM_HEADS,   // GPU rms_norm_heads
    DN_SILU_Z,           // CPU silu(z) * o_flat
    DN_SSM_OUT,          // ssm_out matvec
    // FFN (64 layers)
    FFN_RMS_NORM,
    FFN_GATE_UP_PAIR,
    FFN_SILU_LOOP,
    FFN_DOWN,
    // Full-attn block (16 layers) — coarse for now
    ATTN_BLOCK,
);

#[inline]
pub fn enabled() -> bool {
    // Thread-local cache so the env lookup doesn't hit the OS on
    // every section.
    thread_local! {
        static CACHED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    CACHED.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        let v = std::env::var("LARQL_QWEN35_FINE_PROFILE").is_ok();
        c.set(Some(v));
        v
    })
}

/// Wrap an expression with timing. Compiles to the bare expression
/// when the profile env var is unset.
#[macro_export]
macro_rules! time_section {
    ($section:expr, $body:expr) => {{
        if $crate::attention::fine_profile::enabled() {
            let t0 = std::time::Instant::now();
            let r = $body;
            $section.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            r
        } else {
            $body
        }
    }};
}

pub fn dump_and_reset() {
    let totals = all();
    let grand_total_ns: u64 = totals.iter().map(|(_, n)| n).sum();
    eprintln!(
        "FINE_PROFILE (total {:.3} ms):",
        grand_total_ns as f64 / 1e6
    );
    let mut sorted = totals;
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, ns) in &sorted {
        if *ns == 0 {
            continue;
        }
        let ms = *ns as f64 / 1e6;
        let pct = (*ns as f64) / (grand_total_ns.max(1) as f64) * 100.0;
        eprintln!("  {name:22} {ms:8.3} ms  ({pct:5.2} %)");
    }
    reset();
}
