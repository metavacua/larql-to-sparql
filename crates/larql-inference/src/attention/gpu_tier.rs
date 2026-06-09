//! Per-class GPU residency knobs.
//!
//! Phase E.7. larql's value proposition is *minimizing* VRAM use by
//! pushing as much work as possible to CPU + host RAM, with only the
//! pieces that demonstrably need GPU acceleration (attention compute,
//! KV cache, recurrent state) staying device-resident. The default
//! `LARQL_QWEN35_GPU=1` mode pushes everything to GPU for max
//! throughput; this module lets the caller selectively pull specific
//! tensor classes back to CPU.
//!
//! Each `GpuClass` corresponds to a dispatch site in the Qwen3.6
//! forward path. When the corresponding env var is set, the dispatch
//! at that site treats the backend as if it weren't attached and
//! falls back to the existing CPU path.
//!
//! Env var convention: `LARQL_QWEN35_GPU_NO_<CLASS>=1` disables the
//! GPU path for that class. Unset / `0` / empty keeps it on GPU when
//! the master `LARQL_QWEN35_GPU=1` toggle is on.
//!
//! Concrete tiers (combinations) you might want for the bench curve:
//!
//! | Tier                | Disabled classes                                   | VRAM rough |
//! |---|---|---:|
//! | full (default)      | none                                               | ~13 GiB    |
//! | no_ffn              | FFN                                                | ~6 GiB     |
//! | attn_only           | FFN + DN projections + DN recurrence + LM head     | ~1-2 GiB   |
//! | cpu_baseline        | everything (or just don't set LARQL_QWEN35_GPU)     | 0          |
//!
//! Thread-local cached env lookups keep the per-call overhead at one
//! atomic load.

/// Tensor class for per-site GPU residency dispatch.
#[derive(Clone, Copy, Debug)]
pub enum GpuClass {
    /// Final output projection `lm_head @ x_final → logits`.
    LmHead,
    /// Full-attention block's Q/K/V/O projection matvecs.
    AttnProj,
    /// Full-attention block's softmax + score / weighted-V scan
    /// (`gqa_decode_step`). Separate from `AttnProj` because the scan
    /// is the part that *uses* the KV cache — disabling it forces the
    /// CPU autoregressive path even when the Q/K/V projections
    /// themselves are GPU-side.
    AttnDecode,
    /// DeltaNet block's projection matvecs (attn_qkv, attn_gate, ssm_out).
    DnProj,
    /// DeltaNet recurrence chain (conv1d, L2 norm, recurrence kernel,
    /// rms_norm_heads). The recurrent state buffer follows this
    /// setting — when disabled the recurrence runs on CPU and the
    /// state lives on host.
    DnRecurrence,
    /// SwiGLU FFN gate / up / down projection matvecs (and the
    /// device-resident fused chain when applicable).
    Ffn,
}

impl GpuClass {
    fn env_var(self) -> &'static str {
        match self {
            Self::LmHead => "LARQL_QWEN35_GPU_NO_LM_HEAD",
            Self::AttnProj => "LARQL_QWEN35_GPU_NO_ATTN_PROJ",
            Self::AttnDecode => "LARQL_QWEN35_GPU_NO_ATTN_DECODE",
            Self::DnProj => "LARQL_QWEN35_GPU_NO_DN_PROJ",
            Self::DnRecurrence => "LARQL_QWEN35_GPU_NO_DN_RECURRENCE",
            Self::Ffn => "LARQL_QWEN35_GPU_NO_FFN",
        }
    }
}

/// Returns `true` when this tensor class should run on the GPU
/// backend (the env var is *not* set). Default = `true`.
///
/// Cheap to call in hot paths: each tensor-class env var lookup is
/// memoised in a thread-local on first use.
#[inline]
pub fn is_enabled(class: GpuClass) -> bool {
    use std::cell::Cell;
    thread_local! {
        static LM_HEAD: Cell<Option<bool>> = const { Cell::new(None) };
        static ATTN_PROJ: Cell<Option<bool>> = const { Cell::new(None) };
        static ATTN_DECODE: Cell<Option<bool>> = const { Cell::new(None) };
        static DN_PROJ: Cell<Option<bool>> = const { Cell::new(None) };
        static DN_RECURRENCE: Cell<Option<bool>> = const { Cell::new(None) };
        static FFN: Cell<Option<bool>> = const { Cell::new(None) };
    }
    let cell = match class {
        GpuClass::LmHead => &LM_HEAD,
        GpuClass::AttnProj => &ATTN_PROJ,
        GpuClass::AttnDecode => &ATTN_DECODE,
        GpuClass::DnProj => &DN_PROJ,
        GpuClass::DnRecurrence => &DN_RECURRENCE,
        GpuClass::Ffn => &FFN,
    };
    cell.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        // `is_ok` is true for any non-error read, including empty string.
        // We accept "1" / non-empty as the "disable" signal; unset
        // (the common case) leaves the class GPU-enabled.
        let disabled = std::env::var(class.env_var())
            .ok()
            .filter(|s| !s.is_empty() && s != "0")
            .is_some();
        let enabled = !disabled;
        c.set(Some(enabled));
        enabled
    })
}

/// Convenience helper: wrap the master `backend` option in a per-class
/// check. Returns the backend only when both (a) it's actually attached
/// AND (b) this class is not disabled by env var.
///
/// Use this at dispatch sites:
/// ```ignore
/// let backend = weights.backend.as_deref();
/// let ffn_backend = gpu_tier::backend_for(GpuClass::Ffn, backend);
/// swiglu_ffn_lazy(.., ffn_backend);
/// ```
#[inline]
pub fn backend_for(
    class: GpuClass,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<&dyn larql_compute::ComputeBackend> {
    if is_enabled(class) {
        backend
    } else {
        None
    }
}
