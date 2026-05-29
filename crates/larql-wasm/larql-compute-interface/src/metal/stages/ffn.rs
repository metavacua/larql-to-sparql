//! Feed-forward block — gate+up → activation → down.
//!
//! Two variants depending on `FfnType`:
//!
//! - **Gated** (Llama / Gemma / Qwen / most modern): `out = down(act(gate) ⊙ up)`
//!   with activation = SiLU or GELU-tanh. Dispatched as
//!   `gate_matvec + up_matvec + geglu + down_matvec`.
//!
//! - **Standard** (StarCoder2): `out = down(act(up))`. Dispatched as
//!   `up_matvec + activation + down_matvec`. No gate.
//!
//! All matvecs are format-aware (`stages::quant_matvec`). Activation is a
//! single multi-position dispatch over `seq_len * inter` elementwise
//! threads.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::quant_matvec;

pub use crate::pipeline::Activation;

/// Whether the Metal backend ships a shader for `act`. SiLU and
/// GELU-tanh are wired today; GeluExact and ReLU have no kernels.
///
/// Pure function — independent of any Metal pipeline. Used by the
/// dispatch helpers below and exposed so callers (and tests) can
/// validate a layer before opening a command encoder.
pub fn metal_supports_activation(act: Activation) -> bool {
    matches!(act, Activation::Silu | Activation::GeluTanh)
}

/// Panic with a clear message when `act` has no Metal shader.
/// Production decode call sites use this to fail loud rather than
/// silently routing GeluExact / ReLU layers to SiLU (the prior
/// behaviour, which produced wrong logits with no signal).
pub fn assert_metal_activation_supported(act: Activation, site: &'static str) {
    if !metal_supports_activation(act) {
        panic!(
            "{site}: no shader for {act:?}. \
             Add a kernel and pipeline before routing this activation, or \
             route the layer through CPU. Silently falling back to SiLU is \
             no longer supported."
        );
    }
}

/// Pick the GEGLU-style "act(gate) * up" kernel for this activation.
fn geglu_pipeline_for<'a>(
    activation: Activation,
    silu: &'a ComputePipelineState,
    gelu_tanh: &'a ComputePipelineState,
) -> &'a ComputePipelineState {
    assert_metal_activation_supported(activation, "metal::stages::ffn::geglu_pipeline_for");
    match activation {
        Activation::Silu => silu,
        Activation::GeluTanh => gelu_tanh,
        // assert above prevents reaching here.
        Activation::GeluExact | Activation::ReLU => unreachable!(),
    }
}

/// Pick the in-place activation kernel (Standard / non-gated FFN path).
fn activation_pipeline_for<'a>(
    activation: Activation,
    silu: &'a ComputePipelineState,
    gelu_tanh: &'a ComputePipelineState,
) -> &'a ComputePipelineState {
    assert_metal_activation_supported(activation, "metal::stages::ffn::activation_pipeline_for");
    match activation {
        Activation::Silu => silu,
        Activation::GeluTanh => gelu_tanh,
        Activation::GeluExact | Activation::ReLU => unreachable!(),
    }
}

/// Optional fused activation+down kernels. When `down_format` matches
/// (`Q4_K` → `q4k`, `Q6_K` → `q6k`) and the matching kernel is
/// supplied, [`encode_gated`] skips the separate GEGLU dispatch and
/// the inter-sized activation buffer write/read per position.
pub struct FusedGegluDown<'a> {
    /// `q4k_geglu_silu_down` — Q4_K down + SiLU (Llama-style).
    pub q4k_silu: Option<&'a crate::metal::kernel::KernelHandle>,
    /// `q4k_geglu_gelu_tanh_down` — Q4_K down + GELU-tanh.
    pub q4k_gelu_tanh: Option<&'a crate::metal::kernel::KernelHandle>,
    /// `q6k_geglu_silu_down` — Q6_K down + SiLU (production
    /// Llama 2 / Mistral with Ollama-convention extracts).
    pub q6k_silu: Option<&'a crate::metal::kernel::KernelHandle>,
    /// `q6k_geglu_gelu_tanh_down` — Q6_K down + GELU-tanh
    /// (production Gemma 3 / 4 with Ollama-convention extracts).
    pub q6k_gelu_tanh: Option<&'a crate::metal::kernel::KernelHandle>,
}

/// Gated FFN (Llama / Gemma / Qwen): `down(act(gate) * up)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    geglu_silu_pipeline: &ComputePipelineState,
    geglu_gelu_tanh_pipeline: &ComputePipelineState,
    fused_down: FusedGegluDown<'_>,
    gate_format: crate::QuantFormat,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    gate_buf: &Buffer,
    up_buf: &Buffer,
    down_buf: &Buffer,
    ffn_norm_out: &Buffer, // f32 input for Q4_K / Q6_K / Q4_KF
    ffn_q8_in: &Buffer,    // Q8 input for Q4_0 / Q8_0
    ffn_q8s_in: &Buffer,
    gate_scratch: &Buffer, // holds per-position `inter` floats
    up_scratch: &Buffer,
    act_scratch: &Buffer,
    down_out: &Buffer,
    seq_len: usize,
    inter: usize,
    hidden: usize,
    h_stride_bytes: u64,     // hidden * 4
    inter_stride_bytes: u64, // inter * 4
    q8_stride_bytes: u64,    // Q8 input bytes per pos
    q8s_stride_bytes: u64,   // Q8 scales bytes per pos
) {
    // Gate+up per position. `q4k_matmul` wiring tried twice on Gemma 3 4B,
    // both falsified end-to-end:
    //   - 2026-04-28: kernel-isolated 1.79× → long-prompt prefill regressed
    //     10% (2933 → 3268 ms on 340 tokens).
    //   - 2026-05-09: re-bench under post-dispatch-fix + post-QKV-defuse
    //     state still regressed 5–7% across 10/50/150-token prompts
    //     (e.g. 1392 → 1469 ms at 150 tokens).
    // Diagnosis: the kernel is bandwidth-bound, and on long prompts the
    // matmul's [seq_len × hidden] X working set thrashes GPU L1 — the
    // dequant amortisation gain is paid back in DRAM↔L1 traffic.
    // The matmul kernel + `q4k_matmul` backend method + parity tests
    // remain shipped (useful for re-validation on future hardware), but
    // wiring it into the production prefill path is empirically dead.
    // Closing the prefill gap to ollama needs a different matmul kernel
    // (e.g. K-dim tiled, or Apple `simdgroup_matrix` intrinsics), not a
    // re-wiring of the current one.
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc,
            gate_format,
            gate_buf,
            ffn_norm_out,
            h_off,
            ffn_q8_in,
            q8_off,
            ffn_q8s_in,
            q8s_off,
            gate_scratch,
            inter_off,
            pipes,
            inter,
            hidden,
        );
        quant_matvec::encode(
            enc,
            up_format,
            up_buf,
            ffn_norm_out,
            h_off,
            ffn_q8_in,
            q8_off,
            ffn_q8s_in,
            q8s_off,
            up_scratch,
            inter_off,
            pipes,
            inter,
            hidden,
        );
    }

    // Fast path: Q4_K down + supplied fused kernel → skip GEGLU
    // dispatch entirely, fuse activation into down.
    //
    // Q6_K fields on `FusedGegluDown` are present (kernels built and
    // parity-tested) but **deliberately not routed here**. With
    // GELU-tanh activation the fused kernel recomputes tanh() N=hidden
    // times per input element (once per output row) vs once in the
    // separated `geglu_gelu_tanh` dispatch. At N=2560 (Gemma 3 4B) the
    // extra 2560× tanh cost regresses decode 67.9→62.2 tok/s regardless
    // of TG-memory caching (gate/up bandwidth was never the bottleneck).
    // Re-enable when a cheaper activation variant or act[] precompute
    // avoids the per-row tanh explosion.
    // The fused Q4_K geglu+down kernel produces NaN in the dense prefill
    // path on Gemma 3 4B (q4k-downq4k) and Gemma 4 31B (q4k) — the model
    // emits empty output because every hidden-state value comes back NaN.
    // The kernel's own unit test (`test_kernel_q4k_geglu_down.rs`) passes,
    // so the bug is shape- or data-pattern-specific and not visible from
    // synthetic inputs. The separated path (GEGLU dispatch + q4k_matvec)
    // produces correct, generative output for the same weights, so default
    // is now SEPARATED. Set `LARQL_FUSED_DOWN=1` to re-enable the fused
    // path for benchmarking once the kernel is fixed.
    let use_fused = crate::options::env_flag(crate::options::ENV_FUSED_DOWN);
    let fused_kernel = if use_fused {
        // Q6_K + non-tanh combos return None deliberately (no fused
        // kernel exists). GeluExact / ReLU also return None — they
        // hit the explicit panic in `geglu_pipeline_for` below if the
        // separated path also reaches them.
        match (down_format, activation) {
            (crate::QuantFormat::Q4_K, Activation::Silu) => fused_down.q4k_silu,
            (crate::QuantFormat::Q4_K, Activation::GeluTanh) => fused_down.q4k_gelu_tanh,
            _ => None,
        }
    } else {
        None
    };
    let _ = (fused_down.q6k_silu, fused_down.q6k_gelu_tanh); // silence unused-field warnings

    if let Some(kernel) = fused_kernel {
        for pos in 0..seq_len {
            let h_off = pos as u64 * h_stride_bytes;
            let inter_off = pos as u64 * inter_stride_bytes;
            let n_tgs = (hidden as u64).div_ceil(kernel.rows_per_tg);
            let n_val = hidden as u32;
            let k_val = inter as u32;
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(down_buf), 0);
            enc.set_buffer(1, Some(gate_scratch), inter_off);
            enc.set_buffer(2, Some(up_scratch), inter_off);
            enc.set_buffer(3, Some(down_out), h_off);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
        }
        return;
    }

    // Separated path: GEGLU then format-aware down.
    {
        let total_inter = (seq_len * inter) as u64;
        let total_inter_val = (seq_len * inter) as u32;
        let geglu_pipe =
            geglu_pipeline_for(activation, geglu_silu_pipeline, geglu_gelu_tanh_pipeline);
        enc.set_compute_pipeline_state(geglu_pipe);
        enc.set_buffer(0, Some(gate_scratch), 0);
        enc.set_buffer(1, Some(up_scratch), 0);
        enc.set_buffer(2, Some(act_scratch), 0);
        enc.set_bytes(3, 4, &total_inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
    }

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc,
            down_format,
            down_buf,
            act_scratch,
            inter_off,
            ffn_q8_in,
            q8_off,
            ffn_q8s_in,
            q8s_off,
            down_out,
            h_off,
            pipes,
            hidden,
            inter,
        );
    }
}

/// Standard FFN (StarCoder2): `down(act(up))`. No gate.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    silu_pipeline: &ComputePipelineState,
    gelu_tanh_pipeline: &ComputePipelineState,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    up_buf: &Buffer,
    down_buf: &Buffer,
    ffn_norm_out: &Buffer,
    ffn_q8_in: &Buffer,
    ffn_q8s_in: &Buffer,
    up_scratch: &Buffer,
    act_scratch: &Buffer,
    down_out: &Buffer,
    seq_len: usize,
    inter: usize,
    hidden: usize,
    h_stride_bytes: u64,
    inter_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
) {
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc,
            up_format,
            up_buf,
            ffn_norm_out,
            h_off,
            ffn_q8_in,
            q8_off,
            ffn_q8s_in,
            q8s_off,
            up_scratch,
            inter_off,
            pipes,
            inter,
            hidden,
        );
    }

    {
        let total_inter = (seq_len * inter) as u64;
        let total_inter_val = (seq_len * inter) as u32;
        let act_pipe = activation_pipeline_for(activation, silu_pipeline, gelu_tanh_pipeline);
        enc.set_compute_pipeline_state(act_pipe);
        enc.set_buffer(0, Some(up_scratch), 0);
        enc.set_buffer(1, Some(act_scratch), 0);
        enc.set_bytes(2, 4, &total_inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
    }

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc,
            down_format,
            down_buf,
            act_scratch,
            inter_off,
            ffn_q8_in,
            q8_off,
            ffn_q8s_in,
            q8s_off,
            down_out,
            h_off,
            pipes,
            hidden,
            inter,
        );
    }
}

#[cfg(test)]
mod activation_support_tests {
    use super::*;

    #[test]
    fn metal_supports_silu_and_gelu_tanh() {
        assert!(metal_supports_activation(Activation::Silu));
        assert!(metal_supports_activation(Activation::GeluTanh));
    }

    #[test]
    fn metal_does_not_support_gelu_exact_or_relu() {
        assert!(!metal_supports_activation(Activation::GeluExact));
        assert!(!metal_supports_activation(Activation::ReLU));
    }

    /// Pin the panic message — production decode reads this string when
    /// a malformed model lands on Metal. Same message must mention the
    /// activation variant (so logs are actionable) and the site (so
    /// the reader knows where to look).
    #[test]
    #[should_panic(expected = "no shader for ReLU")]
    fn assert_panics_on_relu_with_clear_message() {
        assert_metal_activation_supported(Activation::ReLU, "test_site");
    }

    #[test]
    #[should_panic(expected = "no shader for GeluExact")]
    fn assert_panics_on_gelu_exact_with_clear_message() {
        assert_metal_activation_supported(Activation::GeluExact, "test_site");
    }

    #[test]
    fn assert_is_a_noop_on_supported() {
        assert_metal_activation_supported(Activation::Silu, "test");
        assert_metal_activation_supported(Activation::GeluTanh, "test");
    }
}
