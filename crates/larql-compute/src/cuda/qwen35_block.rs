//! Phase E.6.A — fused device-resident DeltaNet post-projection chain.
//!
//! Each Qwen3.6 DeltaNet layer fires seven host-returning kernel hooks
//! today (attn_qkv → attn_gate → conv1d → l2 Q → l2 K → recurrence →
//! rms_norm_heads → ssm_out). The matvecs do the heavy arithmetic, but
//! the *recurrence-adjacent* hooks (conv1d → l2 Q → l2 K → recurrence
//! → rms_norm_heads) each sync + dtoh to host and re-htod for the next
//! kernel, even though every intermediate buffer is GPU-resident in
//! intent. The four mid-block syncs alone burn ~20 ms/layer at the
//! current 43 ms total per DeltaNet linear layer.
//!
//! This module collapses that five-kernel post-projection chain into a
//! single device-resident pipeline with one `sync` + `dtoh` at the
//! end. The matvecs (attn_qkv, attn_gate, ssm_out) are intentionally
//! still host-returning — they go through the existing
//! `quant_matvec` path. Absorbing them into the device chain is the
//! E.6.B scope.
//!
//! Layout conventions (carried over from the per-kernel helpers in
//! `deltanet.rs`):
//! - `qkv_conv` is head-major flat — `qkv_conv[h * head_dim + d]`
//!   for each of the Q/K/V slabs.
//! - `l2_norm_dim_major_f32` expects `[head_dim, n_heads]` (dim-major).
//! - `deltanet_step_decay_first_f32` expects Q/K/V in
//!   `[head_dim, n_heads]` (dim-major); outputs `o` in the same.
//! - `rms_norm_heads_head_major_f32` expects head-major flat
//!   `[num_heads * head_dim]`.
//!
//! So we transpose `(n_heads, head_dim) -> (head_dim, n_heads)` between
//! the split + L2 + recurrence and back to head-major flat before
//! rms_norm_heads. Two small dedicated reshape kernels keep the
//! reshape on device.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const QWEN35_BLOCK_SRC: &str = r#"
// silu_inplace_f32: x[i] = x[i] * sigmoid(x[i]).
//
// Matches the host's `silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))`
// formulation (NOT the algebraically equivalent `x / (1 + exp(-x))`).
// fp32 rounding differs by a ulp between the two formulations, and
// the host code uses the two-step `1/d` then `x*y` order — see
// `deltanet_block::silu` and `deltanet_state::sigmoid`. Forcing the
// kernel to match shaves a few ulp off silu drift, which matters for
// `v` (un-normalised) feeding into the recurrence's rank-1 state
// update.
extern "C" __global__ void silu_inplace_f32(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sig = 1.0f / (1.0f + expf(-v));
    x[i] = v * sig;
}

// reshape_head_to_dim_f32: convert flat head-major
// `src[src_offset + h * head_dim + d]` into dim-major
// `dst[d * n_heads + h]`. `src_offset` lets the caller pass the
// whole `qkv_conv` buffer plus an offset to address the Q / K / V
// slabs in turn without having to sub-slice the cudarc buffer.
extern "C" __global__ void reshape_head_to_dim_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int head_dim,
    int n_heads,
    int src_offset
) {
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim || h >= n_heads) return;
    dst[(size_t)d * n_heads + h] = src[(size_t)src_offset + (size_t)h * head_dim + d];
}

// reshape_dim_to_head_f32: convert dim-major
// `src[d * n_heads + h]` into flat head-major `dst[h * head_dim + d]`.
// `src_offset` is unused for now but accepted for symmetry with the
// head→dim direction (also lets a future caller share buffers).
extern "C" __global__ void reshape_dim_to_head_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int head_dim,
    int n_heads,
    int src_offset
) {
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim || h >= n_heads) return;
    dst[(size_t)h * head_dim + d] = src[(size_t)src_offset + (size_t)d * n_heads + h];
}

// silu_mul_inplace_f32: o[i] *= silu(z[i]) using host's
// `x * sigmoid(x)` formulation (see `silu_inplace_f32` above).
extern "C" __global__ void silu_mul_inplace_f32(
    float* __restrict__ o,
    const float* __restrict__ z,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float zv = z[i];
    float sig = 1.0f / (1.0f + expf(-zv));
    float silu = zv * sig;
    o[i] = o[i] * silu;
}
"#;

static QWEN35_MODULE: OnceLock<(
    std::sync::Arc<CudaModule>,
    CudaFunction, // silu_inplace_f32
    CudaFunction, // reshape_head_to_dim_f32
    CudaFunction, // reshape_dim_to_head_f32
    CudaFunction, // silu_mul_inplace_f32
)> = OnceLock::new();

fn functions(
    drv: &Driver,
) -> Result<
    (
        &'static CudaFunction,
        &'static CudaFunction,
        &'static CudaFunction,
        &'static CudaFunction,
    ),
    CudaInitError,
> {
    if let Some((_, silu, head_to_dim, dim_to_head, silu_mul)) = QWEN35_MODULE.get() {
        return Ok((silu, head_to_dim, dim_to_head, silu_mul));
    }
    // Match `deltanet.rs`: disable fmad so the kernel's fp32
    // operations have the same rounding pattern as the host
    // reference. See the comment over `compile_ptx_with_opts` in
    // `deltanet.rs` for the why.
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(QWEN35_BLOCK_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile qwen35_block: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load qwen35_block module: {e:?}")))?;
    let silu = module
        .load_function("silu_inplace_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load silu_inplace: {e:?}")))?;
    let head_to_dim = module
        .load_function("reshape_head_to_dim_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load reshape_head_to_dim: {e:?}")))?;
    let dim_to_head = module
        .load_function("reshape_dim_to_head_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load reshape_dim_to_head: {e:?}")))?;
    let silu_mul = module
        .load_function("silu_mul_inplace_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load silu_mul_inplace: {e:?}")))?;
    let _ = QWEN35_MODULE.set((module, silu, head_to_dim, dim_to_head, silu_mul));
    let (_, silu, head_to_dim, dim_to_head, silu_mul) = QWEN35_MODULE.get().unwrap();
    Ok((silu, head_to_dim, dim_to_head, silu_mul))
}

/// Launch the deltanet recurrence kernel from `deltanet.rs` on the
/// caller-provided device buffers. Pre-launched device state buffer
/// (already cached by the backend) keeps the recurrent state on GPU.
#[allow(clippy::too_many_arguments)]
fn launch_deltanet_step_device(
    drv: &Driver,
    func: &CudaFunction,
    q_dev: &CudaSlice<f32>,
    k_dev: &CudaSlice<f32>,
    v_dev: &CudaSlice<f32>,
    log_g_dev: &CudaSlice<f32>,
    beta_dev: &CudaSlice<f32>,
    state_dev: &mut CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    s: usize,
    h_k: usize,
    h_v: usize,
    block_gqa: bool,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (h_v as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (2 * s * std::mem::size_of::<f32>()) as u32,
    };
    let s_i = s as i32;
    let h_k_i = h_k as i32;
    let h_v_i = h_v as i32;
    let block_gqa_i: i32 = if block_gqa { 1 } else { 0 };
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_dev)
            .arg(v_dev)
            .arg(log_g_dev)
            .arg(beta_dev)
            .arg(state_dev)
            .arg(out_dev)
            .arg(&s_i)
            .arg(&h_k_i)
            .arg(&h_v_i)
            .arg(&block_gqa_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch deltanet_step: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_conv1d_step_device(
    drv: &Driver,
    func: &CudaFunction,
    weight_dev: &CudaSlice<f32>,
    state_dev: &mut CudaSlice<f32>,
    new_dev: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    d_conv: usize,
    conv_dim: usize,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((conv_dim as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_conv_i = d_conv as i32;
    let conv_dim_i = conv_dim as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(weight_dev)
            .arg(state_dev)
            .arg(new_dev)
            .arg(out_dev)
            .arg(&d_conv_i)
            .arg(&conv_dim_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch conv1d_step: {e:?}")))?;
    }
    Ok(())
}

fn launch_l2_normalize_device(
    drv: &Driver,
    func: &CudaFunction,
    x_dev: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    head_dim: usize,
    n_heads: usize,
    eps: f32,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
    };
    let head_dim_i = head_dim as i32;
    let n_heads_i = n_heads as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(out_dev)
            .arg(&head_dim_i)
            .arg(&n_heads_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch l2_norm: {e:?}")))?;
    }
    Ok(())
}

fn launch_rms_norm_heads_device(
    drv: &Driver,
    func: &CudaFunction,
    x_dev: &CudaSlice<f32>,
    weight_dev: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_heads_i = num_heads as i32;
    let head_dim_i = head_dim as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(weight_dev)
            .arg(out_dev)
            .arg(&num_heads_i)
            .arg(&head_dim_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch rms_norm_heads: {e:?}")))?;
    }
    Ok(())
}

fn launch_silu_inplace(
    drv: &Driver,
    func: &CudaFunction,
    x_dev: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(&n_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch silu_inplace: {e:?}")))?;
    }
    Ok(())
}

fn launch_reshape(
    drv: &Driver,
    func: &CudaFunction,
    src_dev: &CudaSlice<f32>,
    dst_dev: &mut CudaSlice<f32>,
    head_dim: usize,
    n_heads: usize,
    src_offset: usize,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 64;
    let cfg = LaunchConfig {
        grid_dim: ((head_dim as u32).div_ceil(block_dim), n_heads as u32, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let head_dim_i = head_dim as i32;
    let n_heads_i = n_heads as i32;
    let src_offset_i = src_offset as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(src_dev)
            .arg(dst_dev)
            .arg(&head_dim_i)
            .arg(&n_heads_i)
            .arg(&src_offset_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch reshape: {e:?}")))?;
    }
    Ok(())
}

fn launch_silu_mul_inplace(
    drv: &Driver,
    func: &CudaFunction,
    o_dev: &mut CudaSlice<f32>,
    z_dev: &CudaSlice<f32>,
    n: usize,
) -> Result<(), CudaInitError> {
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(o_dev)
            .arg(z_dev)
            .arg(&n_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch silu_mul: {e:?}")))?;
    }
    Ok(())
}

/// Partial DeltaNet fusion (Phase E.6.B.2): chains L2 normalise Q +
/// L2 normalise K + delta-rule recurrence + RMSNorm-heads on the
/// device with a single sync at exit. Inputs come from host (q/k/v/
/// log_g/beta already host-computed by the unchanged conv1d → silu →
/// split/reshape pipeline), output is the head-major-flat
/// `o_normed[n_v_heads * head_v_dim]` ready for the caller to apply
/// `silu(z) *` on host and the `ssm_out` matvec separately.
///
/// Distinct from `deltanet_postproj_step_cached` in two ways:
///   1. **Excludes** the silu/conv1d/silu_z steps so the GPU-side
///      `silu` expf drift (the E.6.A.6 multi-position parity culprit)
///      doesn't enter the recurrence input. Per-step parity test was
///      validated to hold under this exact host-cpu-silu split.
///   2. **Excludes** the matvec projections (attn_qkv, attn_gate,
///      ssm_out, ssm_beta, ssm_alpha) — those stay on their existing
///      per-call paths.
///
/// Saves ~3 stream syncs per linear DeltaNet layer (L2-Q + L2-K +
/// recurrence + rms_norm_heads currently sync individually → now one
/// combined sync). At ~100 µs / sync × 48 layers that's ~14 ms /
/// token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deltanet_recurrence_block_cached(
    backend: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    log_g: &[f32],
    beta: &[f32],
    ssm_norm_weight: &[f32],
    recurrent_state: &mut [f32],
    head_v_dim: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    eps: f32,
    sequence_pos: usize,
    block_gqa: bool,
) -> Result<Vec<f32>, CudaInitError> {
    let key_dim = head_v_dim * n_k_heads;
    let value_dim = head_v_dim * n_v_heads;
    if head_v_dim == 0
        || n_v_heads == 0
        || n_k_heads == 0
        || q.len() != key_dim
        || k.len() != key_dim
        || v.len() != value_dim
        || log_g.len() != n_v_heads
        || beta.len() != n_v_heads
        || ssm_norm_weight.len() != head_v_dim
        || recurrent_state.len() != head_v_dim * head_v_dim * n_v_heads
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid recurrence_block shape: q={} k={} v={} log_g={} beta={} norm_w={} rec_state={} hd={head_v_dim} h_v={n_v_heads} h_k={n_k_heads}",
            q.len(),
            k.len(),
            v.len(),
            log_g.len(),
            beta.len(),
            ssm_norm_weight.len(),
            recurrent_state.len()
        )));
    }

    let drv = backend.driver();
    let (_silu_f, _head_to_dim_f, dim_to_head_f, _silu_mul_f) = functions(drv)?;
    let (_conv1d_f, deltanet_f, l2_f, rms_norm_f) = super::deltanet::module_functions(drv)?;

    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k)?;
    let v_dev = drv.device_buf_from(v)?;
    let log_g_dev = drv.device_buf_from(log_g)?;
    let beta_dev = drv.device_buf_from(beta)?;

    let mut q_norm_dev = drv.device_alloc_uninit(key_dim)?;
    let mut k_norm_dev = drv.device_alloc_uninit(key_dim)?;
    let mut o_dim_dev = drv.device_alloc_uninit(value_dim)?;
    let mut o_flat_dev = drv.device_alloc_uninit(value_dim)?;
    let mut o_normed_dev = drv.device_alloc_uninit(value_dim)?;

    launch_l2_normalize_device(
        drv,
        l2_f,
        &q_dev,
        &mut q_norm_dev,
        head_v_dim,
        n_k_heads,
        1e-6,
    )?;
    launch_l2_normalize_device(
        drv,
        l2_f,
        &k_dev,
        &mut k_norm_dev,
        head_v_dim,
        n_k_heads,
        1e-6,
    )?;

    backend.with_qwen35_state_device_buf(recurrent_state, sequence_pos, |state_dev| {
        launch_deltanet_step_device(
            drv,
            deltanet_f,
            &q_norm_dev,
            &k_norm_dev,
            &v_dev,
            &log_g_dev,
            &beta_dev,
            state_dev,
            &mut o_dim_dev,
            head_v_dim,
            n_k_heads,
            n_v_heads,
            block_gqa,
        )
    })?;

    launch_reshape(
        drv,
        dim_to_head_f,
        &o_dim_dev,
        &mut o_flat_dev,
        head_v_dim,
        n_v_heads,
        0,
    )?;

    backend.with_norm_device_buf(ssm_norm_weight, |w_dev| {
        launch_rms_norm_heads_device(
            drv,
            rms_norm_f,
            &o_flat_dev,
            w_dev,
            &mut o_normed_dev,
            n_v_heads,
            head_v_dim,
            eps,
        )
    })?;

    drv.sync()?;
    drv.to_host(&o_normed_dev)
}

/// Fused device-resident DeltaNet post-projection chain (Phase E.6.A).
///
/// Inputs are host slices (the calling code already has them on host).
/// All intermediates live on device. A single sync + dtoh delivers
/// `o_normed` back to the caller, which then applies `silu(z) * o`
/// elementwise (also on device — see `apply_silu_z_inplace_device`)
/// and the `ssm_out` matvec.
///
/// Returns `o_normed_flat[value_dim]` in head-major layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deltanet_postproj_step_cached(
    backend: &CudaBackend,
    qkv_mixed: &[f32],
    ssm_conv1d_weight: &[f32],
    log_g: &[f32],
    beta: &[f32],
    z: &[f32],
    ssm_norm_weight: &[f32],
    conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    head_v_dim: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    d_conv: usize,
    eps: f32,
    sequence_pos: usize,
    block_gqa: bool,
) -> Result<Vec<f32>, CudaInitError> {
    let key_dim = head_v_dim * n_k_heads;
    let value_dim = head_v_dim * n_v_heads;
    let conv_dim = 2 * key_dim + value_dim;
    let state_rows = d_conv.saturating_sub(1);

    if d_conv == 0
        || head_v_dim == 0
        || n_v_heads == 0
        || n_k_heads == 0
        || qkv_mixed.len() != conv_dim
        || ssm_conv1d_weight.len() != d_conv * conv_dim
        || conv_state.len() != state_rows * conv_dim
        || log_g.len() != n_v_heads
        || beta.len() != n_v_heads
        || z.len() != value_dim
        || ssm_norm_weight.len() != head_v_dim
        || recurrent_state.len() != head_v_dim * head_v_dim * n_v_heads
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid postproj shape: qkv={} conv_w={} conv_state={} log_g={} beta={} z={} norm_w={} rec_state={} d_conv={d_conv} hd={head_v_dim} h_v={n_v_heads} h_k={n_k_heads}",
            qkv_mixed.len(),
            ssm_conv1d_weight.len(),
            conv_state.len(),
            log_g.len(),
            beta.len(),
            z.len(),
            ssm_norm_weight.len(),
            recurrent_state.len()
        )));
    }

    let drv = backend.driver();
    let (silu_f, head_to_dim_f, dim_to_head_f, silu_mul_f) = functions(drv)?;
    let (conv1d_f, deltanet_f, l2_f, rms_norm_f) = super::deltanet::module_functions(drv)?;

    // Upload one-shot inputs.
    let qkv_mixed_dev = drv.device_buf_from(qkv_mixed)?;
    let log_g_dev = drv.device_buf_from(log_g)?;
    let beta_dev = drv.device_buf_from(beta)?;
    let z_dev = drv.device_buf_from(z)?;

    // Allocate intermediates.
    let mut qkv_conv_dev = drv.device_alloc_uninit(conv_dim)?;
    let mut q_dim_dev = drv.device_alloc_uninit(key_dim)?;
    let mut k_dim_dev = drv.device_alloc_uninit(key_dim)?;
    let mut v_dim_dev = drv.device_alloc_uninit(value_dim)?;
    let mut q_norm_dev = drv.device_alloc_uninit(key_dim)?;
    let mut k_norm_dev = drv.device_alloc_uninit(key_dim)?;
    let mut o_dim_dev = drv.device_alloc_uninit(value_dim)?;
    let mut o_flat_dev = drv.device_alloc_uninit(value_dim)?;
    let mut o_normed_dev = drv.device_alloc_uninit(value_dim)?;

    // 1. Conv1D step — mutates cached device conv state.
    backend.with_norm_device_buf(ssm_conv1d_weight, |weight_dev| {
        backend.with_qwen35_state_device_buf(conv_state, sequence_pos, |state_dev| {
            launch_conv1d_step_device(
                drv,
                conv1d_f,
                weight_dev,
                state_dev,
                &qkv_mixed_dev,
                &mut qkv_conv_dev,
                d_conv,
                conv_dim,
            )
        })
    })?;

    // 2. silu(qkv_conv) in place.
    launch_silu_inplace(drv, silu_f, &mut qkv_conv_dev, conv_dim)?;

    // 3. Reshape head-major splits into dim-major Q/K/V.
    //    qkv_conv is laid out as [Q_slab | K_slab | V_slab] head-major.
    //    The kernel takes a src_offset so we keep the whole buffer
    //    and read the three slabs without sub-slicing the cudarc
    //    buffer (which would yield CudaView, not CudaSlice).
    launch_reshape(
        drv,
        head_to_dim_f,
        &qkv_conv_dev,
        &mut q_dim_dev,
        head_v_dim,
        n_k_heads,
        0,
    )?;
    launch_reshape(
        drv,
        head_to_dim_f,
        &qkv_conv_dev,
        &mut k_dim_dev,
        head_v_dim,
        n_k_heads,
        key_dim,
    )?;
    launch_reshape(
        drv,
        head_to_dim_f,
        &qkv_conv_dev,
        &mut v_dim_dev,
        head_v_dim,
        n_v_heads,
        2 * key_dim,
    )?;

    // 4. L2 normalise Q and K per head (dim-major layout).
    launch_l2_normalize_device(
        drv,
        l2_f,
        &q_dim_dev,
        &mut q_norm_dev,
        head_v_dim,
        n_k_heads,
        1e-6,
    )?;
    launch_l2_normalize_device(
        drv,
        l2_f,
        &k_dim_dev,
        &mut k_norm_dev,
        head_v_dim,
        n_k_heads,
        1e-6,
    )?;

    // 5. Deltanet recurrence — uses cached recurrent state buffer.
    backend.with_qwen35_state_device_buf(recurrent_state, sequence_pos, |state_dev| {
        launch_deltanet_step_device(
            drv,
            deltanet_f,
            &q_norm_dev,
            &k_norm_dev,
            &v_dim_dev,
            &log_g_dev,
            &beta_dev,
            state_dev,
            &mut o_dim_dev,
            head_v_dim,
            n_k_heads,
            n_v_heads,
            block_gqa,
        )
    })?;

    // 6. Reshape o from dim-major to head-major flat.
    launch_reshape(
        drv,
        dim_to_head_f,
        &o_dim_dev,
        &mut o_flat_dev,
        head_v_dim,
        n_v_heads,
        0,
    )?;

    // 7. rms_norm_heads on head-major flat.
    backend.with_norm_device_buf(ssm_norm_weight, |w_dev| {
        launch_rms_norm_heads_device(
            drv,
            rms_norm_f,
            &o_flat_dev,
            w_dev,
            &mut o_normed_dev,
            n_v_heads,
            head_v_dim,
            eps,
        )
    })?;

    // 8. silu(z) * o_normed elementwise (in place into o_normed).
    launch_silu_mul_inplace(drv, silu_mul_f, &mut o_normed_dev, &z_dev, value_dim)?;

    // 9. Single sync + dtoh of the block-final output. The device
    // states (conv_state, recurrent_state) stay device-resident —
    // `with_qwen35_state_device_buf` keeps the device buffer
    // authoritative across calls within a sequence.
    drv.sync()?;
    drv.to_host(&o_normed_dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::backend::CudaBackend;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return None;
        }
        CudaBackend::new().ok()
    }

    /// Quantify GPU `silu` vs CPU `silu` drift at production scale.
    /// The CUDA `expf` in `silu_inplace_f32` is implemented by the
    /// CUDA math library (~1-2 ulp) while host `f32::exp` is glibc's
    /// expf (also ~1 ulp, but a different polynomial). Suspected
    /// culprit for the fused-path multi-position parity drift
    /// (E.6.A.6): silu drift on v (which skips L2 norm and feeds
    /// directly into the recurrence's rank-1 state update) compounds
    /// across 9 prompt + 5 decode positions × 48 layers.
    #[test]
    fn silu_inplace_drift_at_qwen35_shape() {
        let Some(backend) = gpu_or_skip() else { return };
        let drv = backend.driver();
        let (silu_f, _, _, _) = functions(drv).expect("module");

        // conv_dim = 2 * 128 * 16 + 128 * 48 = 10240 (Qwen3.6 27B).
        let n = 10240_usize;
        // Diverse fp32 range: include large negatives (where expf
        // matters), small values, and large positives.
        let host: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f32) / (n as f32) * 12.0 - 6.0; // [-6, 6]
                t
            })
            .collect();
        let mut dev = drv.device_buf_from(&host).expect("htod");
        launch_silu_inplace(drv, silu_f, &mut dev, n).expect("silu");
        drv.sync().expect("sync");
        let gpu = drv.to_host(&dev).expect("dtoh");

        // CPU reference matches the actual `silu(x) = x * sigmoid(x)`
        // formula used by `deltanet_block::silu` (NOT the algebraically
        // equivalent `x / (1 + exp(-x))` — fp32 rounding differs by a
        // ulp between the two formulations).
        let mut max_abs_xtimes = 0.0_f32;
        let mut max_abs_xdiv = 0.0_f32;
        let mut sum_abs_xtimes = 0.0_f32;
        let mut sum_abs_xdiv = 0.0_f32;
        for (i, &v) in host.iter().enumerate() {
            let cpu_xtimes = v * (1.0_f32 / (1.0_f32 + (-v).exp()));
            let cpu_xdiv = v / (1.0_f32 + (-v).exp());
            let d_xt = (gpu[i] - cpu_xtimes).abs();
            let d_xd = (gpu[i] - cpu_xdiv).abs();
            max_abs_xtimes = max_abs_xtimes.max(d_xt);
            max_abs_xdiv = max_abs_xdiv.max(d_xd);
            sum_abs_xtimes += d_xt;
            sum_abs_xdiv += d_xd;
        }
        eprintln!(
            "silu drift vs `x*sigmoid(x)` (host formulation): max_abs={max_abs_xtimes:.4e} mean={:.4e}",
            sum_abs_xtimes / n as f32
        );
        eprintln!(
            "silu drift vs `x/(1+exp(-x))`:                   max_abs={max_abs_xdiv:.4e} mean={:.4e}",
            sum_abs_xdiv / n as f32
        );
    }

    /// Validate the head→dim + dim→head reshape mini-kernels at
    /// Qwen3.6's production shapes (head_v_dim=128, n_v_heads=48,
    /// n_k_heads=16). Bugs at the larger grid dimension would not
    /// show up in the tiny-shape end-to-end test.
    #[test]
    fn reshape_kernels_match_cpu_at_qwen35_shape() {
        let backend = match gpu_or_skip() {
            Some(b) => b,
            None => return,
        };
        let head_dim = 128_usize;
        let n_heads = 48_usize;
        let drv = backend.driver();
        let (_, head_to_dim_f, dim_to_head_f, _) = functions(drv).expect("module");

        // Build a flat head-major source: src[h * head_dim + d] = h * 1000 + d.
        let src: Vec<f32> = (0..head_dim * n_heads)
            .map(|i| (i % head_dim) as f32 + ((i / head_dim) as f32) * 1000.0)
            .collect();
        let src_dev = drv.device_buf_from(&src).expect("htod");
        let mut dim_dev = drv.device_alloc_uninit(head_dim * n_heads).expect("alloc");

        launch_reshape(
            drv,
            head_to_dim_f,
            &src_dev,
            &mut dim_dev,
            head_dim,
            n_heads,
            0,
        )
        .expect("head_to_dim");
        drv.sync().expect("sync");
        let dim_host = drv.to_host(&dim_dev).expect("dtoh");

        // Expected: dim_host[d * n_heads + h] = src[h * head_dim + d].
        for h in 0..n_heads {
            for d in 0..head_dim {
                let want = src[h * head_dim + d];
                let got = dim_host[d * n_heads + h];
                assert_eq!(
                    got, want,
                    "head_to_dim @ (h={h}, d={d}): got {got} want {want}"
                );
            }
        }

        // Now the round-trip back.
        let mut head_dev = drv.device_alloc_uninit(head_dim * n_heads).expect("alloc");
        launch_reshape(
            drv,
            dim_to_head_f,
            &dim_dev,
            &mut head_dev,
            head_dim,
            n_heads,
            0,
        )
        .expect("dim_to_head");
        drv.sync().expect("sync");
        let head_host = drv.to_host(&head_dev).expect("dtoh");
        for (i, (a, b)) in head_host.iter().zip(src.iter()).enumerate() {
            assert_eq!(a, b, "round-trip @ {i}");
        }

        // Test src_offset > 0: the K/V slabs in the fused path call
        // reshape_head_to_dim with a non-zero offset into the
        // concatenated qkv_conv buffer. Bug here wouldn't show up in
        // the offset=0 test above.
        let key_dim = head_dim * 16;
        let value_dim = head_dim * 48;
        let conv_dim = 2 * key_dim + value_dim;
        let qkv_src: Vec<f32> = (0..conv_dim).map(|i| 0.001 + i as f32 * 0.0005).collect();
        let qkv_dev = drv.device_buf_from(&qkv_src).expect("htod qkv");
        let mut k_dim_dev = drv.device_alloc_uninit(key_dim).expect("alloc k");
        launch_reshape(
            drv,
            head_to_dim_f,
            &qkv_dev,
            &mut k_dim_dev,
            head_dim,
            16,
            key_dim,
        )
        .expect("k slab reshape");
        drv.sync().expect("sync");
        let k_host = drv.to_host(&k_dim_dev).expect("dtoh k");
        for h in 0..16 {
            for d in 0..head_dim {
                let want = qkv_src[key_dim + h * head_dim + d];
                let got = k_host[d * 16 + h];
                assert_eq!(got, want, "k slab @ (h={h}, d={d}): got {got} want {want}");
            }
        }
        let mut v_dim_dev = drv.device_alloc_uninit(value_dim).expect("alloc v");
        launch_reshape(
            drv,
            head_to_dim_f,
            &qkv_dev,
            &mut v_dim_dev,
            head_dim,
            48,
            2 * key_dim,
        )
        .expect("v slab reshape");
        drv.sync().expect("sync");
        let v_host = drv.to_host(&v_dim_dev).expect("dtoh v");
        for h in 0..48 {
            for d in 0..head_dim {
                let want = qkv_src[2 * key_dim + h * head_dim + d];
                let got = v_host[d * 48 + h];
                assert_eq!(got, want, "v slab @ (h={h}, d={d}): got {got} want {want}");
            }
        }
    }

    /// L2 norm + rms_norm_heads parity at Qwen3.6 production shapes.
    /// Both kernels do a parallel-reduction sum (tree pattern across
    /// 256 threads) while the CPU reference adds elements in sequence.
    /// The difference is normally ~1e-7 in the sum-of-squares, which
    /// scales by `rsqrt(sum + eps)` and then propagates downstream.
    /// Quantifies the drift so we know how much of the multi-position
    /// fused-path divergence is attributable to reduction order.
    #[test]
    fn l2_and_rms_norm_drift_at_qwen35_shape() {
        let Some(backend) = gpu_or_skip() else { return };
        let drv = backend.driver();

        // L2 normalise per K-head: head_dim=128, n_heads=16
        let head_dim = 128_usize;
        let n_k_heads = 16_usize;
        let eps = 1e-6_f32;
        let x: Vec<f32> = (0..head_dim * n_k_heads)
            .map(|i| {
                let r = ((i as i64 * 1103515245 + 12345) & 0x7fffffff) as f32;
                (r / 2.147483647e9) * 0.5 - 0.25
            })
            .collect();
        let mut expected_l2 = vec![0.0_f32; x.len()];
        for h in 0..n_k_heads {
            let mut ss = 0.0_f32;
            for d in 0..head_dim {
                let v = x[d * n_k_heads + h];
                ss += v * v;
            }
            let inv = 1.0_f32 / (ss + eps).sqrt();
            for d in 0..head_dim {
                expected_l2[d * n_k_heads + h] = x[d * n_k_heads + h] * inv;
            }
        }
        let gpu_l2 =
            super::super::deltanet::l2_normalize_per_head(&backend, &x, head_dim, n_k_heads, eps)
                .expect("l2 gpu");
        let mut max_l2 = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        for i in 0..gpu_l2.len() {
            let d = (gpu_l2[i] - expected_l2[i]).abs();
            max_l2 = max_l2.max(d);
            sum_l2 += d;
        }
        eprintln!(
            "l2_norm drift @ qwen35: max_abs={max_l2:.4e} mean_abs={:.4e}",
            sum_l2 / gpu_l2.len() as f32
        );

        // rms_norm_heads: num_heads=48, head_dim=128 (head-major flat)
        let num_heads = 48_usize;
        let x2: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| {
                let r = ((i as i64 * 1664525 + 1013904223) & 0x7fffffff) as f32;
                (r / 2.147483647e9) * 0.4 - 0.2
            })
            .collect();
        let w: Vec<f32> = (0..head_dim).map(|d| 1.0 + d as f32 * 0.003).collect();
        let mut expected_rms = vec![0.0_f32; x2.len()];
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut ss = 0.0_f32;
            for d in 0..head_dim {
                let v = x2[off + d];
                ss += v * v;
            }
            let inv = 1.0_f32 / (ss / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                expected_rms[off + d] = x2[off + d] * inv * w[d];
            }
        }
        let gpu_rms =
            super::super::deltanet::rms_norm_heads(&backend, &x2, &w, num_heads, head_dim, eps)
                .expect("rms gpu");
        let mut max_rms = 0.0_f32;
        let mut sum_rms = 0.0_f32;
        for i in 0..gpu_rms.len() {
            let d = (gpu_rms[i] - expected_rms[i]).abs();
            max_rms = max_rms.max(d);
            sum_rms += d;
        }
        eprintln!(
            "rms_norm drift @ qwen35: max_abs={max_rms:.4e} mean_abs={:.4e}",
            sum_rms / gpu_rms.len() as f32
        );
        let _ = drv;
    }

    /// Recurrence kernel parity at Qwen3.6's production shapes
    /// (s=128, h_k=16, h_v=48). The existing tiny-shape test
    /// (s=4) doesn't catch large-block reductions or the
    /// 16384-element decay loop.
    #[test]
    fn deltanet_step_matches_cpu_at_qwen35_shape() {
        let Some(backend) = gpu_or_skip() else { return };
        let s = 128;
        let h_k = 16;
        let h_v = 48;
        // Reproducible pseudo-random inputs.
        let make = |off: usize, scale: f32| -> Vec<f32> {
            (0..off)
                .map(|i| {
                    ((i as i64 * 1103515245 + 12345) & 0x7fff) as f32 / 32768.0 * scale
                        - 0.5 * scale
                })
                .collect()
        };
        let q = make(s * h_k, 0.4);
        let k = make(s * h_k, 0.4);
        let v = make(s * h_v, 0.6);
        let log_g: Vec<f32> = (0..h_v).map(|h| -0.05 - h as f32 * 0.003).collect();
        let beta: Vec<f32> = (0..h_v).map(|h| 0.3 + h as f32 * 0.005).collect();
        // Non-trivial initial state so the decay × outer-product × readout
        // ordering is exercised end-to-end.
        let mut state: Vec<f32> = (0..s * s * h_v)
            .map(|i| ((i as i64 * 1664525 + 1013904223) & 0xffff) as f32 / 65536.0 * 0.1 - 0.05)
            .collect();
        let mut state_ref = state.clone();

        // CPU reference.
        let scale_q = 1.0_f32 / (s as f32).sqrt();
        let mut expected = vec![0.0_f32; s * h_v];
        for h in 0..h_v {
            let kh = h * h_k / h_v;
            let g = (log_g[h]).exp();
            for r in 0..s {
                for c in 0..s {
                    state_ref[(r * s + c) * h_v + h] *= g;
                }
            }
            let mut sk = vec![0.0_f32; s];
            for c in 0..s {
                for r in 0..s {
                    sk[c] += state_ref[(r * s + c) * h_v + h] * k[r * h_k + kh];
                }
            }
            let mut d = vec![0.0_f32; s];
            for c in 0..s {
                d[c] = (v[c * h_v + h] - sk[c]) * beta[h];
            }
            for r in 0..s {
                for c in 0..s {
                    state_ref[(r * s + c) * h_v + h] += k[r * h_k + kh] * d[c];
                }
            }
            for c in 0..s {
                let mut acc = 0.0_f32;
                for r in 0..s {
                    acc += state_ref[(r * s + c) * h_v + h] * q[r * h_k + kh];
                }
                expected[c * h_v + h] = acc * scale_q;
            }
        }

        let out = super::super::deltanet::deltanet_step(
            &backend, &q, &k, &v, &log_g, &beta, &mut state, s, h_k, h_v, false,
        )
        .expect("cuda deltanet @ qwen35 shape");
        // The kernel does the 16384-element decay loop in parallel
        // with FP non-determinism vs the serial CPU ref, so allow a
        // modest tolerance.
        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f32;
        for i in 0..out.len() {
            let diff = (out[i] - expected[i]).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
        }
        let mean_abs = sum_abs / out.len() as f32;
        eprintln!("recurrence @ qwen35 shape: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
        // Loose-but-not-vacuous tolerance: a per-element drift of
        // 1e-3 here is fine; one of 0.1 would indicate a logic bug.
        assert!(
            max_abs < 1e-3 && mean_abs < 1e-4,
            "recurrence drift: max_abs={max_abs} mean_abs={mean_abs}"
        );
    }

    /// End-to-end tiny-shape parity vs CPU. Mirrors
    /// `deltanet::tests::deltanet_step_matches_cpu_tiny_decay_first`
    /// but composes the full conv1d → l2 → recurrence → rms_norm →
    /// silu_z chain. The CPU reference comes from
    /// `larql-inference::attention::deltanet_block`, but we build a
    /// minimal closed-form analogue here so this crate stays
    /// dependency-free.
    #[test]
    fn deltanet_postproj_matches_unfused_gpu_chain() {
        let backend = match gpu_or_skip() {
            Some(b) => b,
            None => return,
        };

        let head_v_dim = 4;
        let n_v_heads = 2;
        let n_k_heads = 2;
        let d_conv = 2;
        let eps = 1e-6_f32;
        let key_dim = head_v_dim * n_k_heads;
        let value_dim = head_v_dim * n_v_heads;
        let conv_dim = 2 * key_dim + value_dim;
        let state_rows = d_conv - 1;

        let qkv_mixed: Vec<f32> = (0..conv_dim).map(|i| 0.1 + i as f32 * 0.01).collect();
        let conv_w: Vec<f32> = (0..d_conv * conv_dim)
            .map(|i| 0.2 + i as f32 * 0.003)
            .collect();
        let log_g: Vec<f32> = (0..n_v_heads).map(|h| -0.1 - h as f32 * 0.05).collect();
        let beta: Vec<f32> = (0..n_v_heads).map(|h| 0.3 + h as f32 * 0.1).collect();
        let z: Vec<f32> = (0..value_dim).map(|i| 0.05 + i as f32 * 0.02).collect();
        let norm_w: Vec<f32> = (0..head_v_dim).map(|d| 1.0 + d as f32 * 0.1).collect();

        // Two parallel copies of state so the fused path doesn't
        // pollute the per-step reference.
        let mut conv_state_fused = vec![0.0_f32; state_rows * conv_dim];
        let mut rec_state_fused = vec![0.0_f32; head_v_dim * head_v_dim * n_v_heads];
        let mut conv_state_step = vec![0.0_f32; state_rows * conv_dim];
        let mut rec_state_step = vec![0.0_f32; head_v_dim * head_v_dim * n_v_heads];

        let fused = deltanet_postproj_step_cached(
            &backend,
            &qkv_mixed,
            &conv_w,
            &log_g,
            &beta,
            &z,
            &norm_w,
            &mut conv_state_fused,
            &mut rec_state_fused,
            head_v_dim,
            n_v_heads,
            n_k_heads,
            d_conv,
            eps,
            0,
            false,
        )
        .expect("fused postproj");

        // Reference: per-kernel chain using the already-validated
        // `deltanet::*_cached` helpers, replicated in CPU-ish
        // composition. Build qkv_conv → silu → split → reshape → L2
        // → recurrence → rms_norm → silu(z)*.
        let mut qkv_conv = super::super::deltanet::causal_conv1d_step_cached(
            &backend,
            &conv_w,
            &mut conv_state_step,
            &qkv_mixed,
            d_conv,
            conv_dim,
            // sequence_pos=1 so the cache for the step path is
            // distinct from the fused path. (Different `(state_ptr, len)`
            // keys anyway, but a non-zero pos suppresses re-upload.)
            1,
        )
        .expect("step conv1d");
        for v in qkv_conv.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        // Split + transpose head→dim manually.
        let mut q_dim = vec![0.0_f32; key_dim];
        let mut k_dim = vec![0.0_f32; key_dim];
        let mut v_dim = vec![0.0_f32; value_dim];
        for h in 0..n_k_heads {
            for d in 0..head_v_dim {
                q_dim[d * n_k_heads + h] = qkv_conv[h * head_v_dim + d];
                k_dim[d * n_k_heads + h] = qkv_conv[key_dim + h * head_v_dim + d];
            }
        }
        for h in 0..n_v_heads {
            for d in 0..head_v_dim {
                v_dim[d * n_v_heads + h] = qkv_conv[2 * key_dim + h * head_v_dim + d];
            }
        }
        let q_normed = super::super::deltanet::l2_normalize_per_head(
            &backend, &q_dim, head_v_dim, n_k_heads, 1e-6,
        )
        .expect("l2 q");
        let k_normed = super::super::deltanet::l2_normalize_per_head(
            &backend, &k_dim, head_v_dim, n_k_heads, 1e-6,
        )
        .expect("l2 k");
        let o_dim = super::super::deltanet::deltanet_step_cached(
            &backend,
            &q_normed,
            &k_normed,
            &v_dim,
            &log_g,
            &beta,
            &mut rec_state_step,
            head_v_dim,
            n_k_heads,
            n_v_heads,
            1,
            true,
        )
        .expect("recurrence");
        // Reshape dim→head_major flat.
        let mut o_flat = vec![0.0_f32; value_dim];
        for h in 0..n_v_heads {
            for d in 0..head_v_dim {
                o_flat[h * head_v_dim + d] = o_dim[d * n_v_heads + h];
            }
        }
        let mut o_normed = super::super::deltanet::rms_norm_heads(
            &backend, &o_flat, &norm_w, n_v_heads, head_v_dim, eps,
        )
        .expect("rms_norm_heads");
        for i in 0..value_dim {
            let zv = z[i];
            let silu = zv / (1.0 + (-zv).exp());
            o_normed[i] *= silu;
        }

        assert_eq!(fused.len(), o_normed.len());
        for (i, (a, b)) in fused.iter().zip(o_normed.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "postproj[{i}]: fused={a} vs step={b}");
        }
    }
}
