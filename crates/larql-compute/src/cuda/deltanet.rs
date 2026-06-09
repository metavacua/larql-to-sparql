//! CUDA kernels for Qwen3.6 Gated DeltaNet state updates.
//!
//! These wrappers are correctness-first and keep the public state on
//! the host for now: each call uploads the current state, runs the
//! kernel, copies the mutated state back, and returns the output. E.6
//! can make the same kernels device-resident by passing cached device
//! slices instead of host slices.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const DELTANET_SRC: &str = r#"
extern "C" __global__ void causal_conv1d_step_f32(
    const float* __restrict__ weight,
    float* __restrict__ state,
    const float* __restrict__ new_x,
    float* __restrict__ out,
    int d_conv,
    int conv_dim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    int state_rows = d_conv - 1;
    float acc = 0.0f;
    for (int k = 0; k < state_rows; ++k) {
        acc += weight[(size_t)k * conv_dim + c] * state[(size_t)k * conv_dim + c];
    }
    acc += weight[(size_t)(d_conv - 1) * conv_dim + c] * new_x[c];
    out[c] = acc;

    for (int k = 0; k + 1 < state_rows; ++k) {
        state[(size_t)k * conv_dim + c] = state[(size_t)(k + 1) * conv_dim + c];
    }
    if (state_rows > 0) {
        state[(size_t)(state_rows - 1) * conv_dim + c] = new_x[c];
    }
}

extern "C" __global__ void deltanet_step_decay_first_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ log_g,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    int s,
    int h_k,
    int h_v,
    int block_gqa
) {
    int h = blockIdx.x;
    if (h >= h_v) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    // GQA broadcast pattern — arch-dependent. See
    // `delta_net_step` in crates/larql-inference/src/attention/deltanet_recurrence.rs
    // for the bit-verified rationale (BLOCK for qwen3next, CYCLE for qwen35moe).
    // Both reduce to identity when `h_v == h_k`.
    int kh = block_gqa ? (h * h_k / h_v) : (h % h_k);
    float g = expf(log_g[h]);
    float b = beta[h];
    float scale_q = rsqrtf((float)s);

    extern __shared__ float smem[];
    float* sk = smem;
    float* d = smem + s;

    // 1. Decay state in place. Host layout is ndarray row-major
    // state[r, c, h] with h as fastest-moving axis.
    int ss = s * s;
    for (int idx = tid; idx < ss; idx += bdim) {
        int r = idx / s;
        int c = idx - r * s;
        size_t off = ((size_t)r * s + c) * h_v + h;
        state[off] *= g;
    }
    __syncthreads();

    // 2. sk[c] = decayed_state^T @ k.
    for (int c = tid; c < s; c += bdim) {
        float acc = 0.0f;
        for (int r = 0; r < s; ++r) {
            float k_val = k[(size_t)r * h_k + kh];
            size_t off = ((size_t)r * s + c) * h_v + h;
            acc += state[off] * k_val;
        }
        sk[c] = acc;
    }
    __syncthreads();

    // 3. d = (v - sk) * beta.
    for (int c = tid; c < s; c += bdim) {
        d[c] = (v[(size_t)c * h_v + h] - sk[c]) * b;
    }
    __syncthreads();

    // 4. state += k outer d.
    for (int idx = tid; idx < ss; idx += bdim) {
        int r = idx / s;
        int c = idx - r * s;
        float k_val = k[(size_t)r * h_k + kh];
        size_t off = ((size_t)r * s + c) * h_v + h;
        state[off] += k_val * d[c];
    }
    __syncthreads();

    // 5. out[c, h] = state[:, c, h] dot q[:, kh] / sqrt(s).
    for (int c = tid; c < s; c += bdim) {
        float acc = 0.0f;
        for (int r = 0; r < s; ++r) {
            size_t off = ((size_t)r * s + c) * h_v + h;
            acc += state[off] * q[(size_t)r * h_k + kh];
        }
        out[(size_t)c * h_v + h] = acc * scale_q;
    }
}

extern "C" __global__ void l2_norm_dim_major_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int head_dim,
    int n_heads,
    float eps
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];

    // Compute sum-of-squares with the same per-element addition order
    // as the host's `l2_normalize_per_head_eps` (single sequential
    // accumulator over d = 0..head_dim). Parallel tree reduction
    // would give bit-different results from CPU and the drift
    // compounds non-trivially through the per-position recurrence
    // (E.6.A.6). One thread sums; the others wait at the barrier and
    // then help with the elementwise normalize step.
    if (tid == 0) {
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float v = x[(size_t)d * n_heads + h];
            ss += v * v;
        }
        smem[0] = ss;
    }
    __syncthreads();
    float inv = rsqrtf(smem[0] + eps);
    for (int d = tid; d < head_dim; d += bdim) {
        size_t off = (size_t)d * n_heads + h;
        out[off] = x[off] * inv;
    }
}

extern "C" __global__ void rms_norm_heads_head_major_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int num_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    size_t off = (size_t)h * head_dim;

    // Sequential per-element accumulator in thread 0 so the sum order
    // matches `residual::rms_norm_heads` on the host. See the matching
    // comment in `l2_norm_dim_major_f32` above for the why.
    if (tid == 0) {
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float v = x[off + d];
            ss += v * v;
        }
        smem[0] = ss;
    }
    __syncthreads();
    float inv_rms = rsqrtf(smem[0] / (float)head_dim + eps);
    for (int d = tid; d < head_dim; d += bdim) {
        out[off + d] = x[off + d] * inv_rms * weight[d];
    }
}
"#;

static DELTANET_MODULE: OnceLock<(
    std::sync::Arc<CudaModule>,
    CudaFunction,
    CudaFunction,
    CudaFunction,
    CudaFunction,
)> = OnceLock::new();

/// Public accessor for the (conv1d, deltanet, l2_norm, rms_norm)
/// kernel function references compiled from `DELTANET_SRC`. Used by
/// `qwen35_block::deltanet_postproj_step_cached` to chain the kernels
/// onto its own stream without redundant module loads.
pub(crate) fn module_functions(
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
    functions(drv)
}

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
    if let Some((_, conv, recurrence, l2_norm, rms_norm)) = DELTANET_MODULE.get() {
        return Ok((conv, recurrence, l2_norm, rms_norm));
    }

    // Disable fused multiply-add so the kernel's `acc += a * b`
    // produces the same fp32 rounding as the host's separate
    // multiply + add. With fmad enabled the accumulated drift
    // across 9 prompt + 5 decode positions × 48 layers × hundreds
    // of FMAs per layer breaks the multi-position parity check
    // against llama.cpp even though every individual call matches
    // CPU to ~1e-8 (E.6.A.6).
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(DELTANET_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile deltanet: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load deltanet module: {e:?}")))?;
    let conv = module
        .load_function("causal_conv1d_step_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load conv1d function: {e:?}")))?;
    let recurrence = module
        .load_function("deltanet_step_decay_first_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load deltanet function: {e:?}")))?;
    let l2_norm = module
        .load_function("l2_norm_dim_major_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load l2 norm function: {e:?}")))?;
    let rms_norm = module
        .load_function("rms_norm_heads_head_major_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load rms norm function: {e:?}")))?;
    let _ = DELTANET_MODULE.set((module, conv, recurrence, l2_norm, rms_norm));
    let (_, conv, recurrence, l2_norm, rms_norm) = DELTANET_MODULE.get().unwrap();
    Ok((conv, recurrence, l2_norm, rms_norm))
}

pub(crate) fn causal_conv1d_step(
    backend: &CudaBackend,
    weight: &[f32],
    state: &mut [f32],
    new_x: &[f32],
    d_conv: usize,
    conv_dim: usize,
) -> Result<Vec<f32>, CudaInitError> {
    let state_rows = d_conv.saturating_sub(1);
    if d_conv == 0
        || weight.len() != d_conv * conv_dim
        || state.len() != state_rows * conv_dim
        || new_x.len() != conv_dim
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid conv1d shape weight={} state={} new={} d_conv={d_conv} conv_dim={conv_dim}",
            weight.len(),
            state.len(),
            new_x.len()
        )));
    }

    let drv = backend.driver();
    let (func, _, _, _) = functions(drv)?;
    let weight_dev = drv.device_buf_from(weight)?;
    let mut state_dev = drv.device_buf_from(state)?;
    let new_dev = drv.device_buf_from(new_x)?;
    let mut out_dev = drv.device_alloc_uninit(conv_dim)?;

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
            .arg(&weight_dev)
            .arg(&mut state_dev)
            .arg(&new_dev)
            .arg(&mut out_dev)
            .arg(&d_conv_i)
            .arg(&conv_dim_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch conv1d: {e:?}")))?;
    }
    drv.sync()?;
    let out = drv.to_host(&out_dev)?;
    let next_state = drv.to_host(&state_dev)?;
    state.copy_from_slice(&next_state);
    Ok(out)
}

pub(crate) fn causal_conv1d_step_cached(
    backend: &CudaBackend,
    weight: &[f32],
    state: &mut [f32],
    new_x: &[f32],
    d_conv: usize,
    conv_dim: usize,
    sequence_pos: usize,
) -> Result<Vec<f32>, CudaInitError> {
    let state_rows = d_conv.saturating_sub(1);
    if d_conv == 0
        || weight.len() != d_conv * conv_dim
        || state.len() != state_rows * conv_dim
        || new_x.len() != conv_dim
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid cached conv1d shape weight={} state={} new={} d_conv={d_conv} conv_dim={conv_dim}",
            weight.len(),
            state.len(),
            new_x.len()
        )));
    }

    let drv = backend.driver();
    let (func, _, _, _) = functions(drv)?;
    let weight_dev = drv.device_buf_from(weight)?;
    let new_dev = drv.device_buf_from(new_x)?;
    let mut out_dev = drv.device_alloc_uninit(conv_dim)?;

    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((conv_dim as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_conv_i = d_conv as i32;
    let conv_dim_i = conv_dim as i32;
    backend.with_qwen35_state_device_buf(state, sequence_pos, |state_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(&weight_dev)
                .arg(state_dev)
                .arg(&new_dev)
                .arg(&mut out_dev)
                .arg(&d_conv_i)
                .arg(&conv_dim_i)
                .launch(cfg)
                .map_err(|e| {
                    CudaInitError::DriverMissing(format!("launch cached conv1d: {e:?}"))
                })?;
        }
        Ok(())
    })?;
    drv.sync()?;
    drv.to_host(&out_dev)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn deltanet_step(
    backend: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    log_g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    s: usize,
    h_k: usize,
    h_v: usize,
    block_gqa: bool,
) -> Result<Vec<f32>, CudaInitError> {
    if s == 0
        || h_k == 0
        || h_v == 0
        || q.len() != s * h_k
        || k.len() != s * h_k
        || v.len() != s * h_v
        || log_g.len() != h_v
        || beta.len() != h_v
        || state.len() != s * s * h_v
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid deltanet shape q={} k={} v={} log_g={} beta={} state={} s={s} h_k={h_k} h_v={h_v}",
            q.len(),
            k.len(),
            v.len(),
            log_g.len(),
            beta.len(),
            state.len()
        )));
    }

    let drv = backend.driver();
    let (_, func, _, _) = functions(drv)?;
    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k)?;
    let v_dev = drv.device_buf_from(v)?;
    let log_g_dev = drv.device_buf_from(log_g)?;
    let beta_dev = drv.device_buf_from(beta)?;
    let mut state_dev = drv.device_buf_from(state)?;
    let mut out_dev = drv.device_alloc_uninit(s * h_v)?;

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
            .arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
            .arg(&log_g_dev)
            .arg(&beta_dev)
            .arg(&mut state_dev)
            .arg(&mut out_dev)
            .arg(&s_i)
            .arg(&h_k_i)
            .arg(&h_v_i)
            .arg(&block_gqa_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch deltanet: {e:?}")))?;
    }
    drv.sync()?;
    let out = drv.to_host(&out_dev)?;
    let next_state = drv.to_host(&state_dev)?;
    state.copy_from_slice(&next_state);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn deltanet_step_cached(
    backend: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    log_g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    s: usize,
    h_k: usize,
    h_v: usize,
    sequence_pos: usize,
    block_gqa: bool,
) -> Result<Vec<f32>, CudaInitError> {
    if s == 0
        || h_k == 0
        || h_v == 0
        || q.len() != s * h_k
        || k.len() != s * h_k
        || v.len() != s * h_v
        || log_g.len() != h_v
        || beta.len() != h_v
        || state.len() != s * s * h_v
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid cached deltanet shape q={} k={} v={} log_g={} beta={} state={} s={s} h_k={h_k} h_v={h_v}",
            q.len(),
            k.len(),
            v.len(),
            log_g.len(),
            beta.len(),
            state.len()
        )));
    }

    let drv = backend.driver();
    let (_, func, _, _) = functions(drv)?;
    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k)?;
    let v_dev = drv.device_buf_from(v)?;
    let log_g_dev = drv.device_buf_from(log_g)?;
    let beta_dev = drv.device_buf_from(beta)?;
    let mut out_dev = drv.device_alloc_uninit(s * h_v)?;

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
    backend.with_qwen35_state_device_buf(state, sequence_pos, |state_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(&q_dev)
                .arg(&k_dev)
                .arg(&v_dev)
                .arg(&log_g_dev)
                .arg(&beta_dev)
                .arg(state_dev)
                .arg(&mut out_dev)
                .arg(&s_i)
                .arg(&h_k_i)
                .arg(&h_v_i)
                .arg(&block_gqa_i)
                .launch(cfg)
                .map_err(|e| {
                    CudaInitError::DriverMissing(format!("launch cached deltanet: {e:?}"))
                })?;
        }
        Ok(())
    })?;
    drv.sync()?;
    drv.to_host(&out_dev)
}

pub(crate) fn l2_normalize_per_head(
    backend: &CudaBackend,
    x: &[f32],
    head_dim: usize,
    n_heads: usize,
    eps: f32,
) -> Result<Vec<f32>, CudaInitError> {
    if head_dim == 0 || n_heads == 0 || x.len() != head_dim * n_heads {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid l2 shape x={} head_dim={head_dim} n_heads={n_heads}",
            x.len()
        )));
    }

    let drv = backend.driver();
    let (_, _, func, _) = functions(drv)?;
    let x_dev = drv.device_buf_from(x)?;
    let mut out_dev = drv.device_alloc_uninit(x.len())?;

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
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&head_dim_i)
            .arg(&n_heads_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch l2 norm: {e:?}")))?;
    }
    drv.sync()?;
    drv.to_host(&out_dev)
}

pub(crate) fn rms_norm_heads(
    backend: &CudaBackend,
    x: &[f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, CudaInitError> {
    if num_heads == 0
        || head_dim == 0
        || x.len() != num_heads * head_dim
        || weight.len() != head_dim
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid rms heads shape x={} weight={} num_heads={num_heads} head_dim={head_dim}",
            x.len(),
            weight.len()
        )));
    }

    let drv = backend.driver();
    let (_, _, _, func) = functions(drv)?;
    let x_dev = drv.device_buf_from(x)?;
    let mut out_dev = drv.device_alloc_uninit(x.len())?;

    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_heads_i = num_heads as i32;
    let head_dim_i = head_dim as i32;
    backend.with_norm_device_buf(weight, |weight_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(&x_dev)
                .arg(weight_dev)
                .arg(&mut out_dev)
                .arg(&num_heads_i)
                .arg(&head_dim_i)
                .arg(&eps)
                .launch(cfg)
                .map_err(|e| {
                    CudaInitError::DriverMissing(format!("launch rms heads norm: {e:?}"))
                })?;
        }
        Ok(())
    })?;
    drv.sync()?;
    drv.to_host(&out_dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_or_skip() -> Option<CudaBackend> {
        match CudaBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping CUDA deltanet test: {e}");
                None
            }
        }
    }

    #[test]
    fn causal_conv1d_matches_cpu_tiny() {
        let Some(backend) = gpu_or_skip() else { return };
        let d_conv = 3;
        let conv_dim = 5;
        let weight: Vec<f32> = (0..d_conv * conv_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let mut state: Vec<f32> = (0..(d_conv - 1) * conv_dim)
            .map(|i| (i as f32) * 0.02 - 0.3)
            .collect();
        let mut state_ref = state.clone();
        let new_x: Vec<f32> = (0..conv_dim).map(|i| i as f32 * 0.03 + 0.2).collect();

        let mut expected = vec![0.0_f32; conv_dim];
        for c in 0..conv_dim {
            for k in 0..(d_conv - 1) {
                expected[c] += weight[k * conv_dim + c] * state_ref[k * conv_dim + c];
            }
            expected[c] += weight[(d_conv - 1) * conv_dim + c] * new_x[c];
        }
        for k in 0..(d_conv - 2) {
            for c in 0..conv_dim {
                state_ref[k * conv_dim + c] = state_ref[(k + 1) * conv_dim + c];
            }
        }
        for c in 0..conv_dim {
            state_ref[(d_conv - 2) * conv_dim + c] = new_x[c];
        }

        let out = causal_conv1d_step(&backend, &weight, &mut state, &new_x, d_conv, conv_dim)
            .expect("cuda conv1d");
        for i in 0..conv_dim {
            assert!((out[i] - expected[i]).abs() < 1e-5, "out[{i}]");
        }
        for i in 0..state.len() {
            assert!((state[i] - state_ref[i]).abs() < 1e-6, "state[{i}]");
        }
    }

    #[test]
    fn deltanet_step_matches_cpu_tiny_decay_first() {
        let Some(backend) = gpu_or_skip() else { return };
        let s = 4;
        let h_k = 2;
        let h_v = 3;
        let q: Vec<f32> = (0..s * h_k).map(|i| 0.1 + i as f32 * 0.01).collect();
        let k: Vec<f32> = (0..s * h_k).map(|i| -0.2 + i as f32 * 0.015).collect();
        let v: Vec<f32> = (0..s * h_v).map(|i| 0.05 + i as f32 * 0.02).collect();
        let log_g: Vec<f32> = (0..h_v).map(|i| -0.1 - i as f32 * 0.03).collect();
        let beta: Vec<f32> = (0..h_v).map(|i| 0.4 + i as f32 * 0.05).collect();
        let mut state: Vec<f32> = (0..s * s * h_v).map(|i| -0.03 + i as f32 * 0.001).collect();
        let mut state_ref = state.clone();
        let mut expected = vec![0.0_f32; s * h_v];
        let scale_q = 1.0 / (s as f32).sqrt();

        for h in 0..h_v {
            let kh = h * h_k / h_v;
            let g = log_g[h].exp();
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

        let out = deltanet_step(
            &backend, &q, &k, &v, &log_g, &beta, &mut state, s, h_k, h_v, true,
        )
        .expect("cuda deltanet");
        for i in 0..out.len() {
            assert!((out[i] - expected[i]).abs() < 1e-5, "out[{i}]");
        }
        for i in 0..state.len() {
            assert!((state[i] - state_ref[i]).abs() < 1e-5, "state[{i}]");
        }
    }

    #[test]
    fn l2_normalize_per_head_matches_cpu_tiny() {
        let Some(backend) = gpu_or_skip() else { return };
        let head_dim = 5;
        let n_heads = 3;
        let eps = 1e-6_f32;
        let x: Vec<f32> = (0..head_dim * n_heads)
            .map(|i| -0.4 + i as f32 * 0.07)
            .collect();

        let mut expected = vec![0.0_f32; x.len()];
        for h in 0..n_heads {
            let mut sum_sq = 0.0_f32;
            for d in 0..head_dim {
                let v = x[d * n_heads + h];
                sum_sq += v * v;
            }
            let inv = 1.0 / (sum_sq + eps).sqrt();
            for d in 0..head_dim {
                let off = d * n_heads + h;
                expected[off] = x[off] * inv;
            }
        }

        let out =
            l2_normalize_per_head(&backend, &x, head_dim, n_heads, eps).expect("cuda l2 norm");
        for i in 0..out.len() {
            assert!((out[i] - expected[i]).abs() < 1e-6, "out[{i}]");
        }
    }

    #[test]
    fn rms_norm_heads_matches_cpu_tiny() {
        let Some(backend) = gpu_or_skip() else { return };
        let num_heads = 3;
        let head_dim = 5;
        let eps = 1e-6_f32;
        let x: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| -0.25 + i as f32 * 0.031)
            .collect();
        let weight: Vec<f32> = (0..head_dim).map(|i| 0.8 + i as f32 * 0.05).collect();

        let mut expected = vec![0.0_f32; x.len()];
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sum_sq = 0.0_f64;
            for d in 0..head_dim {
                let v = x[off + d] as f64;
                sum_sq += v * v;
            }
            let rms = (sum_sq / head_dim as f64 + eps as f64).sqrt() as f32;
            for d in 0..head_dim {
                expected[off + d] = x[off + d] / rms * weight[d];
            }
        }

        let out = rms_norm_heads(&backend, &x, &weight, num_heads, head_dim, eps)
            .expect("cuda rms heads norm");
        for i in 0..out.len() {
            assert!((out[i] - expected[i]).abs() < 1e-5, "out[{i}]");
        }
    }
}
