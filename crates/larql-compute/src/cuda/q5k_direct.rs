//! Direct CUDA Q5_K matrix-vector multiply.
//!
//! Q5_K uses the GGUF 176-byte super-block layout from
//! `larql_models::quant::ggml::q5_k`:
//!   bytes  0-1:   f16 `d`  (global scale)
//!   bytes  2-3:   f16 `dmin` (global min)
//!   bytes  4-15:  12 bytes of packed (6-bit scale, 6-bit min) pairs
//!                 — same `get_scale_min_k4` packing as Q4_K
//!   bytes 16-47:  qh (32 bytes of high bits, 1 per element)
//!   bytes 48-175: qs (128 bytes of low nibbles, 2 nibbles per byte)
//!
//! Each 32-element sub-block (8 per super-block) has its own
//! (scale, min) pair. The per-element value:
//!   q5(elem) = ((qs[outer*32 + (elem%32)] >> ((elem/32)%2)*4) & 0x0F)
//!             | (((qh[elem%32] >> (elem/32)) & 1) << 4)
//!   out(elem) = d * scale[sub] * q5(elem) - dmin * min[sub]
//!
//! This module wires up the CUDA matvec equivalent — one block per
//! output row, each thread accumulating partial dot product over
//! multiple super-blocks, then a parallel reduction across the
//! block's threads.
//!
//! Discovered as the dominant Qwen3.6-27B-Q4_K_S decode cost in
//! Phase E.6.A.9 fine-profile: ssm_out (1661 ms / token, 59 %),
//! ffn_down (797 ms / token, 28 %), DeltaNet attn_qkv (~190 ms /
//! token), and the full-attn attn_k/v projections are all Q5_K
//! and were falling back to CPU rayon dequant-per-row. This kernel
//! moves them onto the GPU.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const Q5K_BLOCK_ELEMS: usize = 256;
const Q5K_BLOCK_BYTES: usize = 176;
const THREADS_PER_ROW: u32 = 128;
const ROWS_PER_BLOCK: u32 = 4;

const Q5K_MATVEC_SRC: &str = r#"
__device__ float larql_f16_to_f32(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// 6-bit scale and min unpacking — identical to Q4_K's `k_quant_get_scale_min_k4`.
__device__ unsigned char q5k_scale(const unsigned char* packed, int j) {
    if (j < 4) return packed[j] & 0x3fu;
    return (packed[j + 4] & 0x0fu) | ((packed[j - 4] >> 6) << 4);
}

__device__ unsigned char q5k_min(const unsigned char* packed, int j) {
    if (j < 4) return packed[j + 4] & 0x3fu;
    return (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
}

extern "C" __global__ void q5k_matvec_direct(
    const unsigned char* q5k,
    const float* x,
    float* y,
    int rows,
    int hidden,
    int blocks_per_row
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int row_lane = threadIdx.y;
    extern __shared__ float smem[];
    float* row_smem = smem + row_lane * bdim;

    float acc = 0.0f;
    const unsigned char* row_base = q5k + (unsigned long long)row * blocks_per_row * 176ull;

    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const unsigned char* block = row_base + sb * 176;
        float d = larql_f16_to_f32((unsigned short)block[0] | ((unsigned short)block[1] << 8));
        float dmin = larql_f16_to_f32((unsigned short)block[2] | ((unsigned short)block[3] << 8));
        const unsigned char* packed = block + 4;   // 12 bytes scale+min
        const unsigned char* qh = block + 16;      // 32 bytes high-bits
        const unsigned char* qs = block + 48;      // 128 bytes low-nibbles
        int x_base = sb * 256;

        for (int elem = tid; elem < 256; elem += bdim) {
            int sub = elem >> 5;          // sub-block 0..7
            int l = elem & 31;            // position in sub-block 0..31
            int outer = sub >> 1;         // qs byte-pair group 0..3
            int half = sub & 1;           // 0 = low nibble, 1 = high nibble
            unsigned char qs_byte = qs[outer * 32 + l];
            unsigned char nibble = (half == 0) ? (qs_byte & 0x0fu) : (qs_byte >> 4);
            unsigned char hi_bit = ((qh[l] >> sub) & 1u) << 4;
            float q5 = (float)(nibble | hi_bit);
            float sc = d * (float)q5k_scale(packed, sub);
            float mn = dmin * (float)q5k_min(packed, sub);
            float w = sc * q5 - mn;
            acc += w * x[x_base + elem];
        }
    }

    row_smem[tid] = acc;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) row_smem[tid] += row_smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = row_smem[0];
}
"#;

static Q5K_MATVEC_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn q5k_matvec_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q5K_MATVEC_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(Q5K_MATVEC_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile q5k_matvec: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q5k module: {e:?}")))?;
    let func = module
        .load_function("q5k_matvec_direct")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q5k function: {e:?}")))?;
    let _ = Q5K_MATVEC_FUNC.set((module, func));
    let (_, f) = Q5K_MATVEC_FUNC.get().unwrap();
    Ok(f)
}

/// Device-resident Q5_K matvec. Returns the result on device.
pub(crate) fn matvec_device(
    backend: &CudaBackend,
    q5k_data: &[u8],
    x_dev: &CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let drv = backend.driver();
    matvec_device_on_stream(backend, q5k_data, x_dev, rows, hidden, &drv.stream)
}

/// `cuda-moe-multistream`: variant of [`matvec_device`] on a
/// caller-chosen stream. See `q4k_direct::matvec_device_on_stream` for
/// the cross-stream invariants.
pub(crate) fn matvec_device_on_stream(
    backend: &CudaBackend,
    q5k_data: &[u8],
    x_dev: &CudaSlice<f32>,
    rows: usize,
    hidden: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<CudaSlice<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x_dev.len() != hidden || !hidden.is_multiple_of(Q5K_BLOCK_ELEMS)
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q5k matvec shape rows={rows} hidden={hidden} x_len={}",
            x_dev.len()
        )));
    }
    let blocks_per_row = hidden / Q5K_BLOCK_ELEMS;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(Q5K_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q5k byte size overflow".to_string()))?;
    if q5k_data.len() != expected {
        return Err(CudaInitError::DriverMissing(format!(
            "q5k byte length mismatch: got {}, expected {expected}",
            q5k_data.len()
        )));
    }

    let drv = backend.driver();
    let func = q5k_matvec_function(drv)?;
    let mut y_dev = unsafe { stream.alloc::<f32>(rows) }
        .map_err(|e| CudaInitError::DriverMissing(format!("alloc y_dev on stream: {e:?}")))?;
    let rows_i = rows as i32;
    let hidden_i = hidden as i32;
    let blocks_per_row_i = blocks_per_row as i32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_ROW, ROWS_PER_BLOCK, 1),
        shared_mem_bytes: THREADS_PER_ROW * ROWS_PER_BLOCK * std::mem::size_of::<f32>() as u32,
    };

    backend.with_q5k_device_buf(q5k_data, |q5k_dev| {
        unsafe {
            stream
                .launch_builder(func)
                .arg(q5k_dev)
                .arg(x_dev)
                .arg(&mut y_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&blocks_per_row_i)
                .launch(cfg)
                .map_err(|e| CudaInitError::DriverMissing(format!("launch q5k_matvec: {e:?}")))?;
        }
        Ok(())
    })?;

    Ok(y_dev)
}

/// Host-input / host-output Q5_K matvec. Uploads `x` once, calls
/// the device-resident kernel, syncs, dtohs result.
pub(crate) fn matvec(
    backend: &CudaBackend,
    q5k_data: &[u8],
    x: &[f32],
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x.len() != hidden || !hidden.is_multiple_of(Q5K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q5k matvec shape rows={rows} hidden={hidden} x_len={}",
            x.len()
        )));
    }
    let drv = backend.driver();
    let x_dev = drv.device_buf_from(x)?;
    let y_dev = matvec_device(backend, q5k_data, &x_dev, rows, hidden)?;
    drv.sync()?;
    drv.to_host(&y_dev)
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

    /// Bit-exact check against the CPU `dequantize_q5_k` followed by a
    /// scalar dot product. Catches kernel byte-layout / index bugs
    /// before they reach the inference parity tests.
    #[test]
    fn q5k_matvec_matches_cpu_dequant_dot() {
        let Some(backend) = gpu_or_skip() else { return };
        // 2 rows × 2 super-blocks per row (hidden=512). Each super-block
        // gets a different but structured byte pattern so each scale,
        // qh and qs index path is exercised.
        let rows = 2;
        let blocks_per_row = 2;
        let hidden = blocks_per_row * Q5K_BLOCK_ELEMS;
        let mut q5k = vec![0u8; rows * blocks_per_row * Q5K_BLOCK_BYTES];
        for (idx, byte) in q5k.iter_mut().enumerate() {
            *byte = ((idx as u32 * 2654435761u32) >> 24) as u8;
        }
        // Force every block's d=1.0 (f16) and dmin=0.5 so dequant
        // produces a meaningful range. f16 1.0 = 0x3C00; f16 0.5 = 0x3800.
        for sb in 0..(rows * blocks_per_row) {
            let off = sb * Q5K_BLOCK_BYTES;
            q5k[off] = 0x00;
            q5k[off + 1] = 0x3C;
            q5k[off + 2] = 0x00;
            q5k[off + 3] = 0x38;
        }
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001 - 0.2).collect();

        // CPU reference: dequant the whole [rows*hidden] weight matrix
        // then plain dot product per row.
        let w_f32 = larql_models::quant::ggml::dequantize_q5_k(&q5k, rows * hidden).unwrap();
        let mut expected = vec![0.0_f32; rows];
        for r in 0..rows {
            let mut acc = 0.0_f32;
            for k in 0..hidden {
                acc += w_f32[r * hidden + k] * x[k];
            }
            expected[r] = acc;
        }
        let got = matvec(&backend, &q5k, &x, rows, hidden).expect("q5k matvec");
        for r in 0..rows {
            let rel = (got[r] - expected[r]).abs() / expected[r].abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "row {r}: got {} vs expected {} (rel {rel:.4e})",
                got[r],
                expected[r],
            );
        }
    }
}
