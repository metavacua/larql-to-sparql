//! Direct CUDA Q8_0 matrix-vector multiply (Phase F.6).
//!
//! Q8_0 is the simplest legacy quant: 32-element block of int8 values
//! with a single f16 scale (34 bytes per block). Per-element value:
//!   value = scale * (int8)q[i]
//!
//! F.4 first landed Q8_0 GPU support via host-dequant + cached f32
//! device buffer + cuBLAS sgemv. That works but quadruples the device
//! memory (int8+scale → f32: 1.3 → 5.4 GiB on Qwen3.6-35B-A3B). This
//! direct kernel reads the packed bytes in place, eliminating the f32
//! cache. Same pattern as `q5k_direct` / `q4k_direct`.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const Q8_0_BLOCK_ELEMS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const THREADS_PER_ROW: u32 = 128;
const ROWS_PER_BLOCK: u32 = 4;

const Q8_0_MATVEC_SRC: &str = r#"
__device__ float larql_f16_to_f32(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" __global__ void q8_0_matvec_direct(
    const unsigned char* q8,
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
    const unsigned char* row_base = q8 + (unsigned long long)row * blocks_per_row * 34ull;

    // Each block holds 32 elements; with bdim=128 we walk multiple
    // blocks per thread. Inner loop is just `scale * int8 * x`.
    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const unsigned char* block = row_base + sb * 34;
        float scale = larql_f16_to_f32((unsigned short)block[0] | ((unsigned short)block[1] << 8));
        const signed char* qs = (const signed char*)(block + 2);
        int x_base = sb * 32;
        for (int elem = tid; elem < 32; elem += bdim) {
            float q = (float)qs[elem];
            acc += scale * q * x[x_base + elem];
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

static Q8_0_MATVEC_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn q8_0_matvec_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q8_0_MATVEC_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(Q8_0_MATVEC_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile q8_0_matvec: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q8_0 module: {e:?}")))?;
    let func = module
        .load_function("q8_0_matvec_direct")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q8_0 function: {e:?}")))?;
    let _ = Q8_0_MATVEC_FUNC.set((module, func));
    let (_, f) = Q8_0_MATVEC_FUNC.get().unwrap();
    Ok(f)
}

/// Device-resident Q8_0 matvec. Returns the result on device. Reuses
/// the byte-cache via `with_q8_0_device_buf` (a new helper mirroring
/// `with_q5k_device_buf`) so repeated calls share the same upload.
pub(crate) fn matvec_device(
    backend: &CudaBackend,
    q8_0_data: &[u8],
    x_dev: &CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x_dev.len() != hidden || !hidden.is_multiple_of(Q8_0_BLOCK_ELEMS)
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q8_0 matvec shape rows={rows} hidden={hidden} x_len={}",
            x_dev.len()
        )));
    }
    let blocks_per_row = hidden / Q8_0_BLOCK_ELEMS;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(Q8_0_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q8_0 byte size overflow".to_string()))?;
    if q8_0_data.len() != expected {
        return Err(CudaInitError::DriverMissing(format!(
            "q8_0 byte length mismatch: got {}, expected {expected}",
            q8_0_data.len()
        )));
    }

    let drv = backend.driver();
    let func = q8_0_matvec_function(drv)?;
    let mut y_dev = drv.device_alloc_uninit(rows)?;
    let rows_i = rows as i32;
    let hidden_i = hidden as i32;
    let blocks_per_row_i = blocks_per_row as i32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_ROW, ROWS_PER_BLOCK, 1),
        shared_mem_bytes: THREADS_PER_ROW * ROWS_PER_BLOCK * std::mem::size_of::<f32>() as u32,
    };

    backend.with_q8_0_device_buf(q8_0_data, |q8_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(q8_dev)
                .arg(x_dev)
                .arg(&mut y_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&blocks_per_row_i)
                .launch(cfg)
                .map_err(|e| CudaInitError::DriverMissing(format!("launch q8_0_matvec: {e:?}")))?;
        }
        Ok(())
    })?;

    Ok(y_dev)
}

/// Host-input / host-output Q8_0 matvec. Uploads `x` once, calls the
/// device-resident kernel, syncs, dtohs the result.
pub(crate) fn matvec(
    backend: &CudaBackend,
    q8_0_data: &[u8],
    x: &[f32],
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x.len() != hidden || !hidden.is_multiple_of(Q8_0_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q8_0 matvec shape rows={rows} hidden={hidden} x_len={}",
            x.len()
        )));
    }
    let drv = backend.driver();
    let x_dev = drv.device_buf_from(x)?;
    let y_dev = matvec_device(backend, q8_0_data, &x_dev, rows, hidden)?;
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

    /// Bit-exact check vs CPU dequant + scalar dot. Uses a synthetic
    /// Q8_0 buffer with a couple of mixed-sign blocks per row so the
    /// signed-int8 path is exercised. Same shape as the q5k test.
    #[test]
    fn q8_0_matvec_matches_cpu_dequant_dot() {
        let Some(backend) = gpu_or_skip() else { return };
        let rows = 3;
        let blocks_per_row = 4;
        let hidden = blocks_per_row * Q8_0_BLOCK_ELEMS;
        let mut q8 = vec![0u8; rows * blocks_per_row * Q8_0_BLOCK_BYTES];
        // Use the same hash-stride that the q5k test uses to fill
        // weights; gives a deterministic but non-trivial layout.
        for (idx, byte) in q8.iter_mut().enumerate() {
            *byte = (((idx as u32).wrapping_mul(2654435761u32)) >> 24) as u8;
        }
        // Override every block's f16 scale to ~1.0 (0x3C00) so values
        // don't blow up. The int8 quants stay random.
        for sb in 0..(rows * blocks_per_row) {
            let off = sb * Q8_0_BLOCK_BYTES;
            q8[off] = 0x00;
            q8[off + 1] = 0x3C;
        }
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001 - 0.1).collect();

        let w_f32 = larql_models::quant::ggml::dequantize_q8_0(&q8, rows * hidden).unwrap();
        let mut expected = vec![0.0_f32; rows];
        for r in 0..rows {
            let mut acc = 0.0_f32;
            for k in 0..hidden {
                acc += w_f32[r * hidden + k] * x[k];
            }
            expected[r] = acc;
        }
        let got = matvec(&backend, &q8, &x, rows, hidden).expect("q8_0 matvec");
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
