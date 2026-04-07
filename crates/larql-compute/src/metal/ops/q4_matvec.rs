//! Q4×Q8 matrix-vector dispatch.
//!
//! scores[N] = Q4[N, K] @ Q8_x[K]
//!
//! Dispatches the optimised simdgroup shader: 8 rows per threadgroup,
//! shared memory for Q8 input, simd_sum reduction.

use std::ffi::c_void;
use metal::*;

use crate::metal::buffers::BufferCache;
use crate::metal::shaders::q4_matvec as shader;

/// Dispatch a single Q4 matvec on GPU.
///
/// - `q4_data`: packed Q4_0 weights (cached, mmap-backed)
/// - `q8_x`: pre-quantized input vector (transient)
/// - `q8_scales`: per-block Q8 scales (transient)
/// - Returns: f32 scores vector
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    queue: &CommandQueue,
    bufs: &BufferCache,
    pipeline: &ComputePipelineState,
    q4_data: &[u8],
    q8_x: &[i8],
    q8_scales: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let buf_q4 = bufs.get_bytes(q4_data);
    let buf_q8 = bufs.transient_from_i8(q8_x);
    let buf_scales = bufs.transient_from_f32(q8_scales);
    let buf_out = bufs.output((num_rows * 4) as u64);

    let n_val = num_rows as u32;
    let k_val = hidden as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc, pipeline, &buf_q4, &buf_q8, &buf_scales, &buf_out, n_val, k_val, num_rows);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    crate::metal::buffers::read_buffer_f32(&buf_out, num_rows)
}

/// Encode a Q4 matvec dispatch into an existing command encoder.
/// Used by batched operations to chain multiple dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buf_q4: &Buffer,
    buf_q8: &Buffer,
    buf_scales: &Buffer,
    buf_out: &Buffer,
    n_val: u32,
    k_val: u32,
    num_rows: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buf_q4), 0);
    enc.set_buffer(1, Some(buf_q8), 0);
    enc.set_buffer(2, Some(buf_scales), 0);
    enc.set_buffer(3, Some(buf_out), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);

    let num_tgs = (num_rows as u64).div_ceil(shader::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(shader::THREADS_PER_TG, 1, 1),
    );
}
