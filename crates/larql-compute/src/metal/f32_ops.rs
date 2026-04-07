//! f32 matmul operations via Metal compute shaders.
//!
//! Tiled sgemm (32×32) for large matmuls, falls back to CPU for small ones.
//! The FLOP threshold is set by calibration.

use std::ffi::c_void;
use metal::*;
use ndarray::{Array2, ArrayView2};

use super::buffers::BufferCache;

/// Dispatch parameters for f32 matmul.
pub struct F32Ops {
    pub sgemm_pipeline: ComputePipelineState,
    pub transb_pipeline: ComputePipelineState,
}

impl F32Ops {
    /// C = A × B  (A: [m,k], B: [k,n], C: [m,n])
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_notrans(
        &self,
        queue: &CommandQueue,
        bufs: &BufferCache,
        a_data: &[f32],
        b_data: &[f32],
        m: usize, n: usize, k: usize,
    ) -> Vec<f32> {
        let buf_a = bufs.get_f32(a_data);
        let buf_b = bufs.get_f32(b_data);
        let buf_c = bufs.output((m * n * 4) as u64);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        Self::encode_static(&self.sgemm_pipeline, enc, &buf_a, &buf_b, &buf_c, m, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(&buf_c, m * n)
    }

    /// C = A × B^T  (A: [m,k], B: [n,k], C: [m,n])
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_transb(
        &self,
        queue: &CommandQueue,
        bufs: &BufferCache,
        a_data: &[f32],
        b_data: &[f32],
        m: usize, n: usize, k: usize,
    ) -> Vec<f32> {
        let buf_a = bufs.get_f32(a_data);
        let buf_b = bufs.get_f32(b_data);
        let buf_c = bufs.output((m * n * 4) as u64);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        Self::encode_static(&self.transb_pipeline, enc, &buf_a, &buf_b, &buf_c, m, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(&buf_c, m * n)
    }

    /// Encode one matmul dispatch into a command encoder.
    /// Public for use by pipeline builders (full-layer Metal pipeline).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_static(
        pipeline: &ComputePipelineState,
        encoder: &ComputeCommandEncoderRef,
        buf_a: &Buffer, buf_b: &Buffer, buf_c: &Buffer,
        m: usize, n: usize, k: usize,
    ) {
        let m_val = m as u32;
        let n_val = n as u32;
        let k_val = k as u32;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(buf_a), 0);
        encoder.set_buffer(1, Some(buf_b), 0);
        encoder.set_buffer(2, Some(buf_c), 0);
        encoder.set_bytes(3, 4, &m_val as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &k_val as *const u32 as *const c_void);

        let tg = MTLSize::new(32, 32, 1);
        let grid = MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(32) as u64, 1);
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// f32 matmul with automatic GPU/CPU routing.
    pub fn matmul(
        &self, queue: &CommandQueue, bufs: &BufferCache,
        a: ArrayView2<f32>, b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        if 2 * m * n * k < flop_threshold { return a.dot(&b); }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => { a_owned = a.as_standard_layout().into_owned(); a_owned.as_slice().unwrap() }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => { b_owned = b.as_standard_layout().into_owned(); b_owned.as_slice().unwrap() }
        };

        let c = self.dispatch_notrans(queue, bufs, a_data, b_data, m, n, k);
        Array2::from_shape_vec((m, n), c).unwrap()
    }

    /// f32 matmul_transb with automatic GPU/CPU routing.
    pub fn matmul_transb(
        &self, queue: &CommandQueue, bufs: &BufferCache,
        a: ArrayView2<f32>, b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[0];
        if 2 * m * n * k < flop_threshold { return a.dot(&b.t()); }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => { a_owned = a.as_standard_layout().into_owned(); a_owned.as_slice().unwrap() }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => { b_owned = b.as_standard_layout().into_owned(); b_owned.as_slice().unwrap() }
        };

        let c = self.dispatch_transb(queue, bufs, a_data, b_data, m, n, k);
        Array2::from_shape_vec((m, n), c).unwrap()
    }
}
