//! Pre-allocated per-shape scratch for GPU expert dispatch.
//!
//! Split out of `moe_dispatch.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::buffers::BufferCache;
use crate::MetalBackend;
use metal::*;

/// Pre-allocated scratch for the whole MoE decode loop.
///
/// All sizes are determined by `(top_k, hidden, intermediate_size)` of the
/// first MoE layer, which is constant across MoE layers in the architectures
/// we currently target (Gemma 4 26B A4B). Sizing assumes Q4_K weights with
/// 256-element super-blocks, 144 bytes per row-block.
///
/// `act_buf` is sized to `top_k × inter_padded` and zero-initialised so the
/// `inter_padded - inter` padding columns of every expert's strided slice
/// contribute nothing through the down projection — required when
/// `moe.intermediate_size` is not a multiple of 256 (e.g. Gemma 4 26B's 2112
/// → inter_padded 2304).
pub struct MoeScratch {
    pub(crate) top_k: usize,
    pub(crate) inter: usize,
    pub(crate) inter_padded: usize,
    pub(crate) hidden: usize,
    /// The expert store's weight format — sizes the row strides below and
    /// selects the matvec pipelines at dispatch. Q4_K and Q6_K only.
    pub(crate) format: larql_compute::QuantFormat,
    /// Gate/up STORED row width in elements (block-padded by the writer;
    /// GPT-OSS 2880 → 3072, block-multiple hidden sizes unchanged).
    pub(crate) weight_cols: usize,
    pub(crate) row_bytes: usize,
    pub(crate) down_row_bytes: usize,

    pub(crate) gate_buf: Buffer,
    pub(crate) up_buf: Buffer,
    pub(crate) down_bufs: Vec<Buffer>,

    pub(crate) x_buf: Buffer,
    pub(crate) g_out: Buffer,
    pub(crate) u_out: Buffer,
    pub(crate) act_buf: Buffer,
    pub(crate) expert_outs: Buffer,
    /// De-interleaved per-selected-expert gate/up bias rows, `top_k × inter`
    /// f32 each. Always allocated (small) so the activation kernel's bias
    /// slots bind a real buffer; `has_bias` gates the read.
    pub(crate) gate_bias_buf: Buffer,
    pub(crate) up_bias_buf: Buffer,
    /// Selected experts' down-bias rows for the GPU weighted combine,
    /// `top_k × hidden` f32, slot-aligned with `expert_outs`. Staged by
    /// the inline-combine path only; the CPU-combine paths never read it.
    pub(crate) down_bias_staged: Buffer,
}

// `Buffer` is `Send + Sync` on its own; the Metal types we hold here mirror
// the rest of `MetalBackend` (single-process, single-device).  Stamping it so
// `larql-server` can stash a `MoeScratch` inside `Arc<AppState>` without
// fighting the borrow checker.
unsafe impl Send for MoeScratch {}
unsafe impl Sync for MoeScratch {}

impl MoeScratch {
    /// Public constructor — used by `larql-server`'s shard expert path so it
    /// can preallocate one scratch per (hidden, intermediate, top_k) shape on
    /// startup and reuse it for every incoming RPC.
    pub fn new_public(backend: &MetalBackend, top_k: usize, hidden: usize, inter: usize) -> Self {
        Self::new(
            &backend.bufs,
            top_k,
            hidden,
            inter,
            larql_compute::QuantFormat::Q4_K,
            hidden,
        )
    }

    /// Format-aware public constructor: `format` sizes the row strides and
    /// selects the matvec pipelines; `weight_cols` is the gate/up STORED
    /// row width (`MoeLayerWeights::gate_up_cols`).
    pub fn new_public_with_format(
        backend: &MetalBackend,
        top_k: usize,
        hidden: usize,
        inter: usize,
        format: larql_compute::QuantFormat,
        weight_cols: usize,
    ) -> Self {
        Self::new(&backend.bufs, top_k, hidden, inter, format, weight_cols)
    }

    pub(crate) fn new(
        bufs: &BufferCache,
        top_k: usize,
        hidden: usize,
        inter: usize,
        format: larql_compute::QuantFormat,
        weight_cols: usize,
    ) -> Self {
        let (block, bytes_per_block) = format
            .packed_block_layout()
            .expect("MoE expert scratch requires a block format (Q4_K/Q6_K)");
        let inter_padded = inter.div_ceil(block) * block;
        // Row strides from the STORE's own geometry: gate/up rows at the
        // writer-padded `weight_cols` (a truncating `hidden / block` here
        // mis-strided every non-block-multiple model), down at
        // `inter_padded` — both in this format's bytes per super-block.
        debug_assert!(weight_cols.is_multiple_of(block));
        let row_bytes = (weight_cols / block) * bytes_per_block;
        let down_row_bytes = (inter_padded / block) * bytes_per_block;

        let gate_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let up_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let down_bufs: Vec<Buffer> = (0..top_k)
            .map(|_| bufs.output((hidden * down_row_bytes) as u64))
            .collect();

        let x_buf = bufs.output((weight_cols * 4) as u64);
        let g_out = bufs.output((top_k * inter * 4) as u64);
        let u_out = bufs.output((top_k * inter * 4) as u64);
        let act_buf = bufs.output((top_k * inter_padded * 4) as u64);
        let expert_outs = bufs.output((top_k * hidden * 4) as u64);

        let gate_bias_buf = bufs.output((top_k * inter * 4) as u64);
        let up_bias_buf = bufs.output((top_k * inter * 4) as u64);
        let down_bias_staged = bufs.output((top_k * hidden * 4) as u64);

        // Zero the padding tails once. GEGLU writes only the first `inter`
        // floats of each expert's `inter_padded`-strided slice, so the
        // remaining `inter_padded - inter` floats stay zero forever. The
        // x_buf tail (`weight_cols - hidden`) likewise stays zero so the
        // writer's row padding contributes nothing to any dot product.
        unsafe {
            let ptr = act_buf.contents() as *mut f32;
            std::ptr::write_bytes(ptr, 0, top_k * inter_padded);
            let xp = x_buf.contents() as *mut f32;
            std::ptr::write_bytes(xp, 0, weight_cols);
        }

        Self {
            top_k,
            inter,
            inter_padded,
            hidden,
            format,
            weight_cols,
            row_bytes,
            down_row_bytes,
            gate_buf,
            up_buf,
            down_bufs,
            x_buf,
            g_out,
            u_out,
            act_buf,
            expert_outs,
            gate_bias_buf,
            up_bias_buf,
            down_bias_staged,
        }
    }
}
