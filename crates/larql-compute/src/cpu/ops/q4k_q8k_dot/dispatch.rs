#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_asm::{q4k_q8k_matvec_asm_v3, use_asm_kernel};
#[cfg(target_arch = "x86_64")]
use super::q4k_avx2::q4k_q8k_matvec_avx2;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_neon::q4k_q8k_matvec_neon;
use super::q4k_scalar::q4k_q8k_matvec_scalar;
use super::q8k_activation::Q8KActivation;

/// Public entry point: dispatches to NEON on aarch64, scalar elsewhere.
/// Caller pre-quantises `x` once via `quantize_x_to_q8k` (cost is amortised
/// across all rows of the same matvec, and across all K active experts that
/// share `h_norm`).
pub fn q4k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // 2-row variant tried 2026-05-01 — bit-exact (`q8k_matvec_2row_matches_single_row_bit_exact`)
        // but bench-neutral on M3 Max: per-thread is BW-bound on the
        // per-row Q4_K weight stream (1.1 MB at 82 µs ≈ 14 GB/s), and
        // sharing the small activation Q8K (256 B) across 2 rows didn't
        // free real DRAM bandwidth.  Kept as `q4k_q8k_matvec_neon_2row`
        // for future hardware where ILP may dominate over BW.
        // (NB: the "BW-bound" read was overturned 2026-06-02 — the kernel
        // is compute/issue-bound, see `docs/q4k-decode-kernel.md`.)
        //
        // C12: opt-in hand-asm kernel (`LARQL_Q4K_ASM=1`). Bit-exact with
        // the intrinsic path. v2 (2026-06-11) folds ALL the per-super-block
        // Rust glue into the asm block (vectorised scale/min unpack, sum2,
        // hardware fcvt + epilogue) — the decomposition bench measured the
        // glue at 19.2 cyc/SB vs the asm block's 16.3.
        if use_asm_kernel() {
            q4k_q8k_matvec_asm_v3(out, q8k_x, w, rows, cols);
        } else {
            q4k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: runtime check guarantees AVX2 availability.
        unsafe { q4k_q8k_matvec_avx2(out, q8k_x, w, rows, cols) };
        return;
    }
    #[allow(unreachable_code)]
    q4k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

/// Row-chunked **parallel** Q4_K / Q6_K × Q8_K matvec — the single source for
/// every quantized projection on the decode path (attention Q/K/V/O, the dense
/// FFN gate/up/down slab, and the lm_head vocab projection). Fills `out[0..rows]`
/// with each weight row's dot against `q8k_x`.
///
/// `format` is `"Q4_K"` or `"Q6_K"` (selects the per-row kernel and the byte
/// stride). The per-row kernel ([`q4k_q8k_matvec_into`] / [`q6k_q8k_matvec_into`])
/// is single-threaded; this wraps it across output-row chunks and routes the
/// chunks through the spin pool when enabled, else rayon — so the whole decode
/// path shares one parallelism strategy.
///
/// This centralizes what were four byte-identical `par_chunks_mut` copies
/// (larql-compute `cached.rs`, larql-inference `cached.rs`, and the two lm_head
/// blocks in larql-inference `dense.rs`) — the "consolidation hazard" twins.
/// `out.len()` must be `>= rows`; rows beyond `rows` are left untouched.
pub fn q4k_q8k_matvec_parallel(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    format: &str,
) {
    // Format dispatch goes through the `FormatRoute` registry: the tag
    // resolves to its per-row kernel and its packed super-block geometry
    // (no local `(cols/256)*144`/`*210` re-spelling). The truncating
    // `cols / block_elems` is preserved (k-quant cols are always a
    // 256-multiple in practice).
    //
    // Unknown-format contract (`quant_route`): this is a kernel entry
    // point, so a format with no Q8K matvec route — or a weight slab too
    // short for the requested geometry — PANICS instead of leaving `out`
    // silently zeroed. A wrong-but-plausible logit vector is the
    // dec-readiness review's §1 failure class; every production caller
    // already gates on `layer_supports_direct_matvec`-style probes, so
    // these panics are unreachable outside genuine corruption.
    let fmt = crate::QuantFormat::from_registry_tag(format)
        .unwrap_or_else(|| panic!("q4k_q8k_matvec_parallel: unknown quant format tag {format:?}"));
    let kernel = fmt.route().q8k_matvec.unwrap_or_else(|| {
        panic!("q4k_q8k_matvec_parallel: no Q8K matvec kernel for {format} (Q4_K/Q6_K only)")
    });
    let (block_elems, block_bytes) = fmt
        .packed_block_layout()
        .expect("q8k-matvec formats always have a packed block layout");
    let bytes_per_row = (cols / block_elems) * block_bytes;
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(
        bytes.len() >= rows * bytes_per_row,
        "q4k_q8k_matvec_parallel: {format} weight slab too short: {} bytes < {rows} rows × {bytes_per_row} bytes/row",
        bytes.len(),
    );
    const CHUNK_ROWS: usize = 32;
    crate::cpu::spin_pool::par_chunks_mut(&mut out[..rows], CHUNK_ROWS, |chunk_idx, chunk| {
        let row_start = chunk_idx * CHUNK_ROWS;
        let chunk_len = chunk.len().min(rows.saturating_sub(row_start));
        if chunk_len == 0 {
            return;
        }
        let w_chunk = &bytes[row_start * bytes_per_row..(row_start + chunk_len) * bytes_per_row];
        kernel(&mut chunk[..chunk_len], q8k_x, w_chunk, chunk_len, cols);
    });
}
