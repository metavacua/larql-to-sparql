use super::matvec_f32::{decode_q4k_superblock_into, decode_q6k_superblock_into, dot_256_f32};

/// Amortised Q4_K × f32 matmul: `out[s, r] = sum_k W[r, k] * X[s, k]`.
///
/// `w` is `[rows, hidden]` Q4_K (144-byte super-blocks), `x` is
/// `[seq, hidden]` f32 row-major, `out` is `[seq, rows]` f32 row-major.
///
/// The win over the alternatives prefill could use:
/// * vs `seq ×` [`q4k_matvec_into`]: each weight super-block is decoded to
///   f32 **once** and reused across all `seq` columns, so the Q4_K bytes
///   are read once instead of `seq` times (the per-position matvec loop
///   re-reads every weight `seq` times — ~50× slower than f32 sgemm at
///   long context).
/// * vs dequant-whole-layer + BLAS sgemm: never materialises the full f32
///   weight matrix (4× the bytes of Q4_K), so it skips the per-layer
///   dequant that dominates short-prompt prefill.
///
/// Shape errors (zero dims, `hidden` not a multiple of 256, short `w`)
/// zero the output rather than panic, mirroring [`q4k_matvec_into`].
pub fn q4k_matmul_into(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    rows: usize,
    hidden: usize,
    seq: usize,
) {
    kquant_matmul_into(
        out,
        x,
        w,
        rows,
        hidden,
        seq,
        144,
        decode_q4k_superblock_into,
    );
}

/// Amortised Q6_K × f32 matmul — the Q6_K twin of [`q4k_matmul_into`], used for
/// the `down_proj` (Q6_K is the default down format in a q4k vindex). Same
/// `[seq, rows]` output contract; reads the Q6_K weight once instead of
/// dequantising the whole matrix to f32 first.
pub fn q6k_matmul_into(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    rows: usize,
    hidden: usize,
    seq: usize,
) {
    kquant_matmul_into(
        out,
        x,
        w,
        rows,
        hidden,
        seq,
        210,
        decode_q6k_superblock_into,
    );
}

/// Shared amortised k-quant matmul: `out[s, r] = sum_k W[r, k] * X[s, k]`,
/// `out` is `[seq, rows]` row-major. Each weight super-block is decoded to f32
/// **once** via `decode` and FMA'd across all `seq` columns, so the quantised
/// weight is read once (not `seq×`) and never fully materialised as f32. q4k
/// and q6k differ only in `block_bytes` and the per-block `decode`.
///
/// Shape errors (zero dims, `hidden` not a 256-multiple, short `w`) zero the
/// output (defensive, mirroring `q4k_matvec_into`); callers keep the dims valid
/// (the `use_q4k_ffn` gate, the down padding).
#[allow(clippy::too_many_arguments)]
fn kquant_matmul_into(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    rows: usize,
    hidden: usize,
    seq: usize,
    block_bytes: usize,
    decode: impl Fn(&[u8], usize, usize, &mut [f32; 256]) + Sync,
) {
    debug_assert_eq!(out.len(), seq * rows);
    debug_assert_eq!(x.len(), seq * hidden);
    const ELEMS_PER_BLOCK: usize = 256;
    // Shape errors zero the output (defensive, mirrors `q4k_matvec_into` — see
    // `q4k_matmul_rejects_non_multiple_of_256`); callers keep the dims valid.
    if rows == 0 || seq == 0 || hidden == 0 || !hidden.is_multiple_of(ELEMS_PER_BLOCK) {
        out.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let n_blocks = hidden / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * block_bytes;
    if w.len() < rows * row_bytes {
        out.iter_mut().for_each(|v| *v = 0.0);
        return;
    }

    // Accumulate into a row-major scratch `[rows, seq]` (each row's `seq`
    // outputs contiguous) so the hot loop is row-parallel and write-local;
    // transpose to the `[seq, rows]` contract afterwards. The transpose
    // touches `rows * seq` f32 once — cheap next to the matmul, and it
    // never re-reads the weights.
    let mut tmp = vec![0.0f32; rows * seq];
    const CHUNK_ROWS: usize = 32;
    let x_ref = x;
    let w_ref = w;
    let decode_ref = &decode;
    crate::cpu::spin_pool::par_chunks_mut(&mut tmp, CHUNK_ROWS * seq, |chunk_idx, chunk_slots| {
        let row_base_chunk = chunk_idx * CHUNK_ROWS;
        let mut wf = [0.0f32; ELEMS_PER_BLOCK];
        for local_r in 0..CHUNK_ROWS {
            let r = row_base_chunk + local_r;
            if r >= rows {
                break;
            }
            let out_row = &mut chunk_slots[local_r * seq..local_r * seq + seq];
            let row_base = r * row_bytes;
            for sb in 0..n_blocks {
                decode_ref(w_ref, row_base, sb, &mut wf);
                let x_off = sb * ELEMS_PER_BLOCK;
                for (s, slot) in out_row.iter_mut().enumerate() {
                    let xs = &x_ref[s * hidden + x_off..s * hidden + x_off + ELEMS_PER_BLOCK];
                    // `wf` was decoded once above; dot it against this column.
                    // NEON (8 f32x4 accumulators to hide FMA latency) on
                    // aarch64, portable multi-accumulator fallback elsewhere.
                    *slot += dot_256_f32(&wf, xs);
                }
            }
        }
    });

    for r in 0..rows {
        for s in 0..seq {
            out[s * rows + r] = tmp[r * seq + s];
        }
    }
}
