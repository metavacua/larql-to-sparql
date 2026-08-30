//! Does the number of distinct ALLOCATIONS a command buffer references
//! cost time, with everything else held fixed?
//!
//! The QKV packing rung (2026-08-21) gained 0.195 ms/token end-to-end
//! where its isolated probe predicted 0.038 — and the one thing packing
//! changed that the probe could not express is resource topology: 6
//! allocations per layer became 2, i.e. 144 distinct buffers per token
//! became 48. That is a hypothesis, not a mechanism, and this ladder is
//! its control.
//!
//! Held fixed across every rung:
//!   * total bytes read (48 distinct matrix copies, ~317 MB — past the
//!     SLC, so this is DRAM traffic, not a cache measurement)
//!   * dispatch count (48), shader (x2 via the 1-segment encode), grid
//!   * arithmetic: every copy holds the same values, so every dispatch
//!     computes the same result whatever it is bound through
//!
//! Varied: how many MTLBuffer objects those 48 copies live in —
//!   48 (one per dispatch) → 24 → 12 → 6 → 3 → 2 → 1
//! With A allocations each holds 48/A copies and a dispatch binds its
//! copy by byte offset, so bytes, addresses touched and dispatch order
//! are unchanged; only buffer CARDINALITY moves.
//!
//! Read the slope, not one rung: if latency falls monotonically as
//! allocations coalesce, resource cardinality is a real cost dimension
//! and model layout should minimise it. A flat ladder refutes the
//! packing attribution and sends that 0.195 ms back to the QKV shapes.

use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::Nvfp4Segment;
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

/// Rows per copy; with K below one copy is ~6.6 MB.
const ROWS: usize = 4096;
const K: usize = 2880;
/// Copies, and therefore dispatches — fixed across the ladder.
const COPIES: usize = 48;
/// Allocation counts to walk. Every entry must divide `COPIES`.
const ALLOCATIONS: [usize; 7] = [48, 24, 12, 6, 3, 2, 1];
const REPS: usize = 5;

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    let values: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, ROWS, K).expect("quantise");
    let copy_p = m.packed.len();
    let copy_s = m.scales.len();
    let bytes_per_copy = (copy_p + copy_s) as f64;
    let total_bytes = bytes_per_copy * COPIES as f64;

    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let out = gpu.lowering_scratch(ROWS);

    println!(
        "allocation-cardinality ladder: {COPIES} dispatches × {:.1} MB = {:.0} MB total, \
         [{ROWS}, {K}] NVFP4",
        bytes_per_copy / 1e6,
        total_bytes / 1e6
    );
    println!(
        "{:>6}  {:>9}  {:>8}  {:>9}",
        "allocs", "µs/disp", "GB/s", "ms/48"
    );

    // The rungs walk in one direction, so ANY monotone drift (thermal,
    // charge state, another process arriving) forges a monotone ladder.
    // The opening rung is re-measured last and the two must agree, or
    // the block is void — the KV-B1 sweep was lost to exactly this.
    let mut first = None;
    let mut last = None;
    for allocs in ALLOCATIONS.iter().copied().chain([ALLOCATIONS[0]]) {
        assert!(
            COPIES.is_multiple_of(allocs),
            "allocation count must divide COPIES"
        );
        let per_alloc = COPIES / allocs;
        // Build `allocs` buffers, each holding `per_alloc` copies back to
        // back. Bytes are identical to every other rung; only the number
        // of objects they live in differs.
        let mut packed_bufs = Vec::with_capacity(allocs);
        let mut scales_bufs = Vec::with_capacity(allocs);
        let mut hold: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(allocs);
        for _ in 0..allocs {
            let mut p = Vec::with_capacity(copy_p * per_alloc);
            let mut s = Vec::with_capacity(copy_s * per_alloc);
            for _ in 0..per_alloc {
                p.extend_from_slice(&m.packed);
                s.extend_from_slice(&m.scales);
            }
            hold.push((p, s));
        }
        for (p, s) in &hold {
            packed_bufs.push(gpu.lowering_weight(p));
            scales_bufs.push(gpu.lowering_weight(s));
        }
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in 0..COPIES {
                // Dispatch c reads copy c, wherever it lives: buffer
                // c / per_alloc, at slot c % per_alloc inside it.
                let (a, slot) = (c / per_alloc, c % per_alloc);
                gpu.encode_nvfp4_matvec_segments(
                    enc,
                    &xb,
                    K,
                    &[Nvfp4Segment {
                        packed: &packed_bufs[a],
                        packed_offset: (slot * copy_p) as u64,
                        scales: &scales_bufs[a],
                        scales_offset: (slot * copy_s) as u64,
                        tensor_scale: m.tensor_scale,
                        out: &out,
                        out_offset: 0,
                        n: ROWS,
                    }],
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best = best.min(gpu_span_ms(&cmd));
        }
        let per_disp = best * 1e3 / COPIES as f64;
        println!(
            "{allocs:>6}  {per_disp:>9.2}  {:>8.0}  {best:>9.3}",
            total_bytes / (best / 1e3) / 1e9
        );
        first.get_or_insert(best);
        last = Some(best);
    }
    if let (Some(open), Some(close)) = (first, last) {
        let drift = (close - open) / open * 100.0;
        println!(
            "\nbracket: {} allocs opened {open:.3} ms, closed {close:.3} ms — drift {drift:+.1}%",
            ALLOCATIONS[0]
        );
        if drift.abs() > 5.0 {
            println!(
                "VOID: the bracket moved more than the ladder claims to measure. \
                 Every row above is drift, not cardinality."
            );
        } else {
            println!(
                "Bracket holds. Read the SLOPE across rungs, not any single row; \
                 a flat ladder refutes allocation cardinality as a cost dimension."
            );
        }
    }
}
