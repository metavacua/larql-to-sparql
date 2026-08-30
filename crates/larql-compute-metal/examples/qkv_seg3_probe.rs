//! Is the fused QKV dispatch's THREE-SEGMENT form slower than its parts?
//!
//! The ledger read attn.proj in situ at ~208–296 GB/s across runs while
//! the segmented kernel benched at ~370 — but that bench used ONE
//! segment. The real QKV dispatch resolves three ragged segments
//! (gpt-oss: 4096 + 512 + 512 rows) per row pair. Arms, same bytes:
//!
//!   A: three separate x2 dispatches (Q, K, V)      — no segment resolve
//!   B: one seg3 dispatch (production QKV fusion)   — the real form
//!   C: one x2 dispatch over a 5120-row matrix      — resolve-free upper bound
//!   D: one seg3 dispatch, segments = OFFSETS into one allocation
//!      (the loader-packing rung: same kernel, one base address range)
//!
//! B vs C prices the segment machinery; B vs A prices fusion net of α;
//! D vs B prices allocation packing alone, and D vs C is the residual a
//! kernel change would have to chase.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{MatvecOperands, Nvfp4Kernel, Nvfp4Segment};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const K: usize = 2880;
const ROWS: [usize; 3] = [4096, 512, 512];
const CHAIN: usize = 48;
const REPS: usize = 5;

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    let total: usize = ROWS.iter().sum();
    let values: Vec<f32> = (0..total * K)
        .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
        .collect();
    // One quantisation of the whole stack; segments slice its rows.
    let m = nvfp4::quantize(&values, total, K).expect("quantise");
    let bytes = (m.packed.len() + m.scales.len()) as f64;
    let copies = ((256usize << 20) / (bytes as usize)).clamp(1, CHAIN);
    let groups = K / 16;
    let row_p = groups * 8;
    let row_s = groups;
    // Per-copy buffers: whole stack + per-segment slices of the same bytes.
    let mut whole_p = Vec::new();
    let mut whole_s = Vec::new();
    let mut segs_p: Vec<Vec<metal::Buffer>> = Vec::new();
    let mut segs_s: Vec<Vec<metal::Buffer>> = Vec::new();
    let packed_copies: Vec<Vec<u8>> = (0..copies).map(|_| m.packed.clone()).collect();
    let scales_copies: Vec<Vec<u8>> = (0..copies).map(|_| m.scales.clone()).collect();
    for c in 0..copies {
        whole_p.push(gpu.lowering_weight(&packed_copies[c]));
        whole_s.push(gpu.lowering_weight(&scales_copies[c]));
        let mut sp = Vec::new();
        let mut ss = Vec::new();
        let mut row0 = 0usize;
        for n in ROWS {
            sp.push(gpu.lowering_weight(&packed_copies[c][row0 * row_p..(row0 + n) * row_p]));
            ss.push(gpu.lowering_weight(&scales_copies[c][row0 * row_s..(row0 + n) * row_s]));
            row0 += n;
        }
        segs_p.push(sp);
        segs_s.push(ss);
    }
    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let out = gpu.lowering_scratch(total);

    let run = |mode: u8| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in 0..CHAIN {
                let b = c % copies;
                match mode {
                    0 => {
                        // three separate x2 dispatches
                        let mut off = 0u64;
                        for (i, n) in ROWS.iter().enumerate() {
                            gpu.encode_nvfp4_kernel(
                                Nvfp4Kernel::X2,
                                enc,
                                &MatvecOperands {
                                    packed: &segs_p[b][i],
                                    scales: &segs_s[b][i],
                                    x: &xb,
                                    out: &out,
                                    out_offset: off,
                                    n: *n,
                                    k: K,
                                },
                                m.tensor_scale,
                            );
                            off += (*n as u64) * 4;
                        }
                    }
                    1 => {
                        // one seg3 dispatch — the production QKV form
                        let mut off = 0u64;
                        let segments: Vec<Nvfp4Segment<'_>> = ROWS
                            .iter()
                            .enumerate()
                            .map(|(i, n)| {
                                let s = Nvfp4Segment {
                                    packed: &segs_p[b][i],
                                    packed_offset: 0,
                                    scales: &segs_s[b][i],
                                    scales_offset: 0,
                                    tensor_scale: m.tensor_scale,
                                    out: &out,
                                    out_offset: off,
                                    n: *n,
                                };
                                off += (*n as u64) * 4;
                                s
                            })
                            .collect();
                        gpu.encode_nvfp4_matvec_segments(enc, &xb, K, &segments);
                    }
                    3 => {
                        // one seg3 dispatch, all three segments offsets
                        // into ONE allocation — the loader-packing rung
                        let mut off = 0u64;
                        let mut row0 = 0usize;
                        let segments: Vec<Nvfp4Segment<'_>> = ROWS
                            .iter()
                            .map(|n| {
                                let s = Nvfp4Segment {
                                    packed: &whole_p[b],
                                    packed_offset: (row0 * row_p) as u64,
                                    scales: &whole_s[b],
                                    scales_offset: (row0 * row_s) as u64,
                                    tensor_scale: m.tensor_scale,
                                    out: &out,
                                    out_offset: off,
                                    n: *n,
                                };
                                off += (*n as u64) * 4;
                                row0 += n;
                                s
                            })
                            .collect();
                        gpu.encode_nvfp4_matvec_segments(enc, &xb, K, &segments);
                    }
                    _ => {
                        // one flat x2 over all rows — resolve-free bound
                        gpu.encode_nvfp4_kernel(
                            Nvfp4Kernel::X2,
                            enc,
                            &MatvecOperands {
                                packed: &whole_p[b],
                                scales: &whole_s[b],
                                x: &xb,
                                out: &out,
                                out_offset: 0,
                                n: total,
                                k: K,
                            },
                            m.tensor_scale,
                        );
                    }
                }
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best = best.min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
        }
        best
    };

    let a = run(0);
    let bt = run(1);
    // Parity gate: the packed arm must reproduce the production seg3
    // arm bitwise — same kernel, same rows, different base bindings. A
    // wrong offset would otherwise be TIMED, silently.
    let b_out = gpu.lowering_readback(&out, total).expect("readback B");
    let c = run(2);
    let d = run(3);
    let d_out = gpu.lowering_readback(&out, total).expect("readback D");
    assert!(
        b_out
            .iter()
            .zip(&d_out)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "packed arm diverged from seg3 — offset defect"
    );
    println!(
        "QKV [{}+{}+{}, {K}], {:.1} MB",
        ROWS[0],
        ROWS[1],
        ROWS[2],
        bytes / 1e6
    );
    for (name, v) in [
        ("A 3 dispatches", a),
        ("B seg3 fused  ", bt),
        ("C flat x2     ", c),
        ("D seg3 packed ", d),
    ] {
        println!("{name}: {v:>7.1} µs  {:>4.0} GB/s", bytes / (v / 1e6) / 1e9);
    }
    println!(
        "segment-resolve cost (B−C): {:.1} µs; fusion net (A−B): {:.1} µs; \
         packing recovers (B−D): {:.1} µs; kernel residual (D−C): {:.1} µs",
        bt - c,
        a - bt,
        bt - d,
        d - c
    );
}
