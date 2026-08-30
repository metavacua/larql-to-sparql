//! A-5b: the segmented NVFP4 GEMV (Q+K+V or gate+up in one dispatch) is
//! bit-identical to the per-matrix `x2` dispatches it replaces — rows
//! straddling a segment boundary, absent third segment, output offsets.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::{
    LoweredMatrix, MatvecOperands, MatvecTarget, Nvfp4Kernel, Nvfp4Segment,
};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

struct Mat {
    m: nvfp4::Nvfp4Matrix,
    n: usize,
}

fn mat(n: usize, k: usize, seed: usize) -> Mat {
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i * 17 + seed) % 977) as f32 / 977.0) - 0.5)
        .collect();
    Mat {
        m: nvfp4::quantize(&values, n, k).expect("quantise"),
        n,
    }
}

/// Separate x2 dispatches vs one segmented dispatch, same x, same outs.
fn compare(gpu: &MetalBackend, k: usize, mats: &[Mat], offsets: &[u64]) {
    let x: Vec<f32> = (0..k).map(|i| (i % 11) as f32 * 0.02 - 0.1).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let packed: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.packed))
        .collect();
    let scales: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.scales))
        .collect();
    let mut results: Vec<Vec<Vec<f32>>> = Vec::new();
    for fused in [false, true] {
        // Outputs sized for the offset plus the rows.
        let outs: Vec<_> = mats
            .iter()
            .zip(offsets)
            .map(|(m, off)| gpu.lowering_scratch(m.n + (*off as usize) / 4))
            .collect();
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if fused {
            let segs: Vec<Nvfp4Segment<'_>> = (0..mats.len())
                .map(|i| Nvfp4Segment {
                    packed: &packed[i],
                    packed_offset: 0,
                    scales: &scales[i],
                    scales_offset: 0,
                    tensor_scale: mats[i].m.tensor_scale,
                    out: &outs[i],
                    out_offset: offsets[i],
                    n: mats[i].n,
                })
                .collect();
            gpu.encode_nvfp4_matvec_segments(enc, &xb, k, &segs);
        } else {
            for i in 0..mats.len() {
                gpu.encode_nvfp4_kernel(
                    Nvfp4Kernel::X2,
                    enc,
                    &MatvecOperands {
                        packed: &packed[i],
                        scales: &scales[i],
                        x: &xb,
                        out: &outs[i],
                        out_offset: offsets[i],
                        n: mats[i].n,
                        k,
                    },
                    mats[i].m.tensor_scale,
                );
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let read: Vec<Vec<f32>> = outs
            .iter()
            .zip(mats)
            .zip(offsets)
            .map(|((o, m), off)| {
                let all = gpu
                    .lowering_readback(o, m.n + (*off as usize) / 4)
                    .expect("readback");
                all[(*off as usize) / 4..].to_vec()
            })
            .collect();
        for o in outs {
            gpu.recycle_lowering_scratch(o);
        }
        results.push(read);
    }
    gpu.recycle_lowering_scratch(xb);
    for (i, (a, b)) in results[0].iter().zip(&results[1]).enumerate() {
        assert_eq!(a, b, "segment {i} of {}: fused != separate", mats.len());
        assert!(a.iter().all(|v| v.is_finite()));
    }
}

/// The QKV packing rung: segments as OFFSETS into one shared allocation
/// must be bit-identical to per-buffer segments AND to separate x2
/// dispatches — same kernel, different base bindings.
fn compare_packed(gpu: &MetalBackend, k: usize, mats: &[Mat]) {
    let x: Vec<f32> = (0..k).map(|i| (i % 11) as f32 * 0.02 - 0.1).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let row_p = k / 16 * 8;
    let row_s = k / 16;
    let mut packed_all = Vec::new();
    let mut scales_all = Vec::new();
    let mut offs = Vec::new();
    for m in mats {
        assert!(
            packed_all.len().is_multiple_of(16) && scales_all.len().is_multiple_of(16),
            "fixture violates the bind alignment the contract documents"
        );
        offs.push((packed_all.len() as u64, scales_all.len() as u64));
        packed_all.extend_from_slice(&m.m.packed[..m.n * row_p]);
        scales_all.extend_from_slice(&m.m.scales[..m.n * row_s]);
    }
    let packed_buf = gpu.lowering_weight(&packed_all);
    let scales_buf = gpu.lowering_weight(&scales_all);
    let per_p: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.packed))
        .collect();
    let per_s: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.m.scales))
        .collect();
    let total: usize = mats.iter().map(|m| m.n).sum();
    let mut results: Vec<Vec<f32>> = Vec::new();
    for mode in 0..3 {
        let out = gpu.lowering_scratch(total);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        match mode {
            // separate x2 dispatches, per-matrix buffers
            0 => {
                let mut off = 0u64;
                for (i, m) in mats.iter().enumerate() {
                    gpu.encode_nvfp4_kernel(
                        Nvfp4Kernel::X2,
                        enc,
                        &MatvecOperands {
                            packed: &per_p[i],
                            scales: &per_s[i],
                            x: &xb,
                            out: &out,
                            out_offset: off,
                            n: m.n,
                            k,
                        },
                        m.m.tensor_scale,
                    );
                    off += (m.n as u64) * 4;
                }
            }
            // one fused dispatch, segments = offsets into the shared pack
            1 => {
                let mut off = 0u64;
                let segs: Vec<Nvfp4Segment<'_>> = mats
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let s = Nvfp4Segment {
                            packed: &packed_buf,
                            packed_offset: offs[i].0,
                            scales: &scales_buf,
                            scales_offset: offs[i].1,
                            tensor_scale: m.m.tensor_scale,
                            out: &out,
                            out_offset: off,
                            n: m.n,
                        };
                        off += (m.n as u64) * 4;
                        s
                    })
                    .collect();
                gpu.encode_nvfp4_matvec_segments(enc, &xb, k, &segs);
            }
            // single-segment dispatches at non-zero offsets — the
            // encode_matvec fallback shape for a sliced matrix
            _ => {
                let mut off = 0u64;
                for (i, m) in mats.iter().enumerate() {
                    gpu.encode_nvfp4_matvec_segments(
                        enc,
                        &xb,
                        k,
                        &[Nvfp4Segment {
                            packed: &packed_buf,
                            packed_offset: offs[i].0,
                            scales: &scales_buf,
                            scales_offset: offs[i].1,
                            tensor_scale: m.m.tensor_scale,
                            out: &out,
                            out_offset: off,
                            n: m.n,
                        }],
                    );
                    off += (m.n as u64) * 4;
                }
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        results.push(gpu.lowering_readback(&out, total).expect("readback"));
        gpu.recycle_lowering_scratch(out);
    }
    gpu.recycle_lowering_scratch(xb);
    assert_eq!(results[0], results[1], "packed fused != separate x2");
    assert_eq!(
        results[0], results[2],
        "packed single-segment != separate x2"
    );
    assert!(results[0].iter().all(|v| v.is_finite()));
}

#[test]
fn packed_allocation_segments_match_bit_for_bit() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // gpt-oss geometry (4096+512+512 @ 2880) plus an awkward odd-rows
    // set whose boundaries still meet the 16-byte bind alignment.
    let k = 2880;
    compare_packed(&gpu, k, &[mat(4096, k, 1), mat(512, k, 2), mat(512, k, 3)]);
    let k2 = 2816;
    compare_packed(
        &gpu,
        k2,
        &[mat(4097, k2, 4), mat(31, k2, 5), mat(514, k2, 6)],
    );
}

/// The folded-residual o-proj on a SLICED matrix must read the slice's
/// own rows too.
///
/// This is the path the packed attention layout actually exposed a bug
/// on: `encode_nvfp4_matvec_residual` takes `MatvecOperands`, which has
/// no offset fields, so under packing it bound o-proj at offset 0 and
/// computed the Q projection's rows — with the residual added on top, so
/// the output stayed finite and plausible. It is invisible on gpt-oss
/// (an o_bias sends that layer down the unfused branch) and live on any
/// two-norm layer without one, which is why it needs its own gate rather
/// than inheriting the plain sliced test's.
#[test]
fn residual_fused_matvec_honours_a_sliced_matrix_offset() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2880;
    let (rows_a, rows_b) = (512, 512);
    let a = mat(rows_a, k, 21);
    let b = mat(rows_b, k, 22);
    let row_p = k / 16 * 8;
    let row_s = k / 16;
    let mut packed_all = a.m.packed[..rows_a * row_p].to_vec();
    let mut scales_all = a.m.scales[..rows_a * row_s].to_vec();
    let (b_off_p, b_off_s) = (packed_all.len() as u64, scales_all.len() as u64);
    packed_all.extend_from_slice(&b.m.packed[..rows_b * row_p]);
    scales_all.extend_from_slice(&b.m.scales[..rows_b * row_s]);

    let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.03 - 0.1).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    // A row-DEPENDENT residual: a constant would be added to every row
    // equally and could not distinguish one slice from another.
    let resid: Vec<f32> = (0..rows_b).map(|i| i as f32 * 0.5 - 64.0).collect();
    let rb = gpu.lowering_upload(&resid).expect("residual");
    let packed_buf = gpu.lowering_weight(&packed_all);
    let scales_buf = gpu.lowering_weight(&scales_all);
    let own_p = gpu.lowering_weight(&b.m.packed);
    let own_s = gpu.lowering_weight(&b.m.scales);

    let run = |po: u64, so: u64, own: bool| -> Vec<f32> {
        let out = gpu.lowering_scratch(rows_b);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let op = MatvecOperands {
            packed: if own { &own_p } else { &packed_buf },
            scales: if own { &own_s } else { &scales_buf },
            x: &xb,
            out: &out,
            out_offset: 0,
            n: rows_b,
            k,
        };
        gpu.encode_nvfp4_matvec_residual_sliced(enc, &op, b.m.tensor_scale, &rb, po, so);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let v = gpu.lowering_readback(&out, rows_b).expect("readback");
        gpu.recycle_lowering_scratch(out);
        v
    };

    let control = run(0, 0, true);
    let sliced = run(b_off_p, b_off_s, false);
    let trap = run(0, 0, false);
    gpu.recycle_lowering_scratch(xb);
    gpu.recycle_lowering_scratch(rb);

    assert_eq!(
        control, sliced,
        "residual-fused sliced matvec != own-buffer control"
    );
    assert_ne!(
        control, trap,
        "offset-0 bind agreed with the slice — this fixture cannot detect \
         a dropped offset, so the parity assertion above is vacuous"
    );
    // The residual really is folded in, not silently dropped: with the
    // residual removed the result must differ.
    assert!(control
        .iter()
        .zip(&resid)
        .any(|(o, r)| (o - r).abs() > 1e-6));
}

/// `encode_matvec` on a SLICED matrix (non-zero offsets) must read the
/// slice's own rows, not the allocation's front. The flat kernels bind at
/// offset 0, so the lowering routes a sliced matrix through the segmented
/// form; without that, a packed Q/K/V would silently serve Q's rows for
/// all three. The negative control is the point: binding the same buffer
/// at offset 0 must DISAGREE, or the test proves nothing.
#[test]
fn encode_matvec_honours_a_sliced_matrix_offset() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2880;
    let (rows_a, rows_b) = (512, 512);
    let a = mat(rows_a, k, 11);
    let b = mat(rows_b, k, 12);
    let row_p = k / 16 * 8;
    let row_s = k / 16;
    let mut packed_all = a.m.packed[..rows_a * row_p].to_vec();
    let mut scales_all = a.m.scales[..rows_a * row_s].to_vec();
    let (b_off_p, b_off_s) = (packed_all.len() as u64, scales_all.len() as u64);
    packed_all.extend_from_slice(&b.m.packed[..rows_b * row_p]);
    scales_all.extend_from_slice(&b.m.scales[..rows_b * row_s]);

    let x: Vec<f32> = (0..k).map(|i| (i % 11) as f32 * 0.02 - 0.1).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let packed_buf = gpu.lowering_weight(&packed_all);
    let scales_buf = gpu.lowering_weight(&scales_all);
    // Control: B loaded as its own whole-buffer matrix.
    let own_p = gpu.lowering_weight(&b.m.packed);
    let own_s = gpu.lowering_weight(&b.m.scales);

    // (packed_offset, scales_offset) — the slice, then the offset-0 trap.
    let run = |po: u64, so: u64, own: bool| -> Vec<f32> {
        let out = gpu.lowering_scratch(rows_b);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let m = LoweredMatrix::Nvfp4 {
            packed: if own { &own_p } else { &packed_buf },
            packed_offset: po,
            scales: if own { &own_s } else { &scales_buf },
            scales_offset: so,
            tensor_scale: b.m.tensor_scale,
        };
        gpu.encode_matvec(
            enc,
            &m,
            &MatvecTarget {
                x: &xb,
                out: &out,
                out_offset: 0,
                n: rows_b,
                k,
            },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let v = gpu.lowering_readback(&out, rows_b).expect("readback");
        gpu.recycle_lowering_scratch(out);
        v
    };

    let control = run(0, 0, true);
    let sliced = run(b_off_p, b_off_s, false);
    let trap = run(0, 0, false);
    gpu.recycle_lowering_scratch(xb);

    assert_eq!(
        control, sliced,
        "sliced encode_matvec != own-buffer control"
    );
    assert_ne!(
        control, trap,
        "offset-0 bind agreed with the slice — the fixture cannot detect a \
         dropped offset, so the parity assertion above is vacuous"
    );
}

#[test]
fn three_segments_match_separate_x2_dispatches_bit_for_bit() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2816;
    // Odd first segment so a lane's row pair straddles the Q|K boundary;
    // K and V rows land at a cache-slot offset like the real lowering.
    let mats = [mat(4097, k, 1), mat(2048, k, 2), mat(2048, k, 3)];
    compare(&gpu, k, &mats, &[0, 4 * 2048 * 3, 4 * 2048 * 3]);
}

#[test]
fn two_segments_gate_up_match_and_third_is_absent() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let k = 2560;
    let mats = [mat(8192, k, 4), mat(8192, k, 5)];
    compare(&gpu, k, &mats, &[0, 0]);
    // A single segment is just x2.
    let one = [mat(37, k, 6)];
    compare(&gpu, k, &one, &[0]);
}
