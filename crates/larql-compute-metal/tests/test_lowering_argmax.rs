//! The two-pass device argmax matches the host scan exactly: first index
//! on ties, max at either edge, sizes that do not fill a block, a single
//! block, and a real vocabulary's worth of partials.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::head::{argmax_partials, ArgmaxScratch, ARGMAX_BLOCK};
use larql_compute_metal::MetalBackend;

/// The host contract: strict `>`, scanning upward, so the first maximum
/// wins a tie.
fn host_argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, x)| {
                if *x > bv {
                    (i, *x)
                } else {
                    (bi, bv)
                }
            },
        )
        .0 as u32
}

fn device_argmax(gpu: &MetalBackend, v: &[f32]) -> u32 {
    let x = gpu.lowering_upload(v).expect("upload");
    let parts = argmax_partials(v.len());
    let pv = gpu.lowering_scratch(parts);
    let pi = gpu.lowering_scratch(parts);
    let out = gpu.lowering_scratch(1);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_argmax(
        enc,
        &x,
        v.len(),
        &ArgmaxScratch {
            partial_vals: &pv,
            partial_idx: &pi,
            out: &out,
        },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // The output is a u32 in an f32-sized slot; read the bits.
    let bits = gpu.lowering_readback(&out, 1).expect("readback")[0].to_bits();
    for b in [x, pv, pi, out] {
        gpu.recycle_lowering_scratch(b);
    }
    bits
}

/// Deterministic pseudo-random logits in a realistic range.
fn logits(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 20_000) as f32 * 0.001 - 10.0
        })
        .collect()
}

#[test]
fn device_argmax_matches_host_across_shapes() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for (n, seed) in [
        (1usize, 1u32),
        (7, 2),
        (255, 3),
        (256, 4),
        (257, 5),
        (ARGMAX_BLOCK - 1, 6),
        (ARGMAX_BLOCK, 7),
        (ARGMAX_BLOCK + 1, 8),
        (201_088, 9),
        (262_144, 10),
    ] {
        let v = logits(n, seed);
        assert_eq!(
            device_argmax(&gpu, &v),
            host_argmax(&v),
            "n={n} seed={seed}"
        );
    }
}

#[test]
fn device_argmax_takes_the_first_of_tied_maxima() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let n = 3 * ARGMAX_BLOCK + 17;
    let mut v = logits(n, 11);
    // Same maximum in three blocks, and twice inside one block.
    let top = 99.0;
    for &i in &[
        ARGMAX_BLOCK + 5,
        ARGMAX_BLOCK + 900,
        2 * ARGMAX_BLOCK + 1,
        n - 1,
    ] {
        v[i] = top;
    }
    assert_eq!(device_argmax(&gpu, &v), (ARGMAX_BLOCK + 5) as u32);
    assert_eq!(device_argmax(&gpu, &v), host_argmax(&v));
    // Max at the very first and very last element.
    let mut first = logits(n, 12);
    first[0] = 1e6;
    assert_eq!(device_argmax(&gpu, &first), 0);
    let mut last = logits(n, 13);
    last[n - 1] = 1e6;
    assert_eq!(device_argmax(&gpu, &last), (n - 1) as u32);
    // All equal: index 0, as the host scan gives.
    let flat = vec![-3.5f32; 2 * ARGMAX_BLOCK + 3];
    assert_eq!(device_argmax(&gpu, &flat), 0);
    assert_eq!(host_argmax(&flat), 0);
}

#[test]
fn partial_count_covers_every_element() {
    assert_eq!(argmax_partials(1), 1);
    assert_eq!(argmax_partials(ARGMAX_BLOCK), 1);
    assert_eq!(argmax_partials(ARGMAX_BLOCK + 1), 2);
    assert_eq!(argmax_partials(262_144), 64);
}
