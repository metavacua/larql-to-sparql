//! Effective-bandwidth probe for the f16 gemv at Glimmer decode shapes.
//!
//! Not a correctness gate: prints GB/s per shape so a decode number can
//! be attributed to kernel efficiency vs. platform state (battery power
//! caps, thermal throttling). Run explicitly with `--ignored`.

use larql_compute::backend::MatMul;
use larql_compute_metal::MetalBackend;
use std::time::Instant;

fn probe(metal: &MetalBackend, label: &str, n: usize, k: usize, iters: usize) {
    let w: Vec<u8> = vec![0x3C; n * k * 2]; // 1.0f16 everywhere
    let x: Vec<f32> = vec![0.5; k];
    // Warm: first touch + pipeline + buffer-cache entry.
    let _ = metal.f16_gemv_force(&w, &x, n, k).unwrap();
    let started = Instant::now();
    for _ in 0..iters {
        let out = metal.f16_gemv_force(&w, &x, n, k).unwrap();
        assert_eq!(out.len(), n);
    }
    let per_call = started.elapsed().as_secs_f64() / iters as f64;
    let gb = (n * k * 2) as f64 / 1e9;
    println!(
        "{label:14} [{n:6} x {k}]  {:8.2} ms/call  {:6.1} GB/s effective",
        per_call * 1e3,
        gb / per_call,
    );
}

#[test]
#[ignore = "diagnostic probe, run explicitly"]
fn f16_gemv_bandwidth_at_decode_shapes() {
    let metal = MetalBackend::new().expect("Metal device");
    probe(&metal, "ffn (up/down)", 23_040, 6656, 20);
    probe(&metal, "q/gate/o", 5120, 6656, 20);
    probe(&metal, "k/v", 1024, 6656, 20);
    probe(&metal, "lm head", 202_048, 6656, 5);
}

/// The decode-shaped working set: many DISTINCT resident buffers used
/// round-robin, the way a real step walks 52 layers of distinct
/// weights. If per-call time jumps versus the single-buffer probe, the
/// cost is buffer residency / page-table pressure, not the kernel.
#[test]
#[ignore = "diagnostic probe, run explicitly"]
fn f16_gemv_bandwidth_over_many_distinct_buffers() {
    let metal = MetalBackend::new().expect("Metal device");
    let device = metal::Device::system_default().expect("Metal device");
    println!(
        "recommendedMaxWorkingSetSize: {:.1} GB",
        device.recommended_max_working_set_size() as f64 / 1e9
    );
    let (n, k) = (23_040usize, 6656usize);
    let copies: usize = std::env::var("PROBE_COPIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20); // ~0.3 GB each
                        // PROBE_UNALIGNED=1 offsets each slice by one byte so `get_bytes`
                        // takes its copy branch: the weights live in driver-allocated
                        // buffers instead of no-copy-wrapped client pages. Discriminates
                        // GPU page-mapping cost from everything else at large working sets.
    let unaligned = std::env::var("PROBE_UNALIGNED").is_ok();
    let backing: Vec<Vec<u8>> = (0..copies)
        .map(|_| vec![0x3C; n * k * 2 + usize::from(unaligned)])
        .collect();
    let weights: Vec<&[u8]> = backing
        .iter()
        .map(|b| &b[usize::from(unaligned)..])
        .collect();
    let x: Vec<f32> = vec![0.5; k];
    for w in &weights {
        let _ = metal.f16_gemv_force(w, &x, n, k).unwrap();
    }
    let iters = if copies > 50 { 1 } else { 3 };
    let started = Instant::now();
    for _ in 0..iters {
        for w in &weights {
            let out = metal.f16_gemv_force(w, &x, n, k).unwrap();
            assert_eq!(out.len(), n);
        }
    }
    let per_call = started.elapsed().as_secs_f64() / (iters * copies) as f64;
    let gb = (n * k * 2) as f64 / 1e9;
    println!(
        "distinct x{copies}  [{n:6} x {k}]  {:8.2} ms/call  {:6.1} GB/s effective",
        per_call * 1e3,
        gb / per_call,
    );
}

/// All distinct buffers in ONE submission via `f16_gemv_multi`. If this
/// restores full bandwidth at a working set where serialised calls
/// crawl, the cost is the driver un-wiring idle buffers between
/// submissions — and one-command-buffer-per-token is the fix.
#[test]
#[ignore = "diagnostic probe, run explicitly"]
fn f16_gemv_bandwidth_one_submission_over_many_buffers() {
    let metal = MetalBackend::new().expect("Metal device");
    let (n, k) = (23_040usize, 6656usize);
    let copies: usize = std::env::var("PROBE_COPIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let backing: Vec<Vec<u8>> = (0..copies).map(|_| vec![0x3C; n * k * 2]).collect();
    let x: Vec<f32> = vec![0.5; k];
    let mats: Vec<(&[u8], usize, usize)> = backing.iter().map(|b| (b.as_slice(), n, k)).collect();
    let _ = metal.f16_gemv_multi(&mats, &x).unwrap(); // warm
    let iters = 3;
    let started = Instant::now();
    for _ in 0..iters {
        let out = metal.f16_gemv_multi(&mats, &x).unwrap();
        assert_eq!(out.len(), copies);
    }
    let per_call = started.elapsed().as_secs_f64() / (iters * copies) as f64;
    let gb = (n * k * 2) as f64 / 1e9;
    println!(
        "one-cb x{copies}  [{n:6} x {k}]  {:8.2} ms/matrix  {:6.1} GB/s effective",
        per_call * 1e3,
        gb / per_call,
    );
}
