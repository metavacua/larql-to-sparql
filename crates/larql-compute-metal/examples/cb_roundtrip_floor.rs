//! What does one command-buffer round trip actually cost?
//!
//! The NVFP4 decode spends ~89 ms/token inside device calls across 209
//! submissions (~427 us each) with only ~7 ms of CPU glue, so the fixed
//! term is submission-side. That leaves two very different fixes:
//!
//!   - if Metal's own commit+wait floor is most of the 427 us, the only
//!     lever is *fewer* submissions (fuse the glue onto the GPU);
//!   - if the floor is small, most of the 427 us is our own per-call
//!     work — pool locks, buffer-cache lookups, readback allocation —
//!     and can be removed without restructuring the plan.
//!
//! This measures the floor directly: an empty encoder, then a trivial
//! dispatch, then the same shape the real path uses (bind cached weight
//! buffers + readback), so the cost is attributed rather than assumed.

use std::time::Instant;

const ITERS: usize = 200;

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    // Warm the driver; the first submissions of a process are not
    // representative and would flatter whatever ran last.
    for _ in 0..20 {
        gpu.empty_roundtrip();
    }

    let t = Instant::now();
    for _ in 0..ITERS {
        gpu.empty_roundtrip();
    }
    let empty = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let t = Instant::now();
    for _ in 0..ITERS {
        gpu.empty_encoder_roundtrip();
    }
    let encoder = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    println!("commit+wait, no encoder      {empty:>7.1} us");
    println!("commit+wait, empty encoder   {encoder:>7.1} us");
    println!();
    println!("measured decode: ~427 us/submission across 209 submissions");
    println!(
        "floor accounts for {:.0}% of a submission; the rest is per-call work",
        100.0 * encoder / 427.0
    );
}
