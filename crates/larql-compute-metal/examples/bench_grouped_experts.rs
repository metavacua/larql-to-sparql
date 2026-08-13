//! K3a acceptance bench — does grouping the routed experts repair occupancy?
//!
//! `diag_shape_census` measured the identical `q6k_matvec` kernel at 0.87-0.90
//! of roofline on K3's large shapes but **0.64** on the expert branch
//! `[3584, 3072]`. At `ROWS_PER_TG = 4` one expert launches 896 threadgroups;
//! a token selects 16, which together supply 14,336.
//!
//! This measures whether dispatching them together recovers the gap. Both arms
//! read exactly the same bytes and do exactly the same arithmetic — the only
//! difference is how many independent threadgroups the GPU can schedule at once.
//!
//! Protocol matches `diag_shape_census`: batched into one command buffer so
//! per-dispatch overhead is amortised rather than measured, and inherently
//! cold — 16 distinct expert payloads at 9 MB each is 144 MB per pass, far
//! beyond the 16 MB L2, so nothing is served from cache.
//!
//! Pre-registered bars (DEC-8.8 follow-on):
//!   - exact agreement with per-expert dispatches (checked in unit tests);
//!   - eta rises from 0.64 toward at least 0.80;
//!   - ideally reaches the large-shape band, 0.87-0.90;
//!   - physical bytes unchanged — no padding, no sparsity, no repacking.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example bench_grouped_experts

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
    use larql_compute_metal::MetalBackend;

    const N: usize = 3584; // K3 expert branch rows
    const K: usize = 3072; // latent reduction
    const TOP_K: usize = 16;
    const ROOFLINE: f64 = 367.0;
    const BATCH: usize = 8; // dispatches per command buffer
    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    let Some(be) = MetalBackend::new() else {
        eprintln!("Metal unavailable");
        return;
    };

    println!("K3a — grouped routed-expert dispatch, [{N}, {K}] x {TOP_K} experts");
    println!("=================================================================");

    // One payload per selected expert, distinct so nothing is cache-resident.
    let mut bank = Vec::new();
    let mut offsets = Vec::new();
    let mut per_expert = 0usize;
    for e in 0..TOP_K {
        let w: Vec<f32> = (0..N * K)
            .map(|i| ((((i * 31 + e * 7919) % 251) as f32) - 125.0) * 0.001)
            .collect();
        let q = quantize_q6_k(&w);
        per_expert = q.len();
        offsets.push(ExpertOffset(bank.len() as u32));
        bank.extend_from_slice(&q);
    }
    let total_bytes = (per_expert * TOP_K) as f64;
    let x: Vec<f32> = (0..K).map(|i| ((i % 251) as f32 - 125.0) * 0.01).collect();

    let (single_tgs, grouped_tgs) = be.grouped_threadgroups(N, TOP_K);
    println!(
        "  {:.2} MB/expert, {:.1} MB per token-layer (working set >> 16 MB L2)",
        per_expert as f64 / 1e6,
        total_bytes / 1e6
    );
    println!(
        "  threadgroups: {single_tgs} ungrouped -> {grouped_tgs} grouped ({}x)",
        grouped_tgs / single_tgs
    );
    println!();

    // Both arms batched into one command buffer, so the ratio isolates
    // occupancy from commit+wait amortisation. A naive bench commits once per
    // expert ungrouped and once total grouped, which mixes scheduling with the
    // removal of 15 commit+waits and overstates the gain.
    let (ung_gbs, grp_gbs) = larql_compute_metal::diag::kernel_profile::profile_grouped_experts(
        N, K, TOP_K, BATCH, WARMUP, ITERS,
    );
    let eta_ungrouped = ung_gbs / ROOFLINE;
    let eta_grouped = grp_gbs / ROOFLINE;
    println!(
        "  {:<36} {:>7.1} GB/s  eta {:>5.2}",
        "ungrouped (batched, 16 dispatches)", ung_gbs, eta_ungrouped
    );
    println!(
        "  {:<36} {:>7.1} GB/s  eta {:>5.2}",
        "grouped   (batched, 1 dispatch)", grp_gbs, eta_grouped
    );
    let _ = (&bank, &offsets, &x, total_bytes, per_expert);

    println!();
    println!("--- verdict against the pre-registered bars ---");
    println!(
        "  ungrouped eta {eta_ungrouped:.2}  ->  grouped eta {eta_grouped:.2}   ({:.2}x)",
        eta_grouped / eta_ungrouped
    );
    let verdict = if eta_grouped >= 0.87 {
        "PASS (stretch) — reaches the large-shape band; occupancy fully explains the 0.64"
    } else if eta_grouped >= 0.80 {
        "PASS — clears the 0.80 bar; occupancy explains most of the gap"
    } else if eta_grouped > eta_ungrouped * 1.1 {
        "PARTIAL — grouping helps but something else also binds; profile before K3b"
    } else {
        "FAIL — occupancy was not the cause; do not proceed to K3b on this theory"
    };
    println!("  {verdict}");
    println!();
    println!("  Bytes read are identical in both arms — no padding, no sparsity,");
    println!("  no repacking. Any gain is scheduling, which is what was claimed.");
    if eta_grouped >= 0.80 {
        println!();
        println!("  Feed the measured eta to the composed ledger:");
        println!("    larql k3-ledger ceilings   (routed-expert class is currently 0.64)");
    }
}
